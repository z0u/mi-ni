#!/usr/bin/env python
"""Copy a mi-ni project's Hugging Face storage into one backup dataset repo, never deleting.

This is the payload half of the backup runbook (the ``backup`` skill in the mi-ni template). It runs inside the *backup* repo's nightly GitHub Actions workflow, from a trust domain the project's day-to-day tokens cannot reach, and it copies two Hugging Face sources into one git-backed dataset repo:

- ``store/`` mirrors the CAS bucket. The bucket is write-once-by-hash, so every file the backup lacks is the whole delta; ``refs/`` (mutable name → artifact pointers) is refreshed every run, and its earlier versions live on in the backup's history.
- ``pub/`` replays the publish repo's history, oldest commit first, one backup commit per source commit, so every revision a ``publish.lock`` has ever pinned stays recoverable. The head of ``pub/`` is the *union* of every revision replayed: a file the source later deleted is still there, and a restore of one pinned revision reads the backup commit that replayed it.

Two rules keep the copy trustworthy. It never deletes: no ``delete_patterns``, no ``super_squash_history``, nothing removed on either side. And it never runs code from the sources: this file and the workflow live in the backup repo, copied once from the template.

Configuration is by environment variable — ``SOURCE_BUCKET``, ``SOURCE_PUBLISH_REPO``, ``BACKUP_DATASET`` (all ``namespace/name``) and ``HF_TOKEN`` (read on the sources, write on the backup, nothing else) — so the workflow file is the one place the names appear. The code leg (mirroring the GitHub repo) is plain git and lives in the workflow; it hands its findings in through ``--code-sha`` / ``--snapshot`` so one ``state/last-run.json`` records the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STORE_PREFIX = "store"
PUB_PREFIX = "pub"
# The last source commit replayed into ``pub/``, kept *in* the backup dataset so the
# marker and the files it describes land in one commit. Also what makes every replay a
# real commit: a source commit that only deleted files would otherwise be a no-op
# upload, and the run would replay it again every night.
PUB_MARKER = f"{PUB_PREFIX}/SOURCE_COMMIT"
# Never uploaded from the working dir: huggingface_hub's own download metadata.
IGNORE = [".cache/**", ".git/**"]


@dataclass
class LegReport:
    """What one leg saw and did — the run log, and the delta walk's input next time."""

    source: str
    seen: int = 0
    copied: int = 0
    bytes: int = 0
    commits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in ([], 0) or k in ("source", "seen")}


# -- store bucket → store/ ------------------------------------------------------------


def backup_files(api: Any, dataset: str, prefix: str) -> set[str]:
    """Paths (relative to *prefix*) the backup dataset already holds under it; empty if the prefix is new.

    A backup dataset that doesn't exist yet reads as empty too, so a ``--dry-run`` can size the first copy before the repo is created; a real run then fails at its first upload, with the 404 naming the repo.
    """
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        entries = api.list_repo_tree(dataset, path_in_repo=prefix, recursive=True, repo_type="dataset")
        return {e.path[len(prefix) + 1 :] for e in entries if getattr(e, "type", None) == "file"}
    except EntryNotFoundError, RepositoryNotFoundError:
        return set()


def store_delta(entries: Iterable[Any], have: set[str]) -> list[Any]:
    """The bucket files to copy this run: everything the backup lacks, plus every ref."""
    return [e for e in entries if e.path not in have or e.path.startswith("refs/")]


def batches(entries: list[Any], limit: int) -> Iterable[list[Any]]:
    """Split the delta into runs of at most *limit* bytes (one commit each), so a large first run fits the runner's disk."""
    batch: list[Any] = []
    size = 0
    for e in entries:
        if batch and size + e.size > limit:
            yield batch
            batch, size = [], 0
        batch.append(e)
        size += e.size
    if batch:
        yield batch


def backup_store(api: Any, bucket: str, dataset: str, work: Path, *, batch_bytes: int, dry_run: bool) -> LegReport:
    report = LegReport(source=bucket)
    entries = [e for e in api.list_bucket_tree(bucket, recursive=True) if getattr(e, "type", None) == "file"]
    report.seen = len(entries)
    delta = store_delta(entries, backup_files(api, dataset, STORE_PREFIX))
    report.copied, report.bytes = len(delta), sum(e.size for e in delta)
    if dry_run or not delta:
        return report
    stage = work / STORE_PREFIX
    for batch in batches(delta, batch_bytes):
        shutil.rmtree(stage, ignore_errors=True)
        files = []
        for e in batch:
            dest = stage / e.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            files.append((e, str(dest)))
        api.download_bucket_files(bucket, files=files, raise_on_missing_files=True)
        info = api.upload_folder(
            repo_id=dataset,
            repo_type="dataset",
            folder_path=str(stage),
            path_in_repo=STORE_PREFIX,
            ignore_patterns=IGNORE,
            commit_message=f"store: {len(batch)} file(s) from {bucket}",
        )
        report.commits.append(info.oid)
    shutil.rmtree(stage, ignore_errors=True)
    return report


# -- publish repo → pub/ -------------------------------------------------------------


def last_replayed(api: Any, dataset: str) -> str | None:
    """The source sha the backup last replayed, from :data:`PUB_MARKER`; ``None`` before the first replay."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = api.hf_hub_download(dataset, PUB_MARKER, repo_type="dataset")
    except EntryNotFoundError, RepositoryNotFoundError:
        return None
    return Path(path).read_text().strip() or None


def pending_commits(history: list[Any], marker: str | None) -> list[Any]:
    """Source commits not yet replayed, oldest first.

    *history* is newest-first, as ``list_repo_commits`` returns it. A marker that names no commit in it means the source's history was rewritten since the last run — the incident a backup exists for — so that raises rather than quietly replaying a rewritten past over the record of the real one.
    """
    oldest_first = list(reversed(history))
    if marker is None:
        return oldest_first
    ids = [c.commit_id for c in oldest_first]
    if marker not in ids:
        raise RuntimeError(
            f"the backup last replayed {marker}, which is no longer in the source's history — "
            "the publish repo's history was rewritten. Inspect before replaying anything further."
        )
    return oldest_first[ids.index(marker) + 1 :]


def backup_publish(api: Any, source: str, dataset: str, work: Path, *, max_commits: int, dry_run: bool) -> LegReport:
    report = LegReport(source=source)
    history = api.list_repo_commits(source, repo_type="dataset")
    report.seen = len(history)
    todo = pending_commits(history, last_replayed(api, dataset))
    if len(todo) > max_commits:
        report.notes.append(
            f"{len(todo) - max_commits} commit(s) deferred to a later run (--max-commits {max_commits})"
        )
        todo = todo[:max_commits]
    report.copied = len(todo)
    if dry_run:
        return report
    stage = work / PUB_PREFIX
    for commit in todo:
        # One local dir across revisions, so the upload of each is only what changed —
        # and so a file the source deleted stays (the union head; see the module doc).
        stage.mkdir(parents=True, exist_ok=True)  # a revision with no files still gets its marker
        api.snapshot_download(source, repo_type="dataset", revision=commit.commit_id, local_dir=str(stage))
        (stage / Path(PUB_MARKER).name).write_text(commit.commit_id + "\n")
        info = api.upload_folder(
            repo_id=dataset,
            repo_type="dataset",
            folder_path=str(stage),
            path_in_repo=PUB_PREFIX,
            ignore_patterns=IGNORE,
            commit_message=f"pub: replay {commit.commit_id}",
            commit_description=f"{source} — {commit.title}",
        )
        report.commits.append(info.oid)
    shutil.rmtree(stage, ignore_errors=True)
    return report


# -- the run -------------------------------------------------------------------------


def require(name: str) -> str:
    if not (value := os.environ.get(name)):
        sys.exit(f"{name} is not set — the workflow's env block names the source and backup repos")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", type=Path, default=Path("state/last-run.json"), help="where to write the run record")
    ap.add_argument("--work", type=Path, default=Path(".backup-work"), help="scratch dir for downloads (deleted)")
    ap.add_argument("--code-sha", default=None, help="the source tip the code leg fetched (recorded only)")
    ap.add_argument("--snapshot", default=None, help="the snapshot tag the code leg created, if any (recorded only)")
    ap.add_argument("--max-commits", type=int, default=200, help="publish-repo commits to replay per run")
    ap.add_argument("--batch-bytes", type=int, default=2 << 30, help="bytes of bucket files per backup commit")
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied; upload nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    bucket, pub, dataset = require("SOURCE_BUCKET"), require("SOURCE_PUBLISH_REPO"), require("BACKUP_DATASET")
    started = datetime.now(timezone.utc)
    state: dict[str, Any] = {
        "ran_at": started.isoformat(timespec="seconds"),
        "backup": dataset,
        "dry_run": args.dry_run,
    }
    if args.code_sha:
        state["code"] = {"sha": args.code_sha, **({"snapshot": args.snapshot} if args.snapshot else {})}
    errors: list[str] = []
    for leg, run in (
        (
            "store",
            lambda: backup_store(api, bucket, dataset, args.work, batch_bytes=args.batch_bytes, dry_run=args.dry_run),
        ),
        (
            "pub",
            lambda: backup_publish(api, pub, dataset, args.work, max_commits=args.max_commits, dry_run=args.dry_run),
        ),
    ):
        try:
            report = run()
        except (
            Exception
        ) as e:  # keep going: the other leg's copy is still worth taking, and the record says what failed
            errors.append(f"{leg}: {e}")
            state[leg] = {"error": str(e)}
            continue
        state[leg] = report.to_dict()
        print(
            f"{leg}: {report.copied} of {report.seen} copied, {len(report.commits)} commit(s)"
            + "".join(f"; {n}" for n in report.notes)
        )
    if errors:
        state["errors"] = errors
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=1) + "\n")
    shutil.rmtree(args.work, ignore_errors=True)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
