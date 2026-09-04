#!/usr/bin/env python
"""Copy a mi-ni project's code and Hugging Face storage into backup repos, never deleting.

This is the payload half of the backup runbook (the ``backup`` skill in the mi-ni template). It runs inside the *backup* repo's nightly GitHub Actions workflow, from a trust domain the project's day-to-day tokens cannot reach. Three legs:

- Code. Fetch the source repo's ``main``, fast-forward this repo's ``mirror`` branch to it, tag ``snap/<date>`` when the tip moved since the last run, and carry the source's own tags under ``source/``. A push that isn't a fast-forward is refused, never forced: the source rewrote its history, which the run reports and the snapshot records regardless.
- ``store/`` in the backup dataset mirrors the CAS bucket. The bucket is write-once-by-hash, so every file the backup lacks is the whole delta; ``refs/`` (mutable name → artifact pointers) is refreshed every run, and its earlier versions live on in the backup's history.
- ``pub/`` replays the publish repo's history, oldest commit first, one backup commit per source commit, so every revision a ``publish.lock`` has ever pinned stays recoverable. The head of ``pub/`` is the *union* of every revision replayed: a file the source later deleted is still there, and a restore of one pinned revision reads the backup commit that replayed it.

Two rules keep the copy trustworthy. It never deletes: no ``delete_patterns``, no ``super_squash_history``, no forced push, nothing removed on either side. And it never runs code from the sources: this file and the workflow live in the backup repo, copied once from the template.

Configuration is by environment variable — ``SOURCE_REPO`` (``owner/name`` on GitHub, or any git URL), ``SOURCE_BUCKET``, ``SOURCE_PUBLISH_REPO``, ``BACKUP_DATASET`` (all ``namespace/name`` on Hugging Face) — so the workflow file is the one place the names appear.

Credentials come in two kinds, because the sources and the backup belong to different accounts and no one token spans both. The sources are read with ``SOURCE_HF_TOKEN`` (the bucket and publish repo) and ``SOURCE_GH_TOKEN`` (the repo), both read-only and both unset for public sources, where reads are anonymous. The backup dataset is written with ``HF_TOKEN``, which the workflow mints per run from the dataset's trusted publisher or takes from a stored secret. The source clients never fall back to ``HF_TOKEN``, so the write credential is only ever sent to the backup. Git pushes use whatever credentials the checkout already has (the job's ``GITHUB_TOKEN``). One ``state/last-run.json`` records the whole run, and the previous run's record is where the code leg reads the last tip from.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
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
MIRROR = "refs/heads/mirror"
SOURCE_TAGS = "refs/tags/source/*"


def _trim(d: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in ([], 0, None) or k in keep}


@dataclass
class LegReport:
    """What one Hugging Face leg saw and did — the run log, and the delta walk's input next time."""

    source: str
    seen: int = 0
    copied: int = 0
    bytes: int = 0
    commits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _trim(self.__dict__, ("source", "seen"))


@dataclass
class CodeReport:
    source: str
    sha: str
    snapshot: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _trim(self.__dict__, ("source", "sha"))


def warn(report: LegReport | CodeReport, message: str) -> None:
    """Record a note and surface it as a GitHub Actions annotation."""
    report.notes.append(message)
    print(f"::warning::{message}")


# -- code: source repo → mirror branch + snapshot tags ---------------------------------


def git(
    *args: str, cwd: Path, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check, env={**os.environ, **(env or {})}
    )


def auth_env(url: str, token: str | None) -> dict[str, str]:
    """Environment that makes git send *token* on requests to *url*'s host, for one command only.

    The token travels as an ``AUTHORIZATION`` header set through ``GIT_CONFIG_*``, so it never appears in a URL, a command line, or an error message. The empty first value resets any header the checkout already configured for that host (``actions/checkout`` stores the job's token that way), because a request with two ``AUTHORIZATION`` headers is refused. Only the source fetches get this environment; pushes to ``origin`` keep the checkout's own credential. Empty for a token-less or non-HTTP source.
    """
    if not token or not url.startswith(("http://", "https://")):
        return {}
    scheme, _, host = url.partition("://")
    key = f"http.{scheme}://{host.split('/', 1)[0]}/.extraheader"
    cred = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": key,
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": key,
        "GIT_CONFIG_VALUE_1": f"AUTHORIZATION: basic {cred}",
    }


def snapshot_name(repo: Path, now: datetime) -> str:
    """``snap/<date>``, or a time-suffixed name if that tag already exists on the remote (a second run in one day)."""
    tag = f"snap/{now:%Y-%m-%d}"
    if git("ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}", cwd=repo, check=False).returncode == 0:
        tag = f"snap/{now:%Y-%m-%dT%H%MZ}"
    return tag


def backup_code(
    source_url: str,
    repo: Path,
    *,
    prev_sha: str | None,
    branch: str = "main",
    dry_run: bool,
    token: str | None = None,
) -> CodeReport:
    """Mirror *source_url*'s *branch* into *repo*'s ``origin``: fast-forward ``mirror``, tag a snapshot if the tip moved, copy the source's tags.

    *repo* is a checkout of the backup repo whose ``origin`` is writable (in Actions, the job's own token). *token* is read access to a private source, sent on the fetches only. Nothing here forces: a mirror or tag push that isn't a fast-forward is refused by git and reported.
    """
    if git("remote", "get-url", "source", cwd=repo, check=False).returncode == 0:
        git("remote", "set-url", "source", source_url, cwd=repo)
    else:
        git("remote", "add", "source", source_url, cwd=repo)
    auth = auth_env(source_url, token)
    # The branch is fetched forced, so a rewritten source is *seen* (and snapshotted); the
    # pushes below are what refuse to let it overwrite anything already recorded.
    git("fetch", "--no-tags", "source", f"+refs/heads/{branch}:refs/remotes/source/{branch}", cwd=repo, env=auth)
    tags = git("fetch", "--no-tags", "source", f"refs/tags/*:{SOURCE_TAGS}", cwd=repo, check=False, env=auth)
    tip = git("rev-parse", f"refs/remotes/source/{branch}", cwd=repo).stdout.strip()
    report = CodeReport(source=source_url, sha=tip)
    if tags.returncode != 0:
        warn(report, "some source tags moved since the backup first saw them; the earlier pointers stand")
    if tip != prev_sha:
        report.snapshot = snapshot_name(repo, datetime.now(timezone.utc))
    if dry_run:
        return report

    if git("push", "origin", f"{tip}:{MIRROR}", cwd=repo, check=False).returncode != 0:
        warn(
            report,
            f"mirror not fast-forwarded: source {branch} at {tip} is not a descendant of the mirror — "
            "the source's history was rewritten. Snapshot recorded regardless.",
        )
    if report.snapshot:
        git("tag", report.snapshot, tip, cwd=repo)
        git("push", "origin", f"refs/tags/{report.snapshot}", cwd=repo)
    if git("push", "origin", f"{SOURCE_TAGS}:{SOURCE_TAGS}", cwd=repo, check=False).returncode != 0:
        warn(report, "some source tags were not updated (a moved tag stays where the backup first saw it)")
    return report


# -- store bucket → store/ ------------------------------------------------------------


def backup_files(api: Any, dataset: str, prefix: str) -> set[str]:
    """Paths (relative to *prefix*) the backup dataset already holds under it; empty if the prefix is new.

    A backup dataset that doesn't exist yet reads as empty too, so a ``--dry-run`` can size the first copy before the repo is created; a real run then fails at its first upload, with the 404 naming the repo.
    """
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        entries = api.list_repo_tree(dataset, path_in_repo=prefix, recursive=True, repo_type="dataset")
        return {e.path[len(prefix) + 1 :] for e in entries if getattr(e, "type", None) == "file"}
    except (EntryNotFoundError, RepositoryNotFoundError):
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


def backup_store(
    source_api: Any, backup_api: Any, bucket: str, dataset: str, work: Path, *, batch_bytes: int, dry_run: bool
) -> LegReport:
    """Copy the bucket's delta into ``store/``. *source_api* reads the bucket; *backup_api* reads and writes the dataset."""
    report = LegReport(source=bucket)
    entries = [e for e in source_api.list_bucket_tree(bucket, recursive=True) if getattr(e, "type", None) == "file"]
    report.seen = len(entries)
    delta = store_delta(entries, backup_files(backup_api, dataset, STORE_PREFIX))
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
        source_api.download_bucket_files(bucket, files=files, raise_on_missing_files=True)
        info = backup_api.upload_folder(
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
    except (EntryNotFoundError, RepositoryNotFoundError):
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


def backup_publish(
    source_api: Any, backup_api: Any, source: str, dataset: str, work: Path, *, max_commits: int, dry_run: bool
) -> LegReport:
    """Replay the publish repo's history into ``pub/``. *source_api* reads the publish repo; *backup_api* the dataset."""
    report = LegReport(source=source)
    history = source_api.list_repo_commits(source, repo_type="dataset")
    report.seen = len(history)
    todo = pending_commits(history, last_replayed(backup_api, dataset))
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
        source_api.snapshot_download(source, repo_type="dataset", revision=commit.commit_id, local_dir=str(stage))
        (stage / Path(PUB_MARKER).name).write_text(commit.commit_id + "\n")
        info = backup_api.upload_folder(
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
        sys.exit(f"{name} is not set — fill in the workflow's env block")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--state",
        type=Path,
        default=Path("state/last-run.json"),
        help="the run record: read for the last tip, then rewritten",
    )
    ap.add_argument(
        "--repo", type=Path, default=Path("."), help="checkout of the backup repo; its origin receives the code leg"
    )
    ap.add_argument("--work", type=Path, default=Path(".backup-work"), help="scratch dir for downloads (deleted)")
    ap.add_argument("--source-branch", default="main", help="the source branch to mirror")
    ap.add_argument("--max-commits", type=int, default=200, help="publish-repo commits to replay per run")
    ap.add_argument("--batch-bytes", type=int, default=2 << 30, help="bytes of bucket files per backup commit")
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied; push and upload nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    code, bucket, pub, dataset = (
        require("SOURCE_REPO"),
        require("SOURCE_BUCKET"),
        require("SOURCE_PUBLISH_REPO"),
        require("BACKUP_DATASET"),
    )
    write_token = os.environ.get("HF_TOKEN")
    if not write_token and not args.dry_run:
        sys.exit("HF_TOKEN is not set — the workflow mints it per run (trusted publisher) or takes it from a secret")
    # ``token=False`` is anonymous. ``None`` would fall back to HF_TOKEN, sending the backup's
    # write credential to the sources; the read clients must never do that.
    source_api = HfApi(token=os.environ.get("SOURCE_HF_TOKEN") or False)
    backup_api = HfApi(token=write_token or False)
    prev = json.loads(args.state.read_text()) if args.state.is_file() else {}
    prev_sha = prev.get("code", {}).get("sha")
    state: dict[str, Any] = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backup": dataset,
        "dry_run": args.dry_run,
    }
    errors: list[str] = []
    legs = (
        (
            "code",
            lambda: backup_code(
                code if "://" in code or code.startswith("/") else f"https://github.com/{code}.git",
                args.repo,
                prev_sha=prev_sha,
                branch=args.source_branch,
                dry_run=args.dry_run,
                token=os.environ.get("SOURCE_GH_TOKEN"),
            ),
        ),
        (
            "store",
            lambda: backup_store(
                source_api,
                backup_api,
                bucket,
                dataset,
                args.work,
                batch_bytes=args.batch_bytes,
                dry_run=args.dry_run,
            ),
        ),
        (
            "pub",
            lambda: backup_publish(
                source_api,
                backup_api,
                pub,
                dataset,
                args.work,
                max_commits=args.max_commits,
                dry_run=args.dry_run,
            ),
        ),
    )
    for leg, run in legs:
        try:
            report = run()
        except Exception as e:  # keep going: the other legs are still worth taking, and the record says what failed
            detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
            errors.append(f"{leg}: {detail}")
            state[leg] = {"error": detail}
            continue
        state[leg] = report.to_dict()
        summary = (
            report.sha[:12] + (f", snapshot {report.snapshot}" if report.snapshot else "")
            if isinstance(report, CodeReport)
            else f"{report.copied} of {report.seen} copied, {len(report.commits)} commit(s)"
        )
        print(f"{leg}: {summary}" + "".join(f"; {n}" for n in report.notes))
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
