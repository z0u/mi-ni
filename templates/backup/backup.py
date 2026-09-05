#!/usr/bin/env python
"""Copy a mi-ni project's code and Hugging Face storage into backup repos, without ever deleting what the source still has.

This is the payload half of the backup runbook (the ``backup`` skill in the mi-ni template). It runs inside the *backup* repo's nightly GitHub Actions workflow, from a trust domain the project's day-to-day tokens cannot reach. Three legs, three targets: a *mirror repo* on GitHub (browsable history), a backup *dataset* on the Hub (git-backed, so it keeps history) and a backup *bucket* (mutable, so it can forget).

- Code. Fetch the source repo's ``main``, fast-forward the ``mirror`` branch of the *mirror repo* (a second GitHub repo under the backup account) to it, tag ``snap/<date>`` there when the tip differs from what the mirror already holds, and carry the source's own tags under ``source/``. A push that isn't a fast-forward is refused, never forced: the source rewrote its history, which the run reports and the snapshot records regardless. The push uses a fine-grained token with contents and workflows write on the mirror repo alone, because the job's own token may not write workflow files and the source's commits contain them. A token that can push workflow files also lets the mirrored workflows *run* on push, so the mirror repo has GitHub Actions disabled, and the leg checks that setting through the API before every push and refuses if it is on.
- Store. The CAS bucket is copied into the backup bucket server-side, by Xet hash, so no bytes pass through the runner. The bucket is write-once-by-hash, so the files the backup lacks are the whole delta; ``refs/`` (mutable name → artifact pointers) is re-copied when its hash changed, and a copy of ``refs/`` also goes into the dataset under ``store/refs/`` so earlier pointers stay recoverable. The source bucket is meant to shrink (``mini gc --store``), so the backup bucket follows it with a delay: a file that has been gone from the source for longer than ``--retain-days`` is deleted from the backup, and the date each file was first missed is kept in ``state/store-missing.json``.
- ``pub/`` in the dataset replays the publish repo's history, oldest commit first, one backup commit per source commit, so every revision a ``publish.lock`` has ever pinned stays recoverable. The head of ``pub/`` is the *union* of every revision replayed: a file the source later deleted is still there, and a restore of one pinned revision reads the backup commit that replayed it.

Two rules keep the copy trustworthy. It never deletes anything the source still has, and nothing sooner than the retention window: no ``delete_patterns``, no ``super_squash_history``, no forced push, and the one delete there is runs on a clock the source can only start, never hurry. And it never runs code from the sources: this file and the workflow live in the backup repo, copied once from the template.

Configuration is by environment variable — ``SOURCE_REPO`` and ``MIRROR_REPO`` (``owner/name`` on GitHub, or any git URL), ``SOURCE_BUCKET``, ``SOURCE_PUBLISH_REPO``, ``BACKUP_DATASET``, ``BACKUP_BUCKET`` (all ``namespace/name`` on Hugging Face) — so the workflow file is the one place the names appear.

Credentials come in two kinds, because the sources and the backup belong to different accounts and no one token spans both. The sources are read with ``SOURCE_HF_TOKEN`` (the bucket and publish repo) and ``SOURCE_GH_TOKEN`` (the repo), both read-only and both unset for public sources, where reads are anonymous. The backup dataset is written with ``HF_TOKEN`` and the backup bucket with ``HF_BUCKET_TOKEN`` (falling back to ``HF_TOKEN``); the workflow mints each per run from the target's trusted publisher, or takes one stored secret that covers both. The mirror repo is pushed with ``MIRROR_GH_TOKEN``. The source clients never fall back to a write token, so the write credentials are only ever sent to the backup, and each token travels as a header, never in a URL. One ``state/last-run.json`` records the whole run, and the previous run's record is where the code leg reads the last tip from.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STORE_PREFIX = "store"
REFS_PREFIX = "refs"
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
# Bucket operations per API call. A batch is not transactional, but every operation here
# is idempotent, so a batch that fails halfway is simply finished by the next run.
BUCKET_BATCH = 200


def _trim(d: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in ([], 0, None) or k in keep}


@dataclass
class LegReport:
    """What one Hugging Face leg saw and did — the run log, and the delta walk's input next time."""

    source: str
    seen: int = 0
    copied: int = 0
    bytes: int = 0
    missing: int = 0  # gone from the source, kept in the backup until the retention window ends
    expired: int = 0  # deleted from the backup this run, the window having ended
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


# -- code: source repo → mirror repo (mirror branch + snapshot tags) ----------------------


def git(
    *args: str, cwd: Path, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check, env={**os.environ, **(env or {})}
    )


def auth_env(url: str, token: str | None) -> dict[str, str]:
    """Environment that makes git send *token* on requests to *url*'s host, for one command only.

    The token travels as an ``AUTHORIZATION`` header set through ``GIT_CONFIG_*``, so it never appears in a URL, a command line, or an error message. The empty first value resets any header the checkout already configured for that host (``actions/checkout`` stores the job's token that way), because a request with two ``AUTHORIZATION`` headers is refused. Empty for a token-less or non-HTTP URL.
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


def github_slug(url: str) -> str | None:
    """``owner/name`` for a github.com URL; ``None`` for anything else (a local path in the tests, another host)."""
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        return None
    slug = url[len(prefix) :].removesuffix(".git").strip("/")
    return slug if slug.count("/") == 1 else None


def actions_enabled(slug: str, token: str | None) -> bool:
    """Whether GitHub Actions is switched on for the repo *slug*, per the REST API.

    Needs *token* to hold *Administration: read* on that repo. Any failure to get an answer raises, because the answer gates a push: the leg would rather not push than push into a repo where the mirrored workflows might run.
    """
    req = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/actions/permissions",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token or ''}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as res:
            return bool(json.load(res)["enabled"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"could not read the Actions setting of {slug} ({e.code}): the mirror token needs Administration read "
            "on the mirror repo, and the leg does not push until it can see that Actions is disabled there"
        ) from e


def check_mirror(url: str, token: str | None) -> None:
    """Refuse to push into a GitHub mirror repo that could run what is pushed.

    Pushes made with a personal access token trigger workflows, unlike pushes made with the job's own token, and the mirrored commits carry the source's workflow files. Disabling Actions on the mirror repo is what makes the push safe, so a mirror on github.com must have it off; a local or foreign-host mirror is left to its owner.
    """
    slug = github_slug(url)
    if slug and actions_enabled(slug, token):
        raise RuntimeError(
            f"GitHub Actions is enabled on {slug}: a push there could run the source's workflows. "
            "Disable it (Settings → Actions → General → Disable actions) and re-run."
        )


def remote_refs(repo: Path, remote: str, env: dict[str, str]) -> dict[str, str]:
    """Every ref *remote* holds, name → sha. Empty for an empty repo."""
    out = git("ls-remote", remote, cwd=repo, env=env).stdout
    return {name: sha for sha, name in (line.split("\t") for line in out.splitlines())}


def snapshot_name(have: dict[str, str], now: datetime) -> str:
    """``snap/<date>``, or a time-suffixed name if the mirror (*have*, its refs) already has that tag: a second run in one day."""
    tag = f"snap/{now:%Y-%m-%d}"
    if f"refs/tags/{tag}" in have:
        tag = f"snap/{now:%Y-%m-%dT%H%MZ}"
    return tag


def backup_code(
    source_url: str,
    mirror_url: str,
    repo: Path,
    *,
    branch: str = "main",
    dry_run: bool,
    source_token: str | None = None,
    mirror_token: str | None = None,
) -> CodeReport:
    """Mirror *source_url*'s *branch* into *mirror_url*: fast-forward ``mirror``, tag a snapshot if the tip moved, copy the source's tags.

    The mirror repo is the record of what was backed up: the tip has moved if it differs from the mirror's ``mirror`` branch, so a lost or reset state file costs nothing here. *repo* is any local git repo, scratch as far as this leg is concerned; it holds the fetched objects between the fetch and the push. *source_token* is read access to a private source, sent on the fetches only; *mirror_token* is write access to the mirror repo, sent on the pushes only. Nothing here forces: a mirror or tag push that isn't a fast-forward is refused by git and reported.
    """
    for name, url in (("source", source_url), ("mirror", mirror_url)):
        if git("remote", "get-url", name, cwd=repo, check=False).returncode == 0:
            git("remote", "set-url", name, url, cwd=repo)
        else:
            git("remote", "add", name, url, cwd=repo)
    read, write = auth_env(source_url, source_token), auth_env(mirror_url, mirror_token)
    # The branch is fetched forced, so a rewritten source is *seen* (and snapshotted); the
    # pushes below are what refuse to let it overwrite anything already recorded.
    git("fetch", "--no-tags", "source", f"+refs/heads/{branch}:refs/remotes/source/{branch}", cwd=repo, env=read)
    tags = git("fetch", "--no-tags", "source", f"refs/tags/*:{SOURCE_TAGS}", cwd=repo, check=False, env=read)
    tip = git("rev-parse", f"refs/remotes/source/{branch}", cwd=repo).stdout.strip()
    report = CodeReport(source=source_url, sha=tip)
    if tags.returncode != 0:
        warn(report, "some source tags moved since the backup first saw them; the earlier pointers stand")
    have = remote_refs(repo, "mirror", write)
    if tip != have.get(MIRROR):
        report.snapshot = snapshot_name(have, datetime.now(timezone.utc))
    if dry_run:
        return report

    check_mirror(mirror_url, mirror_token)
    if git("push", "mirror", f"{tip}:{MIRROR}", cwd=repo, check=False, env=write).returncode != 0:
        warn(
            report,
            f"mirror not fast-forwarded: source {branch} at {tip} is not a descendant of the mirror — "
            "the source's history was rewritten. Snapshot recorded regardless.",
        )
    if report.snapshot:
        git("tag", report.snapshot, tip, cwd=repo)
        git("push", "mirror", f"refs/tags/{report.snapshot}", cwd=repo, env=write)
    if git("push", "mirror", f"{SOURCE_TAGS}:{SOURCE_TAGS}", cwd=repo, check=False, env=write).returncode != 0:
        warn(report, "some source tags were not updated (a moved tag stays where the backup first saw it)")
    return report


# -- store bucket → backup bucket (server-side), refs → store/refs/ in the dataset ----------


def bucket_files(api: Any, bucket: str) -> dict[str, Any]:
    """``path → entry`` for every file in *bucket*; empty for a bucket that doesn't exist yet, so a ``--dry-run`` can size the first copy before the backup bucket is created."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return {e.path: e for e in api.list_bucket_tree(bucket, recursive=True) if getattr(e, "type", None) == "file"}
    except RepositoryNotFoundError:
        return {}


def bucket_delta(source: dict[str, Any], have: dict[str, Any]) -> list[Any]:
    """The files to copy this run: everything the backup lacks, and everything whose content changed (the refs)."""
    return [e for p, e in source.items() if p not in have or have[p].xet_hash != e.xet_hash]


def retention(
    missing: dict[str, str], source: set[str], have: set[str], *, today: date, retain_days: int
) -> tuple[dict[str, str], list[str]]:
    """Advance the retention clock: ``(still missing → date first missed, paths to delete now)``.

    A path is *missing* when the backup has it and the source doesn't. It keeps the date it was first missed until it either returns to the source (it drops off the list) or has been gone for more than *retain_days* (it is deleted). The source can start a clock, by deleting a file, but nothing it does can shorten one: a lost manifest only delays every expiry, and a mass deletion upstream gives the whole window to notice.
    """
    gone = have - source
    kept = {p: missing.get(p, today.isoformat()) for p in gone}
    expired = sorted(p for p, first in kept.items() if (today - date.fromisoformat(first)).days > retain_days)
    return {p: d for p, d in kept.items() if p not in expired}, expired


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def backup_store(
    source_api: Any,
    backup_api: Any,
    bucket_api: Any,
    bucket: str,
    dataset: str,
    backup_bucket: str,
    work: Path,
    missing: dict[str, str],
    *,
    retain_days: int,
    dry_run: bool,
) -> tuple[LegReport, dict[str, str]]:
    """Copy the bucket's delta into *backup_bucket* server-side, expire what the source gave up long enough ago, and keep ``refs/`` history in *dataset*.

    *source_api* reads the bucket; *bucket_api* reads and writes the backup bucket; *backup_api* writes the dataset. *missing* is the retention manifest from the last run, and the updated one comes back with the report.
    """
    report = LegReport(source=bucket)
    source = bucket_files(source_api, bucket)
    have = bucket_files(bucket_api, backup_bucket)
    report.seen = len(source)
    delta = bucket_delta(source, have)
    report.copied, report.bytes = len(delta), sum(e.size for e in delta)
    today = datetime.now(timezone.utc).date()
    missing, expired = retention(missing, set(source), set(have), today=today, retain_days=retain_days)
    report.missing, report.expired = len(missing), len(expired)
    newly = sum(1 for d in missing.values() if d == today.isoformat())
    if newly and newly * 2 > len(have):
        warn(report, f"{newly} of {len(have)} backed-up files vanished from the source overnight — inspect the source")
    elif newly:
        report.notes.append(f"{newly} file(s) newly missing from the source, kept for {retain_days} days")
    if dry_run:
        return report, missing

    by_hash = [e for e in delta if e.xet_hash]
    for batch in chunks(by_hash, BUCKET_BATCH):
        bucket_api.batch_bucket_files(backup_bucket, copy=[("bucket", bucket, e.xet_hash, e.path) for e in batch])
    # A file the Hub holds outside Xet (rare in a bucket) has no hash to copy by, so it goes the long way round.
    stage = work / STORE_PREFIX
    for batch in chunks([e for e in delta if not e.xet_hash], BUCKET_BATCH):
        shutil.rmtree(stage, ignore_errors=True)
        files = [(e, str(stage / e.path)) for e in batch]
        for _, dest in files:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
        source_api.download_bucket_files(bucket, files=files, raise_on_missing_files=True)
        bucket_api.batch_bucket_files(backup_bucket, add=[(dest, e.path) for e, dest in files])
    for batch in chunks(expired, BUCKET_BATCH):
        bucket_api.batch_bucket_files(backup_bucket, delete=batch)

    # The refs are the one mutable part of the store, and a bucket forgets an overwritten
    # value at once; the dataset keeps every version, at the cost of moving a few small files.
    refs = [e for e in source.values() if e.path.startswith(f"{REFS_PREFIX}/")]
    if refs:
        shutil.rmtree(stage, ignore_errors=True)
        report.commits += filter(None, [keep_refs(source_api, backup_api, bucket, dataset, refs, stage)])
    shutil.rmtree(stage, ignore_errors=True)
    return report, missing


def keep_refs(source_api: Any, backup_api: Any, bucket: str, dataset: str, refs: list[Any], stage: Path) -> str | None:
    """Commit the bucket's *refs* to ``store/refs/`` in the dataset. The new commit's sha, or None if nothing changed.

    An upload that changes nothing makes no commit, and the Hub answers with the head it already had; only a new sha is this run's.
    """
    files = [(e, str(stage / e.path)) for e in refs]
    for _, dest in files:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
    source_api.download_bucket_files(bucket, files=files, raise_on_missing_files=True)
    head = backup_api.repo_info(dataset, repo_type="dataset").sha
    info = backup_api.upload_folder(
        repo_id=dataset,
        repo_type="dataset",
        folder_path=str(stage / REFS_PREFIX),
        path_in_repo=f"{STORE_PREFIX}/{REFS_PREFIX}",
        ignore_patterns=IGNORE,
        commit_message=f"store: {len(refs)} ref(s) from {bucket}",
    )
    return None if info.oid == head else info.oid


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
        help="the run record: read for the last tip, then rewritten; store-missing.json sits beside it",
    )
    ap.add_argument("--repo", type=Path, default=Path("."), help="a local git repo for the code leg to fetch into")
    ap.add_argument("--work", type=Path, default=Path(".backup-work"), help="scratch dir for downloads (deleted)")
    ap.add_argument("--source-branch", default="main", help="the source branch to mirror")
    ap.add_argument("--max-commits", type=int, default=200, help="publish-repo commits to replay per run")
    ap.add_argument(
        "--retain-days", type=int, default=90, help="days a file stays in the backup bucket after leaving the source"
    )
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied; write nothing anywhere")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    code, bucket, pub, mirror, dataset, backup_bucket = (
        require("SOURCE_REPO"),
        require("SOURCE_BUCKET"),
        require("SOURCE_PUBLISH_REPO"),
        require("MIRROR_REPO"),
        require("BACKUP_DATASET"),
        require("BACKUP_BUCKET"),
    )
    write_token = os.environ.get("HF_TOKEN")
    if not write_token and not args.dry_run:
        sys.exit("HF_TOKEN is not set — the workflow mints it per run (trusted publisher) or takes it from a secret")
    # ``token=False`` is anonymous. ``None`` would fall back to HF_TOKEN, sending the backup's
    # write credential to the sources; the read clients must never do that.
    source_api = HfApi(token=os.environ.get("SOURCE_HF_TOKEN") or False)
    backup_api = HfApi(token=write_token or False)
    bucket_api = HfApi(token=os.environ.get("HF_BUCKET_TOKEN") or write_token or False)
    missing_path = args.state.with_name("store-missing.json")
    missing: dict[str, str] = json.loads(missing_path.read_text()) if missing_path.is_file() else {}
    state: dict[str, Any] = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mirror": mirror,
        "backup": dataset,
        "backup_bucket": backup_bucket,
        "dry_run": args.dry_run,
    }
    errors: list[str] = []

    def store():
        nonlocal missing
        report, missing = backup_store(
            source_api,
            backup_api,
            bucket_api,
            bucket,
            dataset,
            backup_bucket,
            args.work,
            missing,
            retain_days=args.retain_days,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            missing_path.parent.mkdir(parents=True, exist_ok=True)
            missing_path.write_text(json.dumps(dict(sorted(missing.items())), indent=1) + "\n")
        return report

    def git_url(name: str) -> str:
        return name if "://" in name or name.startswith("/") else f"https://github.com/{name}.git"

    legs = (
        (
            "code",
            lambda: backup_code(
                git_url(code),
                git_url(mirror),
                args.repo,
                branch=args.source_branch,
                dry_run=args.dry_run,
                source_token=os.environ.get("SOURCE_GH_TOKEN"),
                mirror_token=os.environ.get("MIRROR_GH_TOKEN"),
            ),
        ),
        ("store", store),
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
            + (f", {report.missing} missing upstream" if report.missing else "")
            + (f", {report.expired} expired" if report.expired else "")
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
