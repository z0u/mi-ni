"""The backup template's payload script (`templates/backup/backup.py`).

The code leg runs against real temporary git repos, since what it checks is git's behavior (what a non-fast-forward push does, what a tag refspec without `+` refuses). The Hugging Face legs run against a fake API that keeps the backup dataset and the backup bucket as dicts and applies each call to them, which is all the script relies on: the never-delete invariant is then checkable as "nothing leaves the dataset dict, and nothing leaves the bucket dict before its window ends".
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

SCRIPT = Path(__file__).parents[1] / "templates" / "backup" / "backup.py"
WORKFLOW = SCRIPT.parent / ".github" / "workflows" / "backup.yml"


@pytest.fixture(scope="module")
def backup():
    spec = importlib.util.spec_from_file_location("backup", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["backup"] = module
    spec.loader.exec_module(module)
    return module


# -- git fixtures ----------------------------------------------------------------------


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name + "\n")
    git("add", ".", cwd=repo)
    git("commit", "-qm", name, cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """The project being backed up: two commits on `main`, tag `v1` on the first."""
    root = tmp_path / "source"
    root.mkdir()
    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "test@example.invalid", cwd=root)
    git("config", "user.name", "Test", cwd=root)
    commit(root, "one")
    git("tag", "v1", cwd=root)
    commit(root, "two")
    return root


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    """A bare repo standing in for the mirror repo on GitHub."""
    root = tmp_path / "mirror.git"
    git("init", "-q", "--bare", str(root), cwd=tmp_path)
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The backup repo's checkout: scratch, as far as the code leg is concerned. Fresh per test, like a runner."""
    root = tmp_path / "backup-repo"
    root.mkdir()
    git("init", "-q", "-b", "main", cwd=root)
    return root


def mirror_refs(mirror: Path) -> dict[str, str]:
    out = git("for-each-ref", "--format=%(refname) %(objectname)", cwd=mirror)
    return dict(line.split(" ") for line in out.splitlines())


# -- code leg --------------------------------------------------------------------------


def test_code_first_run_mirrors_snapshots_and_copies_tags(backup, source, mirror, repo):
    tip = git("rev-parse", "main", cwd=source)
    v1 = git("rev-parse", "v1", cwd=source)

    report = backup.backup_code(str(source), str(mirror), repo, dry_run=False)

    refs = mirror_refs(mirror)
    assert refs["refs/heads/mirror"] == tip
    assert refs["refs/tags/source/v1"] == v1
    assert refs[f"refs/tags/{report.snapshot}"] == tip
    assert re.fullmatch(r"snap/\d{4}-\d\d-\d\d", report.snapshot)
    assert report.to_dict() == {"source": str(source), "sha": tip, "snapshot": report.snapshot}


def test_code_unchanged_tip_makes_no_snapshot(backup, source, mirror, repo):
    tip = git("rev-parse", "main", cwd=source)
    first = backup.backup_code(str(source), str(mirror), repo, dry_run=False)

    second = backup.backup_code(str(source), str(mirror), repo, dry_run=False)

    assert second.to_dict() == {"source": str(source), "sha": tip}
    assert [r for r in mirror_refs(mirror) if r.startswith("refs/tags/snap/")] == [f"refs/tags/{first.snapshot}"]


def test_code_rewritten_history_is_refused_reported_and_still_snapshotted(backup, source, mirror, repo):
    old_tip = git("rev-parse", "main", cwd=source)
    backup.backup_code(str(source), str(mirror), repo, dry_run=False)
    git("reset", "-q", "--hard", "HEAD~1", cwd=source)
    new_tip = commit(source, "rewritten")

    report = backup.backup_code(str(source), str(mirror), repo, dry_run=False)

    refs = mirror_refs(mirror)
    assert refs["refs/heads/mirror"] == old_tip  # never forced
    assert refs[f"refs/tags/{report.snapshot}"] == new_tip  # but the night's tip is recorded
    assert re.fullmatch(r"snap/\d{4}-\d\d-\d\dT\d{4}Z", report.snapshot)  # same-day collision → time suffix
    assert any("rewritten" in n for n in report.notes)


def test_code_a_moved_source_tag_stays_where_first_seen(backup, source, mirror, repo):
    v1 = git("rev-parse", "v1", cwd=source)
    backup.backup_code(str(source), str(mirror), repo, dry_run=False)
    git("tag", "-f", "v1", "main", cwd=source)

    report = backup.backup_code(str(source), str(mirror), repo, dry_run=False)

    assert mirror_refs(mirror)["refs/tags/source/v1"] == v1
    assert any("tags" in n for n in report.notes)


def test_code_dry_run_fetches_but_pushes_nothing(backup, source, mirror, repo):
    report = backup.backup_code(str(source), str(mirror), repo, dry_run=True)

    assert (report.sha, report.snapshot) == (git("rev-parse", "main", cwd=source), report.snapshot)
    assert report.snapshot.startswith("snap/")
    assert mirror_refs(mirror) == {}


def test_github_slug_only_for_github_urls(backup):
    assert backup.github_slug("https://github.com/o/mirror.git") == "o/mirror"
    assert backup.github_slug("https://github.com/o/mirror") == "o/mirror"
    assert backup.github_slug("/tmp/mirror.git") is None
    assert backup.github_slug("https://gitlab.example/o/mirror.git") is None


def test_check_mirror_refuses_a_github_repo_with_actions_enabled(backup, monkeypatch):
    asked: list[tuple[str, str | None]] = []
    monkeypatch.setattr(backup, "actions_enabled", lambda slug, token: (asked.append((slug, token)), True)[1])

    with pytest.raises(RuntimeError, match="Actions is enabled on o/mirror"):
        backup.check_mirror("https://github.com/o/mirror.git", "tok")
    assert asked == [("o/mirror", "tok")]

    monkeypatch.setattr(backup, "actions_enabled", lambda slug, token: False)
    backup.check_mirror("https://github.com/o/mirror.git", "tok")  # off: fine
    backup.check_mirror("/tmp/mirror.git", None)  # not GitHub: not this leg's call


def test_actions_enabled_asks_the_api_with_the_token_and_fails_closed(backup, monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    seen = {}

    def urlopen(req):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        return io.BytesIO(b'{"enabled": false}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert backup.actions_enabled("o/mirror", "tok") is False
    assert seen == {"url": "https://api.github.com/repos/o/mirror/actions/permissions", "auth": "Bearer tok"}

    def forbidden(req):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(RuntimeError, match="Administration read"):
        backup.actions_enabled("o/mirror", "tok")


# -- Hugging Face fixtures ---------------------------------------------------------------


def repo_not_found():
    """huggingface_hub's HTTP errors carry the response they came from (`EntryNotFoundError` is a plain exception)."""
    return RepositoryNotFoundError(
        "no such dataset", response=httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
    )


def entry(path: str, size: int = 1, xet_hash: str | None = "h"):
    return SimpleNamespace(path=path, size=size, type="file", xet_hash=xet_hash)


def hf_commit(sha: str, title: str = "publish"):
    return SimpleNamespace(commit_id=sha, title=title)


class FakeApi:
    """Two sources, one backup dataset and one backup bucket, in memory.

    Bucket contents are `path → bytes`; a file's xet hash is its content, so a server-side copy by hash is a dict lookup.
    """

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.bucket: dict[str, bytes] = {}  # the source bucket
        self.backup_bucket: dict[str, bytes] | None = {}  # None → not created yet
        self.history: list[tuple[str, dict[str, bytes]]] = []  # newest first: (sha, files at that revision)
        self.backup: dict[str, bytes] | None = {}  # the dataset; None → doesn't exist
        self.uploads: list[dict] = []
        self.downloaded: list[str] = []
        self.batches: list[dict] = []

    # -- buckets
    def list_bucket_tree(self, bucket, recursive):
        files = {"ns/store": self.bucket, "ns/backup-store": self.backup_bucket}[bucket]
        if files is None:
            raise repo_not_found()
        return [entry(p, len(b), xet_hash=b.decode()) for p, b in files.items()]

    def download_bucket_files(self, bucket, files, raise_on_missing_files):
        for e, dest in files:
            self.downloaded.append(e.path)
            Path(dest).write_bytes(self.bucket[e.path])

    def batch_bucket_files(self, bucket, *, add=None, copy=None, delete=None):
        assert bucket == "ns/backup-store" and self.backup_bucket is not None
        self.batches.append({"add": len(add or []), "copy": len(copy or []), "delete": len(delete or [])})
        for src, dest in add or []:
            self.backup_bucket[dest] = Path(src).read_bytes()
        for kind, source, xet_hash, dest in copy or []:
            assert (kind, source) == ("bucket", "ns/store")
            self.backup_bucket[dest] = xet_hash.encode()  # by hash: the bytes never pass through here
        for path in delete or []:
            del self.backup_bucket[path]

    # -- publish repo
    def list_repo_commits(self, repo, repo_type):
        return [hf_commit(sha) for sha, _ in self.history]

    def snapshot_download(self, repo, repo_type, revision, local_dir):
        files = dict(self.history)[revision]
        for p, b in files.items():
            (Path(local_dir) / p).parent.mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / p).write_bytes(b)
        return local_dir

    # -- backup dataset
    def list_repo_tree(self, dataset, path_in_repo, recursive, repo_type):
        if self.backup is None:
            raise repo_not_found()
        under = [p for p in self.backup if p.startswith(path_in_repo + "/")]
        if not under:
            raise EntryNotFoundError("no such path")
        return [entry(p) for p in under]

    def hf_hub_download(self, dataset, filename, repo_type):
        if self.backup is None:
            raise repo_not_found()
        if filename not in self.backup:
            raise EntryNotFoundError(filename)
        path = self.tmp / "dl" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.backup[filename])
        return str(path)

    def upload_folder(
        self, *, repo_id, repo_type, folder_path, path_in_repo, ignore_patterns, commit_message, commit_description=None
    ):
        assert self.backup is not None, "uploading to a dataset that doesn't exist"
        staged = {
            f"{path_in_repo}/{p.relative_to(folder_path)}": p.read_bytes()
            for p in Path(folder_path).rglob("*")
            if p.is_file()
        }
        if all(self.backup.get(p) == b for p, b in staged.items()):
            return self.repo_info(repo_id, repo_type)  # the Hub skips an empty commit and answers with its head
        self.backup |= staged
        self.uploads.append({"message": commit_message, "description": commit_description, "files": sorted(staged)})
        return self.repo_info(repo_id, repo_type)

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(sha=f"b{len(self.uploads)}", oid=f"b{len(self.uploads)}")


@pytest.fixture
def api(tmp_path):
    return FakeApi(tmp_path)


# -- store leg ----------------------------------------------------------------------


def store(backup, api, tmp_path, *, missing=None, retain_days=90, dry_run=False):
    return backup.backup_store(
        api,
        api,
        api,
        "ns/store",
        "ns/backup",
        "ns/backup-store",
        tmp_path / "w",
        missing or {},
        retain_days=retain_days,
        dry_run=dry_run,
    )


def test_store_copies_by_hash_what_the_backup_lacks_and_keeps_refs_history(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"one", "cas/bb/2": b"two", "refs/run/x.json": b"ptr"}
    api.backup_bucket = {"cas/aa/1": b"one", "refs/run/x.json": b"old"}

    report, missing = store(backup, api, tmp_path)

    assert api.backup_bucket == api.bucket  # the new blob, and the ref whose hash changed
    assert api.batches == [{"add": 0, "copy": 2, "delete": 0}]
    assert api.downloaded == ["refs/run/x.json"]  # the refs, for the dataset; no CAS bytes through the runner
    assert api.backup == {"store/refs/run/x.json": b"ptr"}
    assert report.to_dict() == {"source": "ns/store", "seen": 3, "copied": 2, "bytes": 6, "commits": ["b1"]}
    assert missing == {}
    assert not (tmp_path / "w" / "store").exists()


def test_store_unchanged_refs_make_no_dataset_commit(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"one", "refs/run/x.json": b"ptr"}
    api.backup_bucket = dict(api.bucket)
    api.backup = {"store/refs/run/x.json": b"ptr"}

    report, _ = store(backup, api, tmp_path)

    assert report.to_dict() == {"source": "ns/store", "seen": 2}  # no commit: the Hub skipped an empty one
    assert api.uploads == []


def test_store_a_file_without_a_xet_hash_goes_the_long_way_round(backup, api, tmp_path, monkeypatch):
    api.bucket = {"cas/aa/1": b"one"}
    monkeypatch.setattr(
        api,
        "list_bucket_tree",
        lambda bucket, recursive: [entry("cas/aa/1", 3, xet_hash=None)] if bucket == "ns/store" else [],
    )

    report, _ = store(backup, api, tmp_path)

    assert api.downloaded == ["cas/aa/1"] and api.backup_bucket == {"cas/aa/1": b"one"}
    assert api.batches == [{"add": 1, "copy": 0, "delete": 0}]


def test_store_keeps_what_the_source_dropped_until_the_window_ends(backup, api, tmp_path):
    api.bucket = {"cas/keep": b"k"}
    api.backup_bucket = {"cas/keep": b"k", "cas/new-gone": b"n", "cas/old-gone": b"o"}

    report, missing = store(backup, api, tmp_path, missing={"cas/old-gone": "2000-01-01"})

    assert api.backup_bucket == {"cas/keep": b"k", "cas/new-gone": b"n"}  # the old one expired, the new one waits
    assert api.batches == [{"add": 0, "copy": 0, "delete": 1}]
    assert list(missing) == ["cas/new-gone"] and missing["cas/new-gone"].startswith("20")
    assert (report.missing, report.expired) == (1, 1)
    assert report.notes == ["1 file(s) newly missing from the source, kept for 90 days"]


def test_store_a_file_that_returns_to_the_source_leaves_the_missing_list(backup, api, tmp_path):
    api.bucket = api.backup_bucket = {"cas/back": b"b"}

    _, missing = store(backup, api, tmp_path, missing={"cas/back": "2000-01-01"})

    assert missing == {} and api.backup_bucket == {"cas/back": b"b"}


def test_store_warns_when_most_of_the_source_vanishes_overnight(backup, api, tmp_path, capsys):
    api.bucket = {}
    api.backup_bucket = {"cas/1": b"1", "cas/2": b"2"}

    report, missing = store(backup, api, tmp_path)

    assert len(missing) == 2 and api.backup_bucket == {"cas/1": b"1", "cas/2": b"2"}  # nothing deleted today
    assert report.notes == ["2 of 2 backed-up files vanished from the source overnight — inspect the source"]
    assert "::warning::" in capsys.readouterr().out


def test_store_dry_run_sizes_the_copy_and_writes_nothing(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"12345"}
    api.backup_bucket = None  # not created yet
    api.backup = None

    report, _ = store(backup, api, tmp_path, dry_run=True)

    assert (report.copied, report.bytes) == (1, 5)
    assert api.uploads == [] and api.downloaded == [] and api.batches == []


def test_retention_is_a_clock_the_source_can_start_but_not_hurry(backup):
    from datetime import date

    today = date(2026, 9, 5)
    missing, expired = backup.retention(
        {"a": "2026-06-01", "b": "2026-06-10", "returned": "2026-01-01"},
        source={"kept", "returned"},
        have={"kept", "returned", "a", "b", "fresh"},
        today=today,
        retain_days=90,
    )
    assert expired == ["a"]  # 96 days
    assert missing == {"b": "2026-06-10", "fresh": "2026-09-05"}  # b at 87 days waits; fresh starts today


# -- publish leg --------------------------------------------------------------------


def test_pending_commits_replays_oldest_first_after_the_marker(backup):
    history = [hf_commit("c3"), hf_commit("c2"), hf_commit("c1")]  # newest first, as the API returns it
    assert [c.commit_id for c in backup.pending_commits(history, None)] == ["c1", "c2", "c3"]
    assert [c.commit_id for c in backup.pending_commits(history, "c2")] == ["c3"]
    assert backup.pending_commits(history, "c3") == []


def test_a_marker_missing_from_the_history_means_a_rewrite_and_stops_the_replay(backup):
    with pytest.raises(RuntimeError, match="rewritten"):
        backup.pending_commits([hf_commit("c2"), hf_commit("c1")], "gone")


def test_publish_replays_each_commit_and_keeps_deleted_files(backup, api, tmp_path):
    api.history = [
        ("c2", {"exports/a/index.html": b"a2", "exports/b/index.html": b"b1"}),  # c2 rewrote a/, deleted old/
        ("c1", {"exports/a/index.html": b"a1", "old.txt": b"gone"}),
    ]

    report = backup.backup_publish(api, api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=200, dry_run=False)

    assert [u["message"] for u in api.uploads] == ["pub: replay c1", "pub: replay c2"]
    assert api.uploads[0]["description"] == "ns/pub — publish"
    assert api.backup == {
        "pub/exports/a/index.html": b"a2",
        "pub/exports/b/index.html": b"b1",
        "pub/old.txt": b"gone",  # the union head: a deletion in the source removes nothing here
        "pub/SOURCE_COMMIT": b"c2\n",
    }
    assert report.to_dict() == {"source": "ns/pub", "seen": 2, "copied": 2, "commits": ["b1", "b2"]}


def test_publish_resumes_from_the_marker_and_defers_past_max_commits(backup, api, tmp_path):
    api.history = [("c4", {}), ("c3", {}), ("c2", {}), ("c1", {})]
    api.backup = {"pub/SOURCE_COMMIT": b"c1\n"}

    report = backup.backup_publish(api, api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=2, dry_run=False)

    # A commit that only deleted files still lands as a real commit, because the marker changed.
    assert [u["message"] for u in api.uploads] == ["pub: replay c2", "pub: replay c3"]
    assert api.backup["pub/SOURCE_COMMIT"] == b"c3\n"
    assert report.notes == ["1 commit(s) deferred to a later run (--max-commits 2)"]


def test_publish_dry_run_counts_without_touching_anything(backup, api, tmp_path):
    api.history = [("c1", {"x": b"x"})]
    api.backup = None

    report = backup.backup_publish(api, api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=200, dry_run=True)

    assert (report.seen, report.copied, api.uploads) == (1, 1, [])


# -- the run ------------------------------------------------------------------------


@pytest.fixture
def hf_tokens() -> list:
    """What each `HfApi(...)` was given, in construction order: the source client, the dataset client, the bucket client."""
    return []


@pytest.fixture
def run(backup, api, source, mirror, repo, tmp_path, monkeypatch, hf_tokens):
    """Run `main()` with the fake API and the temporary git repos, from a scratch cwd; returns (exit code, state dict)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_REPO", str(source))
    monkeypatch.setenv("MIRROR_REPO", str(mirror))
    monkeypatch.setenv("SOURCE_BUCKET", "ns/store")
    monkeypatch.setenv("SOURCE_PUBLISH_REPO", "ns/pub")
    monkeypatch.setenv("BACKUP_DATASET", "ns/backup")
    monkeypatch.setenv("BACKUP_BUCKET", "ns/backup-store")
    monkeypatch.setenv("HF_TOKEN", "t-write")
    monkeypatch.delenv("SOURCE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_BUCKET_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: (hf_tokens.append(token), api)[1])

    def go(*argv: str):
        monkeypatch.setattr(sys, "argv", ["backup.py", "--repo", str(repo), *argv])
        code = backup.main()
        return code, json.loads(Path("state/last-run.json").read_text())

    return go


def test_main_records_all_three_legs(api, source, mirror, run):
    api.bucket = {"cas/1": b"x"}
    api.history = [("c1", {"a": b"a"})]

    code, state = run()

    assert code == 0
    assert state.pop("ran_at").startswith("20")  # an ISO timestamp
    snapshot = state["code"].pop("snapshot")
    assert snapshot.startswith("snap/")
    assert state == {
        "mirror": str(mirror),
        "backup": "ns/backup",
        "backup_bucket": "ns/backup-store",
        "dry_run": False,
        "code": {"source": str(source), "sha": git("rev-parse", "main", cwd=source)},
        "store": {"source": "ns/store", "seen": 1, "copied": 1, "bytes": 1},
        "pub": {"source": "ns/pub", "seen": 1, "copied": 1, "commits": ["b1"]},
    }
    assert json.loads(Path("state/store-missing.json").read_text()) == {}
    assert not Path(".backup-work").exists()


def test_main_carries_the_missing_list_between_runs(api, run):
    api.bucket = {"cas/1": b"x", "cas/2": b"y"}
    run()
    api.bucket = {"cas/1": b"x"}

    _, state = run()

    assert state["store"]["missing"] == 1
    assert list(json.loads(Path("state/store-missing.json").read_text())) == ["cas/2"]
    assert api.backup_bucket == {"cas/1": b"x", "cas/2": b"y"}


def test_main_dry_run_leaves_the_missing_list_alone(api, run):
    api.backup_bucket = {"cas/gone": b"g"}

    run("--dry-run")

    assert not Path("state/store-missing.json").exists()


def test_main_takes_the_last_tip_from_the_mirror_not_the_record(run):
    run()
    Path("state/last-run.json").unlink()  # a reset or lost state file

    _, state = run()

    assert "snapshot" not in state["code"]  # the mirror already held the tip


def test_main_keeps_going_when_one_leg_fails_and_exits_nonzero(api, run, capsys):
    api.bucket = {"cas/1": b"x"}
    api.history = [("c2", {}), ("c1", {})]
    api.backup = {"pub/SOURCE_COMMIT": b"rewritten-away\n"}

    code, state = run()

    assert code == 1
    assert state["store"]["copied"] == 1  # the store leg still ran
    assert "rewritten" in state["pub"]["error"]
    assert state["errors"] == [f"pub: {state['pub']['error']}"]
    assert "error: pub:" in capsys.readouterr().err


def test_main_requires_the_six_names(backup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOURCE_REPO", raising=False)
    monkeypatch.setattr(sys, "argv", ["backup.py"])
    with pytest.raises(SystemExit, match="SOURCE_REPO"):
        backup.main()


# -- credentials -------------------------------------------------------------------


def test_main_reads_the_sources_anonymously_and_writes_with_the_write_token(run, hf_tokens):
    run()

    assert hf_tokens == [False, "t-write", "t-write"]  # source, dataset, bucket; False is anonymous, never a fallback


def test_main_reads_private_sources_with_their_own_read_token(run, hf_tokens, monkeypatch):
    monkeypatch.setenv("SOURCE_HF_TOKEN", "t-read")

    run()

    assert hf_tokens == ["t-read", "t-write", "t-write"]


def test_main_writes_the_bucket_with_its_own_token_when_given_one(run, hf_tokens, monkeypatch):
    monkeypatch.setenv("HF_BUCKET_TOKEN", "t-bucket")

    run()

    assert hf_tokens == [False, "t-write", "t-bucket"]


def test_main_needs_a_write_token_unless_dry_run(run, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")

    with pytest.raises(SystemExit, match="HF_TOKEN"):
        run()
    code, state = run("--dry-run")

    assert (code, state["dry_run"]) == (0, True)


def test_auth_env_sends_the_token_as_a_header_for_that_host_only(backup):
    env = backup.auth_env("https://github.test/owner/project.git", "tok")

    assert env == {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://github.test/.extraheader",
        "GIT_CONFIG_VALUE_0": "",  # resets the checkout's own header for the host first
        "GIT_CONFIG_KEY_1": "http.https://github.test/.extraheader",
        "GIT_CONFIG_VALUE_1": "AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46dG9r",  # base64 of x-access-token:tok
    }
    assert backup.auth_env("https://github.test/owner/project.git", None) == {}
    assert backup.auth_env("/local/path", "tok") == {}


def test_script_parses_at_the_python_the_workflow_pins():
    """`backup.py` is the recovery tool, so it must not outrun the interpreter that runs it.

    The tests themselves run on this project's Python, which is newer than the runner's — so a
    construct valid only here would otherwise reach the template unnoticed. `feature_version`
    re-parses the source as the pinned version would see it, and the pin is read from the
    workflow so the two cannot drift apart.
    """
    pin = re.search(r'python-version:\s*"(\d+)\.(\d+)"', WORKFLOW.read_text())
    assert pin, "the workflow no longer pins python-version"
    ast.parse(SCRIPT.read_text(), feature_version=(int(pin[1]), int(pin[2])))
