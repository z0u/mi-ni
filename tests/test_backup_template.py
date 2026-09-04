"""The backup template's payload script (`templates/backup/backup.py`).

The code leg runs against real temporary git repos, since what it checks is git's behavior (what a non-fast-forward push does, what a tag refspec without `+` refuses). The two Hugging Face legs run against a fake API that keeps the backup dataset as a dict of path → bytes and appends to it on every `upload_folder`, which is all the script relies on: the never-delete invariant is then checkable as "nothing leaves the dict".
"""

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
def repo(tmp_path: Path) -> Path:
    """A checkout of the (empty) backup repo, with a bare `origin` standing in for GitHub."""
    origin = tmp_path / "origin.git"
    git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    checkout = tmp_path / "backup-repo"
    subprocess.run(["git", "clone", "-q", str(origin), str(checkout)], check=True, capture_output=True)
    return checkout


def origin_refs(repo: Path) -> dict[str, str]:
    out = git("ls-remote", "origin", cwd=repo)
    return {ref: sha for sha, ref in (line.split("\t") for line in out.splitlines())}


# -- code leg --------------------------------------------------------------------------


def test_code_first_run_mirrors_snapshots_and_copies_tags(backup, source, repo):
    tip = git("rev-parse", "main", cwd=source)
    v1 = git("rev-parse", "v1", cwd=source)

    report = backup.backup_code(str(source), repo, prev_sha=None, dry_run=False)

    refs = origin_refs(repo)
    assert refs["refs/heads/mirror"] == tip
    assert refs["refs/tags/source/v1"] == v1
    assert refs[f"refs/tags/{report.snapshot}"] == tip
    assert re.fullmatch(r"snap/\d{4}-\d\d-\d\d", report.snapshot)
    assert report.to_dict() == {"source": str(source), "sha": tip, "snapshot": report.snapshot}


def test_code_unchanged_tip_makes_no_snapshot(backup, source, repo):
    tip = git("rev-parse", "main", cwd=source)
    first = backup.backup_code(str(source), repo, prev_sha=None, dry_run=False)

    second = backup.backup_code(str(source), repo, prev_sha=tip, dry_run=False)

    assert second.to_dict() == {"source": str(source), "sha": tip}
    assert [r for r in origin_refs(repo) if r.startswith("refs/tags/snap/")] == [f"refs/tags/{first.snapshot}"]


def test_code_rewritten_history_is_refused_reported_and_still_snapshotted(backup, source, repo):
    old_tip = git("rev-parse", "main", cwd=source)
    backup.backup_code(str(source), repo, prev_sha=None, dry_run=False)
    git("reset", "-q", "--hard", "HEAD~1", cwd=source)
    new_tip = commit(source, "rewritten")

    report = backup.backup_code(str(source), repo, prev_sha=old_tip, dry_run=False)

    refs = origin_refs(repo)
    assert refs["refs/heads/mirror"] == old_tip  # never forced
    assert refs[f"refs/tags/{report.snapshot}"] == new_tip  # but the night's tip is recorded
    assert re.fullmatch(r"snap/\d{4}-\d\d-\d\dT\d{4}Z", report.snapshot)  # same-day collision → time suffix
    assert any("rewritten" in n for n in report.notes)


def test_code_a_moved_source_tag_stays_where_first_seen(backup, source, repo):
    v1 = git("rev-parse", "v1", cwd=source)
    backup.backup_code(str(source), repo, prev_sha=None, dry_run=False)
    git("tag", "-f", "v1", "main", cwd=source)

    report = backup.backup_code(str(source), repo, prev_sha=git("rev-parse", "main", cwd=source), dry_run=False)

    assert origin_refs(repo)["refs/tags/source/v1"] == v1
    assert any("tags" in n for n in report.notes)


def test_code_dry_run_fetches_but_pushes_nothing(backup, source, repo):
    report = backup.backup_code(str(source), repo, prev_sha=None, dry_run=True)

    assert (report.sha, report.snapshot) == (git("rev-parse", "main", cwd=source), report.snapshot)
    assert report.snapshot.startswith("snap/")
    assert origin_refs(repo) == {}


# -- Hugging Face fixtures ---------------------------------------------------------------


def repo_not_found():
    """huggingface_hub's HTTP errors carry the response they came from (`EntryNotFoundError` is a plain exception)."""
    return RepositoryNotFoundError(
        "no such dataset", response=httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
    )


def entry(path: str, size: int = 1):
    return SimpleNamespace(path=path, size=size, type="file")


def hf_commit(sha: str, title: str = "publish"):
    return SimpleNamespace(commit_id=sha, title=title)


class FakeApi:
    """Two sources and one backup dataset, in memory."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.bucket: dict[str, bytes] = {}  # path → content
        self.history: list[tuple[str, dict[str, bytes]]] = []  # newest first: (sha, files at that revision)
        self.backup: dict[str, bytes] | None = {}  # None → the dataset doesn't exist
        self.uploads: list[dict] = []
        self.downloaded: list[str] = []

    # -- store bucket
    def list_bucket_tree(self, bucket, recursive):
        return [entry(p, len(b)) for p, b in self.bucket.items()]

    def download_bucket_files(self, bucket, files, raise_on_missing_files):
        for e, dest in files:
            self.downloaded.append(e.path)
            Path(dest).write_bytes(self.bucket[e.path])

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
        self.backup |= staged
        self.uploads.append({"message": commit_message, "description": commit_description, "files": sorted(staged)})
        return SimpleNamespace(oid=f"b{len(self.uploads)}")


@pytest.fixture
def api(tmp_path):
    return FakeApi(tmp_path)


# -- store leg ----------------------------------------------------------------------


def test_store_copies_only_what_the_backup_lacks_plus_every_ref(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"one", "cas/bb/2": b"two", "refs/run/x": b"ptr"}
    api.backup = {"store/cas/aa/1": b"one", "store/refs/run/x": b"old"}

    report = backup.backup_store(api, api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=1 << 20, dry_run=False)

    assert sorted(api.downloaded) == ["cas/bb/2", "refs/run/x"]
    assert api.backup == {"store/cas/aa/1": b"one", "store/cas/bb/2": b"two", "store/refs/run/x": b"ptr"}
    assert report.to_dict() == {"source": "ns/store", "seen": 3, "copied": 2, "bytes": 6, "commits": ["b1"]}
    assert not (tmp_path / "w" / "store").exists()


def test_store_dry_run_sizes_the_copy_and_uploads_nothing(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"12345"}
    api.backup = None  # not created yet

    report = backup.backup_store(api, api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=1 << 20, dry_run=True)

    assert (report.copied, report.bytes) == (1, 5)
    assert api.uploads == [] and api.downloaded == []


def test_store_splits_a_large_delta_into_one_commit_per_batch(backup, api, tmp_path):
    api.bucket = {f"cas/{i}": bytes(3) for i in range(5)}

    report = backup.backup_store(api, api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=7, dry_run=False)

    assert [len(u["files"]) for u in api.uploads] == [2, 2, 1]
    assert report.commits == ["b1", "b2", "b3"]
    assert len(api.backup) == 5


def test_batches_never_splits_a_file_larger_than_the_limit(backup):
    big, small = entry("big", 10), entry("small", 1)
    assert list(backup.batches([small, big, small], limit=4)) == [[small], [big], [small]]


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
    """What each `HfApi(...)` was given, in construction order: the source client, then the backup client."""
    return []


@pytest.fixture
def run(backup, api, source, repo, tmp_path, monkeypatch, hf_tokens):
    """Run `main()` with the fake API and the temporary git repos, from a scratch cwd; returns (exit code, state dict)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_REPO", str(source))
    monkeypatch.setenv("SOURCE_BUCKET", "ns/store")
    monkeypatch.setenv("SOURCE_PUBLISH_REPO", "ns/pub")
    monkeypatch.setenv("BACKUP_DATASET", "ns/backup")
    monkeypatch.setenv("HF_TOKEN", "t-write")
    monkeypatch.delenv("SOURCE_HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: (hf_tokens.append(token), api)[1])

    def go(*argv: str):
        monkeypatch.setattr(sys, "argv", ["backup.py", "--repo", str(repo), *argv])
        code = backup.main()
        return code, json.loads(Path("state/last-run.json").read_text())

    return go


def test_main_records_all_three_legs(api, source, run):
    api.bucket = {"cas/1": b"x"}
    api.history = [("c1", {"a": b"a"})]

    code, state = run()

    assert code == 0
    assert state.pop("ran_at").startswith("20")  # an ISO timestamp
    snapshot = state["code"].pop("snapshot")
    assert snapshot.startswith("snap/")
    assert state == {
        "backup": "ns/backup",
        "dry_run": False,
        "code": {"source": str(source), "sha": git("rev-parse", "main", cwd=source)},
        "store": {"source": "ns/store", "seen": 1, "copied": 1, "bytes": 1, "commits": ["b1"]},
        "pub": {"source": "ns/pub", "seen": 1, "copied": 1, "commits": ["b2"]},
    }
    assert not Path(".backup-work").exists()


def test_main_reads_the_last_tip_from_the_previous_record(run):
    run()

    _, state = run()

    assert "snapshot" not in state["code"]  # the tip hadn't moved since the run before


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


def test_main_requires_the_four_names(backup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOURCE_REPO", raising=False)
    monkeypatch.setattr(sys, "argv", ["backup.py"])
    with pytest.raises(SystemExit, match="SOURCE_REPO"):
        backup.main()


# -- credentials -------------------------------------------------------------------


def test_main_reads_the_sources_anonymously_and_writes_with_the_write_token(run, hf_tokens):
    run()

    assert hf_tokens == [False, "t-write"]  # source client, backup client; False is anonymous, never a fallback


def test_main_reads_private_sources_with_their_own_read_token(run, hf_tokens, monkeypatch):
    monkeypatch.setenv("SOURCE_HF_TOKEN", "t-read")

    run()

    assert hf_tokens == ["t-read", "t-write"]


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
