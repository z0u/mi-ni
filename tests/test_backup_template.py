"""The backup template's payload script (`templates/backup/backup.py`), against a fake Hugging Face API.

The fake keeps the backup dataset as a dict of path → bytes and appends to it on every `upload_folder`, which is all the script relies on: the two never-delete invariants are then checkable as "nothing leaves the dict".
"""

import importlib.util
import json
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


def repo_not_found():
    """huggingface_hub's HTTP errors carry the response they came from (`EntryNotFoundError` is a plain exception)."""
    return RepositoryNotFoundError(
        "no such dataset", response=httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
    )


def entry(path: str, size: int = 1):
    return SimpleNamespace(path=path, size=size, type="file")


def commit(sha: str, title: str = "publish"):
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
        return [commit(sha) for sha, _ in self.history]

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
        assert "delete" not in commit_message
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

    report = backup.backup_store(api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=1 << 20, dry_run=False)

    assert sorted(api.downloaded) == ["cas/bb/2", "refs/run/x"]
    assert api.backup == {"store/cas/aa/1": b"one", "store/cas/bb/2": b"two", "store/refs/run/x": b"ptr"}
    assert report.to_dict() == {"source": "ns/store", "seen": 3, "copied": 2, "bytes": 6, "commits": ["b1"]}
    assert not (tmp_path / "w" / "store").exists()


def test_store_dry_run_sizes_the_copy_and_uploads_nothing(backup, api, tmp_path):
    api.bucket = {"cas/aa/1": b"12345"}
    api.backup = None  # not created yet

    report = backup.backup_store(api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=1 << 20, dry_run=True)

    assert (report.copied, report.bytes) == (1, 5)
    assert api.uploads == [] and api.downloaded == []


def test_store_splits_a_large_delta_into_one_commit_per_batch(backup, api, tmp_path):
    api.bucket = {f"cas/{i}": bytes(3) for i in range(5)}

    report = backup.backup_store(api, "ns/store", "ns/backup", tmp_path / "w", batch_bytes=7, dry_run=False)

    assert [len(u["files"]) for u in api.uploads] == [2, 2, 1]
    assert report.commits == ["b1", "b2", "b3"]
    assert len(api.backup) == 5


def test_batches_never_splits_a_file_larger_than_the_limit(backup):
    big, small = entry("big", 10), entry("small", 1)
    assert list(backup.batches([small, big, small], limit=4)) == [[small], [big], [small]]


# -- publish leg --------------------------------------------------------------------


def test_pending_commits_replays_oldest_first_after_the_marker(backup):
    history = [commit("c3"), commit("c2"), commit("c1")]  # newest first, as the API returns it
    assert [c.commit_id for c in backup.pending_commits(history, None)] == ["c1", "c2", "c3"]
    assert [c.commit_id for c in backup.pending_commits(history, "c2")] == ["c3"]
    assert backup.pending_commits(history, "c3") == []


def test_a_marker_missing_from_the_history_means_a_rewrite_and_stops_the_replay(backup):
    with pytest.raises(RuntimeError, match="rewritten"):
        backup.pending_commits([commit("c2"), commit("c1")], "gone")


def test_publish_replays_each_commit_and_keeps_deleted_files(backup, api, tmp_path):
    api.history = [
        ("c2", {"exports/a/index.html": b"a2", "exports/b/index.html": b"b1"}),  # c2 rewrote a/, deleted old/
        ("c1", {"exports/a/index.html": b"a1", "old.txt": b"gone"}),
    ]

    report = backup.backup_publish(api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=200, dry_run=False)

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

    report = backup.backup_publish(api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=2, dry_run=False)

    assert [u["message"] for u in api.uploads] == ["pub: replay c2", "pub: replay c3"]
    assert api.backup["pub/SOURCE_COMMIT"] == b"c3\n"
    assert report.notes == ["1 commit(s) deferred to a later run (--max-commits 2)"]
    # A commit that only deleted files still lands as a real commit, because the marker changed.
    assert len(api.uploads) == 2


def test_publish_dry_run_counts_without_touching_anything(backup, api, tmp_path):
    api.history = [("c1", {"x": b"x"})]
    api.backup = None

    report = backup.backup_publish(api, "ns/pub", "ns/backup", tmp_path / "w", max_commits=200, dry_run=True)

    assert (report.seen, report.copied, api.uploads) == (1, 1, [])


# -- the run ------------------------------------------------------------------------


@pytest.fixture
def run(backup, api, tmp_path, monkeypatch):
    """Run `main()` with the fake API, from a scratch cwd; returns (exit code, state dict)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_BUCKET", "ns/store")
    monkeypatch.setenv("SOURCE_PUBLISH_REPO", "ns/pub")
    monkeypatch.setenv("BACKUP_DATASET", "ns/backup")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)

    def go(*argv: str):
        monkeypatch.setattr(sys, "argv", ["backup.py", *argv])
        code = backup.main()
        return code, json.loads(Path("state/last-run.json").read_text())

    return go


def test_main_records_both_legs_and_the_code_leg_findings(api, run):
    api.bucket = {"cas/1": b"x"}
    api.history = [("c1", {"a": b"a"})]

    code, state = run("--code-sha", "abc", "--snapshot", "snap/2026-09-04")

    assert code == 0
    assert state.pop("ran_at").startswith("20")  # an ISO timestamp
    assert state == {
        "backup": "ns/backup",
        "dry_run": False,
        "code": {"sha": "abc", "snapshot": "snap/2026-09-04"},
        "store": {"source": "ns/store", "seen": 1, "copied": 1, "bytes": 1, "commits": ["b1"]},
        "pub": {"source": "ns/pub", "seen": 1, "copied": 1, "commits": ["b2"]},
    }
    assert not Path(".backup-work").exists()


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


def test_main_requires_the_three_names(backup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOURCE_BUCKET", raising=False)
    monkeypatch.setattr(sys, "argv", ["backup.py"])
    with pytest.raises(SystemExit, match="SOURCE_BUCKET"):
        backup.main()
