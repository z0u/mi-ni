"""Tests for the volume module."""

from pathlib import Path

import pytest

from mini.local_volume import LocalVolume
from mini.volume import data_dir_context, get_data_dir


# ---------------------------------------------------------------------------
# Context var tests
# ---------------------------------------------------------------------------


def test_get_data_dir_raises_outside_context():
    """get_data_dir() raises RuntimeError when called outside a job."""
    with pytest.raises(RuntimeError, match="No data directory"):
        get_data_dir()


def test_data_dir_context_nests_and_resets():
    """Inner context overrides outer, outer is restored after it exits, and the var is unset once all of them do."""
    with data_dir_context(Path("/outer")):
        assert get_data_dir() == Path("/outer")
        with data_dir_context(Path("/inner")):
            assert get_data_dir() == Path("/inner")
        assert get_data_dir() == Path("/outer")
    with pytest.raises(RuntimeError, match="No data directory"):
        get_data_dir()


# ---------------------------------------------------------------------------
# LocalVolume tests
# ---------------------------------------------------------------------------


def test_local_volume_no_directory_on_init(tmp_path):
    """LocalVolume does not create the directory on init."""
    vol_path = tmp_path / "experiment-1"
    LocalVolume(vol_path)
    assert not vol_path.exists()


def _roots(vol: LocalVolume, tmp_path: Path, direction: str) -> tuple[Path, Path]:
    """Source and destination roots for a transfer: one side is the volume, the other is local disk."""
    local, remote = tmp_path / "local", vol.path / "remote"
    return (local, remote) if direction == "upload" else (remote, local)


async def _transfer(vol: LocalVolume, src: Path, dest: Path, direction: str) -> None:
    """One transfer in either direction. Both verbs take the *full* destination path, so the two calls mirror each other."""
    if direction == "upload":
        await vol.upload(src, str(dest.relative_to(vol.path)))
    else:
        await vol.download(str(src.relative_to(vol.path)), dest)


@pytest.mark.parametrize("direction", ["upload", "download"])
async def test_local_volume_transfers_a_single_file(tmp_path: Path, direction: str):
    vol = LocalVolume(tmp_path / "vol")
    src, dest = _roots(vol, tmp_path, direction)
    src.mkdir(parents=True)
    (src / "data.csv").write_text("a,b,c")

    await _transfer(vol, src / "data.csv", dest / "input" / "data.csv", direction)
    assert (dest / "input" / "data.csv").read_text() == "a,b,c"


@pytest.mark.parametrize("direction", ["upload", "download"])
async def test_local_volume_transfers_a_directory_tree_and_merges(tmp_path: Path, direction: str):
    """A tree copies whole; a second transfer onto the same destination merges into it rather than failing or replacing it."""
    vol = LocalVolume(tmp_path / "vol")
    src, dest = _roots(vol, tmp_path, direction)
    (src / "one").mkdir(parents=True)
    (src / "one" / "train.csv").write_text("train")
    (src / "one" / "test.csv").write_text("test")
    (src / "two").mkdir()
    (src / "two" / "extra.csv").write_text("extra")

    await _transfer(vol, src / "one", dest / "dataset", direction)
    assert (dest / "dataset" / "train.csv").read_text() == "train"
    assert (dest / "dataset" / "test.csv").read_text() == "test"

    await _transfer(vol, src / "two", dest / "dataset", direction)
    assert (dest / "dataset" / "extra.csv").read_text() == "extra"
    assert (dest / "dataset" / "train.csv").read_text() == "train"  # what was already there survives


# ---------------------------------------------------------------------------
# Integration: get_data_dir() inside apparatus-mapped functions
# ---------------------------------------------------------------------------


def test_local_apparatus_provides_data_dir(tmp_path):
    """get_data_dir() returns a valid Path inside a LocalApparatus-mapped function."""
    from mini.local_apparatus import LocalApparatus

    captured_dirs: list[Path] = []

    def fn(x):
        captured_dirs.append(get_data_dir())
        return x

    app = LocalApparatus("test-vol", max_workers=1, data_dir=tmp_path / "vol")
    results = list(app.map(fn, [1, 2]))
    assert results == [1, 2]
    assert len(captured_dirs) == 2
    assert all(isinstance(d, Path) for d in captured_dirs)
    # All jobs in the same run share the same data dir
    assert captured_dirs[0] == captured_dirs[1]


def test_local_apparatus_custom_data_dir(tmp_path):
    """LocalApparatus accepts a custom data_dir."""
    from mini.local_apparatus import LocalApparatus

    custom = tmp_path / "my-data"

    def fn(x):
        d = get_data_dir()
        assert d.is_dir(), f"{d} is not a directory"  # created by the time the worker sees it
        return d

    app = LocalApparatus("test", max_workers=1, data_dir=custom)
    results = list(app.map(fn, [1]))
    assert results[0] == custom


def test_local_apparatus_no_dir_created_without_map(tmp_path):
    """LocalApparatus does not create the data directory until map() is called."""
    from mini.local_apparatus import LocalApparatus

    data_dir = tmp_path / "vol"
    LocalApparatus("test", max_workers=1, data_dir=data_dir)
    assert not data_dir.exists()
