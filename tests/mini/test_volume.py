"""Tests for the volume module."""

from pathlib import Path

import pytest

from mini.volume import LocalVolume, data_dir_context, get_data_dir


# ---------------------------------------------------------------------------
# Context var tests
# ---------------------------------------------------------------------------


def test_get_data_dir_raises_outside_context():
    """get_data_dir() raises RuntimeError when called outside a job."""
    with pytest.raises(RuntimeError, match='No data directory'):
        get_data_dir()


def test_get_data_dir_returns_path_inside_context():
    """get_data_dir() returns the configured path inside data_dir_context."""
    with data_dir_context(Path('/some/path')):
        assert get_data_dir() == Path('/some/path')


def test_get_data_dir_resets_after_context():
    """get_data_dir() raises again after the context exits."""
    with data_dir_context(Path('/tmp/vol')):
        pass
    with pytest.raises(RuntimeError):
        get_data_dir()


def test_nested_contexts():
    """Inner context overrides outer; outer restored after inner exits."""
    with data_dir_context(Path('/outer')):
        assert get_data_dir() == Path('/outer')
        with data_dir_context(Path('/inner')):
            assert get_data_dir() == Path('/inner')
        assert get_data_dir() == Path('/outer')


# ---------------------------------------------------------------------------
# LocalVolume tests
# ---------------------------------------------------------------------------


def test_local_volume_creates_directory(tmp_path):
    """LocalVolume creates the directory on init."""
    vol_path = tmp_path / 'experiment-1'
    vol = LocalVolume(vol_path)
    assert vol.path == vol_path
    assert vol_path.is_dir()


def test_local_volume_path(tmp_path):
    """path returns the configured directory."""
    vol = LocalVolume(tmp_path / 'data')
    assert vol.path == tmp_path / 'data'


def test_local_volume_upload_file(tmp_path):
    """upload copies a single file into the volume."""
    vol = LocalVolume(tmp_path / 'vol')

    # Create a source file
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'data.csv').write_text('a,b,c')

    vol.upload(src / 'data.csv', 'input/data.csv')
    assert (vol.path / 'input' / 'data.csv').read_text() == 'a,b,c'


def test_local_volume_upload_directory(tmp_path):
    """upload copies a directory tree into the volume."""
    vol = LocalVolume(tmp_path / 'vol')

    # Create source directory with files
    src = tmp_path / 'src' / 'dataset'
    src.mkdir(parents=True)
    (src / 'train.csv').write_text('train')
    (src / 'test.csv').write_text('test')

    vol.upload(src, 'input/dataset')
    assert (vol.path / 'input' / 'dataset' / 'train.csv').read_text() == 'train'
    assert (vol.path / 'input' / 'dataset' / 'test.csv').read_text() == 'test'


def test_local_volume_download_file(tmp_path):
    """download copies a single file from the volume to a local path."""
    vol = LocalVolume(tmp_path / 'vol')

    # Put a file in the volume
    (vol.path / 'models').mkdir()
    (vol.path / 'models' / 'best.pt').write_bytes(b'model-data')

    dst = tmp_path / 'local'
    dst.mkdir()
    vol.download('models/best.pt', dst / 'best.pt')
    assert (dst / 'best.pt').read_bytes() == b'model-data'


def test_local_volume_download_directory(tmp_path):
    """download copies a directory tree from the volume."""
    vol = LocalVolume(tmp_path / 'vol')

    # Put files in the volume
    (vol.path / 'models' / 'run-1').mkdir(parents=True)
    (vol.path / 'models' / 'run-1' / 'weights.pt').write_bytes(b'w1')
    (vol.path / 'models' / 'run-1' / 'config.json').write_text('{}')

    dst = tmp_path / 'local'
    vol.download('models/run-1', dst)
    assert (dst / 'weights.pt').read_bytes() == b'w1'
    assert (dst / 'config.json').read_text() == '{}'


# ---------------------------------------------------------------------------
# Integration: get_data_dir() inside apparatus-mapped functions
# ---------------------------------------------------------------------------


def test_local_apparatus_provides_data_dir():
    """get_data_dir() returns a valid Path inside a LocalApparatus-mapped function."""
    from mini.local_apparatus import LocalApparatus

    captured_dirs: list[Path] = []

    def fn(x):
        captured_dirs.append(get_data_dir())
        return x

    app = LocalApparatus('test-vol', max_workers=1)
    results = list(app.map(fn, [1, 2]))
    assert results == [1, 2]
    assert len(captured_dirs) == 2
    assert all(isinstance(d, Path) for d in captured_dirs)
    # All jobs in the same run share the same data dir
    assert captured_dirs[0] == captured_dirs[1]


def test_local_apparatus_data_dir_exists():
    """The data directory returned by get_data_dir() exists on disk."""
    from mini.local_apparatus import LocalApparatus

    def fn(x):
        d = get_data_dir()
        assert d.is_dir(), f'{d} is not a directory'
        return x

    app = LocalApparatus('test-vol-exists', max_workers=1)
    results = list(app.map(fn, [1]))
    assert results == [1]


def test_local_apparatus_custom_data_dir(tmp_path):
    """LocalApparatus accepts a custom data_dir."""
    from mini.local_apparatus import LocalApparatus

    custom = tmp_path / 'my-data'

    def fn(x):
        return get_data_dir()

    app = LocalApparatus('test', max_workers=1, data_dir=custom)
    results = list(app.map(fn, [1]))
    assert results[0] == custom
    assert custom.is_dir()
