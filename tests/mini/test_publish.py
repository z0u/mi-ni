"""Tests for the HFStore publishing sink."""

from unittest.mock import patch

import pytest

from mini.publish import HFStore, artifacts_repo


@pytest.fixture
def api():
    """Patch HfApi so no network calls happen; yield the mock instance."""
    with patch('huggingface_hub.HfApi') as cls:
        yield cls.return_value


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def test_dataset_url(api):
    store = HFStore('z0u/artifacts')  # dataset is the default
    assert store.url('gpt/loss.mp4') == 'https://huggingface.co/datasets/z0u/artifacts/resolve/main/gpt/loss.mp4'


def test_model_url(api):
    store = HFStore('z0u/nanogpt', repo_type='model')
    assert store.url('checkpoint.eqx') == 'https://huggingface.co/z0u/nanogpt/resolve/main/checkpoint.eqx'


def test_space_url(api):
    store = HFStore('z0u/demo', repo_type='space')
    assert store.url('app.py') == 'https://huggingface.co/spaces/z0u/demo/resolve/main/app.py'


def test_url_honours_revision(api):
    store = HFStore('z0u/artifacts', revision='v1')
    assert store.url('x.png').endswith('/resolve/v1/x.png')
    assert store.url('x.png', revision='v2').endswith('/resolve/v2/x.png')


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def test_token_from_env(api, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'tok-abc')
    monkeypatch.delenv('HUGGINGFACE_TOKEN', raising=False)
    with patch('huggingface_hub.HfApi') as cls:
        HFStore('z0u/artifacts')
        cls.assert_called_once_with(token='tok-abc')


def test_explicit_token_wins(monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'env-tok')
    with patch('huggingface_hub.HfApi') as cls:
        HFStore('z0u/artifacts', token='explicit')
        cls.assert_called_once_with(token='explicit')


# ---------------------------------------------------------------------------
# publish()
# ---------------------------------------------------------------------------


def test_publish_file(api, tmp_path):
    f = tmp_path / 'model.eqx'
    f.write_bytes(b'weights')

    store = HFStore('z0u/artifacts')
    url = store.publish(f, 'nanogpt/model.eqx')

    api.upload_file.assert_called_once()
    kwargs = api.upload_file.call_args.kwargs
    assert kwargs['repo_id'] == 'z0u/artifacts'
    assert kwargs['repo_type'] == 'dataset'
    assert kwargs['path_or_fileobj'] == str(f)
    assert kwargs['path_in_repo'] == 'nanogpt/model.eqx'
    assert url == 'https://huggingface.co/datasets/z0u/artifacts/resolve/main/nanogpt/model.eqx'


def test_publish_directory(api, tmp_path):
    d = tmp_path / 'run-1'
    d.mkdir()
    (d / 'a.txt').write_text('a')

    store = HFStore('z0u/artifacts')
    url = store.publish(d, 'nanogpt/run-1')

    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs['folder_path'] == str(d)
    assert kwargs['path_in_repo'] == 'nanogpt/run-1'
    assert url.endswith('/resolve/main/nanogpt/run-1')


def test_publish_defaults_remote_to_basename(api, tmp_path):
    f = tmp_path / 'loss.png'
    f.write_bytes(b'\x89PNG')

    HFStore('z0u/artifacts').publish(f)

    assert api.upload_file.call_args.kwargs['path_in_repo'] == 'loss.png'


def test_publish_creates_repo_once(api, tmp_path):
    f = tmp_path / 'a.txt'
    f.write_text('a')
    g = tmp_path / 'b.txt'
    g.write_text('b')

    store = HFStore('z0u/artifacts', private=True)
    store.publish(f)
    store.publish(g)

    api.create_repo.assert_called_once_with(
        repo_id='z0u/artifacts',
        repo_type='dataset',
        private=True,
        exist_ok=True,
    )


def test_publish_skips_create_when_disabled(api, tmp_path):
    f = tmp_path / 'a.txt'
    f.write_text('a')

    HFStore('z0u/artifacts', create=False).publish(f)

    api.create_repo.assert_not_called()


# ---------------------------------------------------------------------------
# fetch()
# ---------------------------------------------------------------------------


def test_fetch_file(api, tmp_path):
    cached = tmp_path / 'cache' / 'model.eqx'
    cached.parent.mkdir()
    cached.write_bytes(b'weights')
    dst = tmp_path / 'out' / 'model.eqx'

    store = HFStore('z0u/artifacts', token='tok')
    with patch('huggingface_hub.hf_hub_download', return_value=str(cached)) as dl:
        store.fetch('nanogpt/model.eqx', dst)

    dl.assert_called_once()
    assert dl.call_args.kwargs['filename'] == 'nanogpt/model.eqx'
    assert dst.read_bytes() == b'weights'


def test_fetch_directory(api, tmp_path):
    snap = tmp_path / 'snap'
    (snap / 'nanogpt' / 'run-1').mkdir(parents=True)
    (snap / 'nanogpt' / 'run-1' / 'a.txt').write_text('a')
    dst = tmp_path / 'out'

    store = HFStore('z0u/artifacts')
    with patch('huggingface_hub.snapshot_download', return_value=str(snap)) as dl:
        store.fetch('nanogpt/run-1/', dst)

    assert dl.call_args.kwargs['allow_patterns'] == ['nanogpt/run-1/*']
    assert (dst / 'a.txt').read_text() == 'a'


# ---------------------------------------------------------------------------
# config resolution — artifacts_repo() / from_config()
# ---------------------------------------------------------------------------


def test_artifacts_repo_from_env(monkeypatch):
    monkeypatch.setenv('HF_ARTIFACTS_REPO', 'me/from-env')
    assert artifacts_repo() == 'me/from-env'


def test_artifacts_repo_from_pyproject(monkeypatch, tmp_path):
    monkeypatch.delenv('HF_ARTIFACTS_REPO', raising=False)
    (tmp_path / 'pyproject.toml').write_text('[tool.mini]\nartifacts_repo = "me/from-toml"\n')
    monkeypatch.chdir(tmp_path)
    assert artifacts_repo() == 'me/from-toml'


def test_artifacts_repo_env_wins_over_pyproject(monkeypatch, tmp_path):
    (tmp_path / 'pyproject.toml').write_text('[tool.mini]\nartifacts_repo = "me/from-toml"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HF_ARTIFACTS_REPO', 'me/from-env')
    assert artifacts_repo() == 'me/from-env'


def test_artifacts_repo_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv('HF_ARTIFACTS_REPO', raising=False)
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "x"\n')  # no [tool.mini]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match='No artifacts repo'):
        artifacts_repo()


def test_from_config_uses_resolved_repo(api, monkeypatch):
    monkeypatch.setenv('HF_ARTIFACTS_REPO', 'me/pub')
    store = HFStore.from_config()
    assert store.repo_id == 'me/pub'
    assert store.url('a.png') == 'https://huggingface.co/datasets/me/pub/resolve/main/a.png'
