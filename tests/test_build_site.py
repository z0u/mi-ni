"""Tests for the static-site builder's author-link resolver (pure policy)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'build_site', Path(__file__).resolve().parent.parent / 'scripts' / 'build_site.py'
)
assert _SPEC and _SPEC.loader
build_site = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_site)


@pytest.fixture
def resolver() -> 'build_site.LinkResolver':
    return build_site.LinkResolver(
        render_map={
            'probe/report.py': 'probe/report.html',
            'acts/report.py': 'acts/report.html',
            'guide.md': 'guide.html',
        },
        source_files=frozenset({'probe/experiment.py', 'acts/experiment.py', 'probe/report.py'}),
        site_base='https://o.github.io/r/',
        source_base='https://github.com/o/r/blob/main/',
    )


def test_rendered_link_is_absolute_pages_url_when_externalizing(resolver):
    got = resolver.resolve('../acts/report.py', from_dir='probe', externalizing=True)
    assert got == 'https://o.github.io/r/acts/report.html'


def test_rendered_link_stays_relative_when_localizing(resolver):
    # No <base> locally, so a relative .html link navigates within _site.
    got = resolver.resolve('../acts/report.py', from_dir='probe', externalizing=False)
    assert got == '../acts/report.html'


def test_source_file_resolves_to_github(resolver):
    got = resolver.resolve('./experiment.py', from_dir='probe', externalizing=True)
    assert got == 'https://github.com/o/r/blob/main/docs/probe/experiment.py'


def test_fragment_is_preserved(resolver):
    got = resolver.resolve('../acts/report.py#cell-3', from_dir='probe', externalizing=True)
    assert got == 'https://o.github.io/r/acts/report.html#cell-3'


def test_repo_source_link_outside_docs_resolves_to_github(resolver):
    # A report linking to its source modules escapes docs/ but stays in the repo;
    # it should resolve to the GitHub source so it survives the asset <base>.
    # (Fixture has repo_root=None, so existence is trusted.)
    assert (
        resolver.resolve('../src/experiment', from_dir='.', externalizing=True)
        == 'https://github.com/o/r/blob/main/src/experiment'
    )
    assert (
        resolver.resolve('../../src/experiment/model/README.md#gate', from_dir='gpt-sweep', externalizing=True)
        == 'https://github.com/o/r/blob/main/src/experiment/model/README.md#gate'
    )


def test_link_escaping_the_repo_root_is_unresolved(resolver):
    assert resolver.resolve('../../../etc/passwd', from_dir='probe', externalizing=True) is None


def test_missing_repo_source_target_is_unresolved(tmp_path):
    # With a repo_root set, a link to a path that doesn't exist is left to warn.
    r = build_site.LinkResolver(
        render_map={},
        source_files=frozenset(),
        site_base=None,
        source_base='https://github.com/o/r/blob/main/',
        repo_root=tmp_path,
    )
    assert r.resolve('../src/nope', from_dir='.', externalizing=True) is None
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'real.py').write_text('')
    assert (
        r.resolve('../src/real.py', from_dir='.', externalizing=True) == 'https://github.com/o/r/blob/main/src/real.py'
    )


def test_external_and_anchored_links_are_left_alone(resolver):
    assert resolver.resolve('https://example.com', from_dir='probe', externalizing=True) is None
    assert resolver.resolve('#section', from_dir='probe', externalizing=True) is None
    assert resolver.resolve('/absolute', from_dir='probe', externalizing=True) is None


def test_unknown_target_is_unresolved(resolver):
    assert resolver.resolve('./nope.py', from_dir='probe', externalizing=True) is None


def test_missing_bases_degrade_to_unresolved():
    r = build_site.LinkResolver(
        render_map={'acts/report.py': 'acts/report.html'},
        source_files=frozenset({'probe/experiment.py'}),
        site_base=None,
        source_base=None,
    )
    # Externalizing needs an absolute target; with no base it can't make one.
    assert r.resolve('../acts/report.py', from_dir='probe', externalizing=True) is None
    # …but localize still keeps rendered links relative (no base needed).
    assert r.resolve('../acts/report.py', from_dir='probe', externalizing=False) == '../acts/report.html'
