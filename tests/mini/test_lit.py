"""Tests for ``mini.lit``: parsing, weaving, incremental re-runs, the memo, and the page."""

import textwrap
from pathlib import Path

import pytest

from mini.lit import Runner, memo, parse, render
from mini.lit import set_cache_dir
from mini.lit.document import Cell, Prose
from mini.lit.page import page, to_html
from mini.reports import Publisher


def write(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip())
    return p


class TestParse:
    def test_prose_cells_and_front_matter(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                """
                ---
                title: T
                code: hide
                ---
                # H

                ```{python}
                x = 1
                ```
                after
                """,
            )
        )
        assert doc.meta == {"title": "T", "code": "hide"}
        assert doc.title == "T"
        assert not doc.show_code
        assert [type(s).__name__ for s in doc.segments] == ["Prose", "Cell", "Prose"]
        cell = doc.cells[0]
        assert cell == Cell("x = 1\n", line=8)
        assert doc.segments[2] == Prose("after\n", line=10)

    def test_title_from_h1_and_display_fences_are_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                """
                # The title

                ```python
                not_a_cell = True
                ```

                ````markdown
                ```{python}
                nested = "still prose"
                ```
                ````
                """,
            )
        )
        assert doc.title == "The title"
        assert doc.cells == []
        assert isinstance(doc.segments[0], Prose) and "nested" in doc.segments[0].text

    def test_longer_fence_closes_a_cell_with_backticks(self, tmp_path):
        doc = parse(write(tmp_path, '````{python}\ns = "```"\n````\ntail\n'))
        assert doc.cells[0].source == 's = "```"\n'
        assert doc.segments[-1] == Prose("tail\n", 4)


class TestWeave:
    def test_interpolation_loop_display_and_stdout(self, tmp_path):
        p = write(
            tmp_path,
            """
            ```{python}
            xs = [1, 2, 3]
            print("hello")
            "*shown*"
            ```
            Total {{ xs|sum }}.
            {% for x in xs %}
            - item {{ x }}
            {% endfor %}
            """,
        )
        w = Runner(p).weave()
        assert w.errors == [] and not w.stopped
        assert "```python\nxs = [1, 2, 3]" in w.markdown  # code shown by default
        assert '<pre class="stdout">hello\n</pre>' in w.markdown
        assert "*shown*" in w.markdown
        assert "Total 6." in w.markdown
        assert "- item 1\n- item 2\n- item 3\n" in w.markdown

    def test_hidden_code(self, tmp_path):
        p = write(tmp_path, "---\ncode: hide\n---\n```{python}\nsecret = 1\n```\nafter {{ secret }}\n")
        md = Runner(p).weave().markdown
        assert "secret = 1" not in md and "after 1" in md

    def test_stop_renders_rest_with_pending_marks(self, tmp_path):
        p = write(
            tmp_path,
            """
            ```{python}
            res = None
            if res is None:
                stop("_pending_")
            summary = res["x"]
            ```
            Value {{ summary.m }} and {{ fig(res) }}.
            ```{python}
            raise RuntimeError("must not run")
            ```
            """,
        )
        w = Runner(p).weave()
        assert w.stopped and w.errors == []
        assert "_pending_" in w.markdown
        assert 'Value <mark class="pending">summary</mark> and <mark class="pending">fig</mark>.' in w.markdown
        assert len(w.outputs) == 1

    def test_error_points_at_document_line_and_stops(self, tmp_path):
        p = write(tmp_path, "intro\n\n```{python}\nx = 1\n1 / 0\n```\nafter {{ x }}\n")
        w = Runner(p).weave()
        assert len(w.errors) == 1
        assert f'File "{p}", line 5' in (w.errors[0].error or "")
        assert "ZeroDivisionError" in w.markdown
        assert "after 1" in w.markdown  # x was bound before the error, prose still renders

    def test_undefined_name_in_prose_is_reported_not_raised(self, tmp_path):
        p = write(tmp_path, "{{ nope }}\n")
        w = Runner(p).weave()
        assert "UndefinedError" in w.markdown

    def test_dataclass_and_inspect_work_in_cells(self, tmp_path):
        p = write(
            tmp_path,
            """
            ```{python}
            import inspect
            from dataclasses import dataclass

            @dataclass
            class C:
                a: int

            def f():
                return 1

            inspect.getsource(f).strip()
            ```
            """,
        )
        w = Runner(p).weave()
        assert w.errors == []
        assert "def f():\n    return 1" in w.markdown

    def test_figure_is_saved_through_the_publisher(self, tmp_path):
        pytest.importorskip("matplotlib")
        p = write(tmp_path, "```{python}\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig\n```\n")
        pub = Publisher(asset_dir=tmp_path / "_assets")
        md = Runner(p, publish=pub).weave().markdown
        assert '<img src="_assets/cell-0.png"' in md
        assert (tmp_path / "_assets" / "cell-0.png").exists()


class TestIncremental:
    def test_only_changed_suffix_reruns(self, tmp_path):
        p = write(
            tmp_path, "```{python}\nlog = []\na = 1\n```\n```{python}\nlog.append('b')\nb = a + 1\n```\nb={{ b }}\n"
        )
        r = Runner(p)
        assert r.weave().cells_run == 2
        # prose-only edit: nothing re-runs
        p.write_text(p.read_text().replace("b={{ b }}", "B={{ b }}"))
        w = r.weave()
        assert w.cells_run == 0 and "B=2" in w.markdown
        # edit the second cell: only it re-runs, against the namespace after the first
        p.write_text(p.read_text().replace("b = a + 1", "b = a + 10"))
        w = r.weave()
        assert w.cells_run == 1 and "B=11" in w.markdown
        # edit the first cell: both re-run
        p.write_text(p.read_text().replace("a = 1", "a = 2"))
        w = r.weave()
        assert w.cells_run == 2 and "B=12" in w.markdown

    def test_functions_defined_earlier_see_the_restored_namespace(self, tmp_path):
        p = write(tmp_path, "```{python}\nk = 1\ndef f():\n    return k\n```\n```{python}\nout = f()\n```\n{{ out }}\n")
        r = Runner(p)
        r.weave()
        p.write_text(p.read_text().replace("out = f()", "out = f() + 1"))
        assert "\n2\n" in r.weave().markdown


class TestMemo:
    @pytest.fixture(autouse=True)
    def cache(self, tmp_path):
        set_cache_dir(tmp_path / "cache")
        yield
        set_cache_dir(None)

    def test_hit_across_processes_and_miss_on_source_change(self, tmp_path):
        p = write(
            tmp_path,
            "```{python}\nfrom mini.lit import memo\ncalls = []\n@memo\ndef f(x):\n    calls.append(x)\n    return x * 2\nf(3), f(3), len(calls)\n```\n",
        )
        assert "(6, 6, 1)" in Runner(p).weave().markdown
        # a second runner (a fresh process, as far as the memo is concerned) hits the disk cache
        set_cache_dir(tmp_path / "cache")  # clears the in-memory tier
        assert "(6, 6, 0)" in Runner(p).weave().markdown
        p.write_text(p.read_text().replace("x * 2", "x * 3"))
        assert "(9, 9, 1)" in Runner(p).weave().markdown

    def test_array_inputs_key_by_content(self):
        np = pytest.importorskip("numpy")
        calls = []

        @memo
        def g(a):
            calls.append(1)
            return float(a.sum())

        assert g(np.arange(3)) == g(np.arange(3)) == 3.0
        assert len(calls) == 1
        g(np.arange(4))
        assert len(calls) == 2

    def test_hit_requires_its_assets(self, tmp_path):
        pub = Publisher(asset_dir=tmp_path / "_assets")
        calls = []

        @memo
        def h():
            calls.append(1)
            return pub.asset_url(b"png", name="fig.png")

        from mini.reports import use_publisher

        use_publisher(pub)
        try:
            assert h() == h() == "_assets/fig.png" and len(calls) == 1
            (tmp_path / "_assets" / "fig.png").unlink()
            h()
            assert len(calls) == 2 and (tmp_path / "_assets" / "fig.png").exists()
        finally:
            use_publisher(None)


class TestRender:
    def test_writes_html_and_markdown(self, tmp_path):
        p = write(tmp_path, "# Hi\n\n```{python}\nv = 2\n```\nv is {{ v }} and \\(x^2\\).\n")
        r = render(p, out_dir=tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "<title>Hi</title>" in html and "v is 2" in html
        assert "katex" in html  # math present → KaTeX loaded
        assert "v is 2" in (tmp_path / "out" / "index.md").read_text()
        assert r.woven.errors == []

    def test_live_output_is_a_separate_tree(self, tmp_path, monkeypatch):
        from mini.lit.render import output_dir

        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        doc = tmp_path / "docs" / "foo" / "report.py"
        assert output_dir(doc) == tmp_path / ".mini" / "lit" / "foo"
        assert output_dir(doc, live=True) == tmp_path / ".mini" / "lit-live" / "foo"

    def test_page_without_math_skips_katex(self):
        assert "katex" not in page(to_html("plain"), title="t")

    def test_markdown_dialect(self):
        html = to_html("/// admonition | T\n    type: note\nbody\n///\n\nx[^1]\n\n[^1]: note\n\n| a |\n|---|\n| 1 |\n")
        assert 'class="admonition note"' in html
        assert 'class="footnote"' in html
        assert "<table>" in html


class TestParsePy:
    def test_header_and_string_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                '''
                # title: T
                # code: hide

                """
                # H
                """

                # a comment belongs to the cell below
                x = 1
                """
                after {{ x }}
                """
                y = 2
                ''',
                name="doc.py",
            )
        )
        assert doc.meta == {"title": "T", "code": "hide"}
        assert doc.title == "T" and not doc.show_code
        assert doc.segments == (
            Prose("# H\n", 4),
            Cell("# a comment belongs to the cell below\nx = 1\n", 8),
            Prose("after {{ x }}\n", 10),
            Cell("y = 2\n", 13),
        )

    def test_strings_in_code_are_not_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                '''
                def f():
                    """a docstring"""
                    return "# %% not special"

                s = """
                nor this
                """
                r"""prose with \\(x\\)"""
                ''',
                name="doc.py",
            )
        )
        assert [type(s).__name__ for s in doc.segments] == ["Cell", "Prose"]
        assert isinstance(doc.segments[0], Cell) and doc.segments[0].source.count('"""') == 4
        assert doc.segments[1] == Prose("prose with \\(x\\)\n", 8)

    def test_weaves_like_the_markdown_spelling(self, tmp_path):
        p = write(
            tmp_path,
            '"""\n# Doc\n"""\nxs = [1, 2]\n"""\n{% for x in xs %}\n- {{ x }}\n{% endfor %}\n"""\n',
            name="doc.py",
        )
        w = Runner(p).weave()
        assert w.errors == []
        assert "# Doc\n" in w.markdown and "- 1\n- 2\n" in w.markdown and "```python\nxs = [1, 2]\n```" in w.markdown
