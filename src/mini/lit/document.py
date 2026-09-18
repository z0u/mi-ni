"""
Literate documents: prose and executable Python cells, woven into one document.

A document comes in two spellings of the same thing:

- **Python** (``.py``, the recommended one): a plain module, so ruff, ty, and the IDE see every cell. A top-level string literal is prose, and the code between two prose strings is a cell. A ``# %%`` comment at column 0 (the percent format, so VS Code offers "Run Cell" on it) is optional: it splits a cell, and carries that cell's options (``# %% hide``). Metadata is ``# key: value`` comment lines at the top of the file (``# title:``, ``# code: hide``). Write prose with math or other backslashes as a raw string (``r'''…'''``).
- **Markdown** (``.md``): a fenced block whose info string is ``{python}`` is a cell; everything else is prose, and the same keys go in ``---`` front matter.

Cells run top to bottom in one shared namespace, and the prose between them is a Jinja template rendered against that namespace *as it stands at that point*, so ``{{ best.mean }}``-style interpolation, ``{% for %}`` loops for tables, and helper calls like ``{{ h2_figure(res) }}`` all work without a notebook runtime. The result of weaving is plain Markdown (:attr:`Woven.markdown`), which :mod:`mini.lit.page` turns into HTML.

Only two things here are not plain Python or plain Markdown: a cell's last expression is displayed (a ``str`` is Markdown, an object with ``_repr_html_`` is HTML, a matplotlib figure is saved and shown), and :func:`stop` ends execution early — the rest of the prose still renders, with every unresolved name shown as a *pending* mark, so a preregistration reads whole before its results exist.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import html
import io
import re
import sys
import textwrap
import time
import tokenize
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import jinja2

from mini.reports import Publisher, current_publisher, use_publisher

__all__ = ["Document", "Prose", "Cell", "Woven", "Stop", "stop", "parse", "Runner"]

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
MARKER_RE = re.compile(r"^# %%(.*)$")
HEADER_RE = re.compile(r"^#\s*([\w-]+):\s*(.+?)\s*$")
CELL_INFO_RE = re.compile(r"^\s*\{python\}(.*)$")
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Prose:
    text: str
    line: int


@dataclass(frozen=True)
class Cell:
    source: str
    line: int
    options: frozenset[str] = frozenset()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    path: Path
    meta: dict[str, str]
    segments: tuple[Prose | Cell, ...]

    @property
    def cells(self) -> list[Cell]:
        return [s for s in self.segments if isinstance(s, Cell)]

    @property
    def title(self) -> str:
        if t := self.meta.get("title"):
            return t
        for s in self.segments:
            if isinstance(s, Prose) and (m := H1_RE.search(s.text)):
                return m.group(1).strip()
        return self.path.stem

    @property
    def show_code(self) -> bool:
        return self.meta.get("code", "show") != "hide"


def _front_matter(text: str) -> tuple[dict[str, str], int]:
    meta: dict[str, str] = {}
    if not (m := FRONT_MATTER_RE.match(text)):
        return meta, 0
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, m.end()


class _Splitter:
    """The fence state machine behind :func:`parse`: lines in, prose and cells out."""

    def __init__(self) -> None:
        self.segments: list[Prose | Cell] = []
        self._buf: list[str] = []
        self._buf_line = 1
        self._fence: str | None = None  # the open fence token, cell or not
        self._cell_opts: frozenset[str] | None = None  # set while the open fence is a cell

    def _closes(self, token: str, info: str) -> bool:
        f = self._fence
        return f is not None and token[0] == f[0] and len(token) >= len(f) and not info.strip()

    def _flush(self, as_cell: bool) -> None:
        if as_cell:
            self.segments.append(Cell("".join(self._buf), self._buf_line, self._cell_opts or frozenset()))
        elif self._buf:
            self.segments.append(Prose("".join(self._buf), self._buf_line))
        self._buf = []

    def feed(self, i: int, line: str) -> None:
        m = FENCE_RE.match(line)
        if m and self._fence is None:
            self._fence = m.group(1)
            if cm := CELL_INFO_RE.match(m.group(2)):
                self._flush(as_cell=False)
                self._cell_opts = frozenset(cm.group(1).split())
                self._buf_line = i + 1
                return
        elif m and self._closes(m.group(1), m.group(2)):
            self._fence = None
            if self._cell_opts is not None:
                self._flush(as_cell=True)
                self._cell_opts = None
                self._buf_line = i + 1
                return
        if not self._buf:
            self._buf_line = i if self._cell_opts is None else self._buf_line
        self._buf.append(line)

    def finish(self) -> list[Prose | Cell]:
        self._flush(
            as_cell=self._cell_opts is not None
        )  # an unterminated cell is kept, as CommonMark keeps an unterminated fence
        return self.segments


def parse(path: Path | str, text: str | None = None) -> Document:
    """Split a document into prose and cells, with its metadata: the ``.py`` spelling or the ``.md`` one, by suffix."""
    path = Path(path)
    text = path.read_text() if text is None else text
    return _parse_md(path, text) if path.suffix == ".md" else _parse_py(path, text)


def _parse_py(path: Path, text: str) -> Document:
    """The Python spelling.

    Metadata is the run of ``# key: value`` comment lines before the first code. A top-level expression statement that is a string literal is prose, dedented, and the code between prose statements is a cell. A ``# %%`` comment at column 0 (found by tokenizing, so one inside a string is text) also starts a cell, and carries its options, which last until the next marker or prose; Jupytext's ``[markdown]`` tag is tolerated and ignored. A string anywhere else (a docstring in a function, a value) is code, so a *variable* docstring hung under a constant reads as prose here — write it as a comment.
    """
    lines = text.splitlines(keepends=True)
    segments: list[Prose | Cell] = []
    opts: frozenset[str] = frozenset()
    cursor = 1

    def cell(start: int, end: int) -> None:  # the code on lines start..end, as a cell if there is any
        body = lines[start - 1 : end]
        while body and not body[0].strip():
            body, start = body[1:], start + 1
        while body and not body[-1].strip():
            body = body[:-1]
        if any(line.strip() and not line.lstrip().startswith("#") for line in body):  # comments alone are not a cell
            segments.append(Cell("".join(body), start, opts))

    for line, end, payload in _py_events(path, text):
        cell(cursor, line - 1)
        if isinstance(payload, frozenset):
            opts = payload
        else:
            segments.append(Prose(textwrap.dedent(payload.value.value).strip("\n") + "\n", line))
            opts = frozenset()
        cursor = end + 1
    cell(cursor, len(lines))
    return Document(path, _py_header(text), tuple(segments))


def _py_events(path: Path, text: str) -> list[tuple[int, int, Any]]:
    """The cell boundaries in order: ``(line, end line, payload)``, the payload a marker's options or a prose statement."""
    events: list[tuple[int, int, Any]] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and tok.start[1] == 0 and (m := MARKER_RE.match(tok.string)):
            events.append((tok.start[0], tok.start[0], frozenset(m.group(1).replace("[markdown]", " ").split())))
    for node in ast.parse(text, str(path)).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            events.append((node.lineno, node.end_lineno or node.lineno, node))
    return sorted(events, key=lambda e: e[0])


def _py_header(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith("#"):
            break
        if m := HEADER_RE.match(line):
            meta[m.group(1)] = m.group(2)
    return meta


def _parse_md(path: Path, text: str) -> Document:
    """The Markdown spelling.

    Front matter is the optional leading ``---`` block of ``key: value`` lines. Fences follow CommonMark: a closing fence uses the opener's character, at least as long, with nothing after it, so a cell whose source contains backticks just uses a longer fence. A ``{python}`` fence *inside* another fence (an example in a `````markdown````` block) is prose.
    """
    meta, body_start = _front_matter(text)
    first_line = text.count("\n", 0, body_start) + 1
    splitter = _Splitter()
    for i, line in enumerate(text[body_start:].splitlines(keepends=True), start=first_line):
        splitter.feed(i, line)
    return Document(path, meta, tuple(splitter.finish()))


class Stop(Exception):
    """Raised by :func:`stop` to end execution early, showing *message* (Markdown) in place of the cell."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


def stop(message: str = "") -> NoReturn:
    """End the document here: later cells do not run, and later prose renders with pending marks.

    The Marimo idiom this replaces is ``mo.stop(res is None, mo.md(...))``; write ``if res is None: stop("...")``. It never returns, so a type checker narrows the guarded name past the call.
    """
    raise Stop(message)


class _Pending(jinja2.Undefined):
    """An undefined name after :func:`stop`: renders as a mark, and absorbs calls, attributes, and items so ``{{ fig(res).x }}`` renders as one mark rather than raising."""

    def __str__(self) -> str:
        return f'<mark class="pending">{html.escape(self._undefined_name or "…")}</mark>'

    def __call__(self, *a: Any, **k: Any) -> _Pending:  # ty: ignore[invalid-method-override]
        return self

    def __getattr__(self, name: str) -> _Pending:
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __getitem__(self, key: Any) -> _Pending:  # ty: ignore[invalid-method-override]
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False


def _environment(undefined: type[jinja2.Undefined]) -> jinja2.Environment:
    # ``{#`` is a Jinja comment by default, and ``{#id}`` a Markdown attribute — so comments are ``{## … ##}``.
    return jinja2.Environment(
        undefined=undefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        comment_start_string="{##",
        comment_end_string="##}",
        autoescape=False,
    )


_STRICT = _environment(jinja2.StrictUndefined)
_LENIENT = _environment(_Pending)


@dataclass
class CellOutput:
    cell: Cell
    stdout: str = ""
    value: str = ""  # displayed as Markdown
    error: str | None = None
    stopped: bool = False
    seconds: float = 0.0


@dataclass
class Woven:
    doc: Document
    markdown: str
    outputs: list[CellOutput]
    seconds: float
    stopped: bool = False
    cells_run: int = 0

    @property
    def errors(self) -> list[CellOutput]:
        return [o for o in self.outputs if o.error]


def display(value: Any, *, publish: Publisher | None, name: str) -> str:
    """A cell's last expression as Markdown: strings pass through, HTML reprs and figures become HTML islands."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for attr in ("_repr_markdown_", "_repr_html_"):
        if (r := getattr(value, attr, None)) is not None and (rendered := r()) is not None:
            return rendered
    if type(value).__module__.startswith("matplotlib.") and hasattr(value, "savefig"):
        return _figure_html(value, publish=publish, name=name)
    return f"<pre><code>{html.escape(repr(value))}</code></pre>"


def _figure_html(fig: Any, *, publish: Publisher | None, name: str) -> str:
    import base64

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=192, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    png = buf.getvalue()
    src = (
        publish.asset_url(png, name=f"{name}.png")
        if publish is not None
        else f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
    )
    w = int.from_bytes(png[16:20], "big")  # PNG width from IHDR; drawn at 192 dpi, shown at 96
    return f'<figure><img src="{src}" width="{w // 2}" style="max-width:100%;height:auto" alt=""></figure>'


class Runner:
    """Runs a document's cells and weaves the result, keeping enough state to make the next run cheap.

    Cells run top to bottom in one namespace, so a run is defined by the prefix of cells before it. The runner keeps a shallow snapshot of the namespace after each cell; on the next :meth:`weave`, cells whose source (and whose predecessors' sources) are unchanged are skipped and the namespace restored from the snapshot, which is the REPL rule: editing cell *k* re-runs *k* onward, and a prose-only edit re-runs nothing. Objects mutated in place by a later cell escape the snapshot; ``reset()`` (or restarting) clears everything.

    *publish* receives every asset the document writes (figures through ``themed``, or via ``display``); the runner installs it as the current publisher while cells run.
    """

    def __init__(self, path: Path | str, *, publish: Publisher | None = None):
        self.path = Path(path).resolve()
        self.publish = publish
        # Cells run in a real module, registered in ``sys.modules``: ``@dataclass`` and
        # ``inspect`` resolve a definition's module by name, and functions defined in one
        # run must keep seeing the live namespace after a partial re-run restores it in
        # place (``__globals__`` is this dict, so it is never replaced, only refilled).
        tag = hashlib.sha256(str(self.path).encode()).hexdigest()[:8]
        self._module_name = f"__lit_{self.path.stem}_{tag}"
        self._module = sys.modules[self._module_name] = types.ModuleType(self._module_name)
        self._digests: list[str] = []
        self._snapshots: list[dict[str, Any]] = []
        self._outputs: list[CellOutput] = []
        self._templates: dict[str, jinja2.Template] = {}

    def reset(self) -> None:
        self._digests.clear()
        self._snapshots.clear()
        self._outputs.clear()

    def _template(self, text: str, env: jinja2.Environment) -> jinja2.Template:
        key = f"{id(env)}:{hashlib.sha256(text.encode()).hexdigest()}"
        if (t := self._templates.get(key)) is None:
            t = self._templates[key] = env.from_string(text)
        return t

    def _run_cell(self, cell: Cell, index: int, ns: dict[str, Any]) -> CellOutput:
        out = CellOutput(cell)
        filename = str(self.path)
        t0 = time.perf_counter()
        buf = io.StringIO()
        try:
            tree = ast.parse(cell.source, filename)
            ast.increment_lineno(tree, cell.line - 1)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, filename, "exec"), ns)
                if isinstance(last, ast.Expr):
                    value = eval(compile(ast.Expression(last.value), filename, "eval"), ns)
                    out.value = display(value, publish=self.publish, name=f"cell-{index}")
        except Stop as s:
            out.value = s.message
            out.stopped = True
        except Exception:
            tb = traceback.format_exc()
            marker = f'File "{filename}"'  # trim the runner's own frames: the reader wants the document's line
            out.error = tb[tb.find(marker) :] if marker in tb else tb
        out.stdout = buf.getvalue()
        out.seconds = time.perf_counter() - t0
        return out

    def _restore(self, cells: list[Cell]) -> tuple[dict[str, Any], int]:
        """Restore the namespace to just after the last unchanged cell; return it and how many cells that keeps."""
        digests = [c.digest for c in cells]
        keep = 0
        while keep < min(len(digests), len(self._digests)) and digests[keep] == self._digests[keep]:
            keep += 1
        self._digests, self._snapshots, self._outputs = (
            self._digests[:keep],
            self._snapshots[:keep],
            self._outputs[:keep],
        )
        ns = self._module.__dict__
        ns.clear()
        ns.update(
            self._snapshots[keep - 1]
            if keep
            else {"__name__": self._module_name, "__file__": str(self.path), "stop": stop}
        )
        return ns, keep

    def _cell_markdown(self, doc: Document, cell: Cell, out: CellOutput) -> list[str]:
        parts = []
        if "show" in cell.options or (doc.show_code and "hide" not in cell.options):
            parts.append(_code_block(cell.source))
        if out.stdout:
            parts.append(f'<pre class="stdout">{html.escape(out.stdout)}</pre>\n')
        if out.value:
            parts.append(f"\n{out.value}\n")
        if out.error:
            parts.append(_error_block(out.error))
        return parts

    def weave(self, doc: Document | None = None) -> Woven:
        """Run what changed and weave the whole document into Markdown."""
        t0 = time.perf_counter()
        doc = doc or parse(self.path)
        ns, keep = self._restore(doc.cells)
        if (doc_dir := str(self.path.parent)) not in sys.path:
            sys.path.insert(0, doc_dir)
        previous = current_publisher()
        use_publisher(self.publish)
        outputs: list[CellOutput] = []
        parts: list[str] = []
        stopped = False
        cells_run = 0
        index = 0
        try:
            for seg in doc.segments:
                if isinstance(seg, Prose):
                    parts.append(self._prose(seg, ns, lenient=stopped))
                    continue
                if stopped:
                    continue
                if index < keep:
                    out = self._outputs[index]
                else:
                    out = self._run_cell(seg, index, ns)
                    cells_run += 1
                    if out.error is None:
                        self._digests.append(seg.digest)
                        self._snapshots.append(dict(ns))
                        self._outputs.append(out)
                index += 1
                outputs.append(out)
                parts.extend(self._cell_markdown(doc, seg, out))
                stopped = bool(out.error) or out.stopped
        finally:
            use_publisher(previous)
        return Woven(doc, "\n".join(parts), outputs, time.perf_counter() - t0, stopped=stopped, cells_run=cells_run)

    def _prose(self, seg: Prose, ns: dict[str, Any], *, lenient: bool) -> str:
        try:
            return self._template(seg.text, _LENIENT if lenient else _STRICT).render(ns)
        except jinja2.TemplateError as e:
            return _error_block(f"{self.path}:{seg.line}: {type(e).__name__}: {e}")


def _code_block(source: str) -> str:
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", source)), default=0) + 1)
    return f"\n{fence}python\n{source.rstrip()}\n{fence}\n"


def _error_block(text: str) -> str:
    return f'\n<pre class="error">{html.escape(text)}</pre>\n'
