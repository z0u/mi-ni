"""
Literate documents: prose and executable Python cells, woven into one document.

A document comes in two spellings of the same thing:

- **Python** (``.py``, the recommended one): a plain module, so ruff, ty, and the IDE see every cell. A top-level string literal is prose, and the code between two prose strings is a cell. Metadata is ``# key: value`` comment lines at the top of the file (``# title:``, and ``# code: show`` to include cell source in the output). Write prose with math or other backslashes as a raw string (``r'''…'''``).
- **Markdown** (``.md``): a fenced block whose info string is ``{python}`` is a cell; everything else is prose, and the same keys go in ``---`` front matter.

Cells run top to bottom in one shared namespace, and the prose between them is rendered against that namespace *as it stands at that point*. A plain string is a Jinja template, so ``{{ best.mean }}``-style interpolation, ``{% for %}`` loops for tables, and helper calls like ``{{ h2_figure(res) }}`` all work without a notebook runtime. An f-string (``rf'''…{best.mean:.2f}…'''``, in the ``.py`` spelling) is evaluated as the Python it is, one field at a time, so the names it reads are visible to ruff, ty, vulture, and go-to-definition; it is never templated, so a literal brace in one is doubled as usual. The result of weaving is plain Markdown (:attr:`Woven.markdown`), which :mod:`mini.lit.page` turns into HTML.

Only two things here are not plain Python or plain Markdown: a cell's last expression is displayed (a ``str`` is Markdown, an object with ``_repr_html_`` is HTML, a matplotlib figure is saved and shown), and :func:`stop` ends execution early — the rest of the prose still renders, with every unresolved name shown as a *pending* mark, so a preregistration reads whole before its results exist.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import html
import io
import re
import sys
import textwrap
import threading
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

import jinja2

from mini.reports import Publisher, current_publisher, use_publisher

__all__ = ["Document", "Prose", "Cell", "Woven", "Stop", "stop", "parse", "is_literate_document", "Runner"]

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADER_RE = re.compile(r"^#\s*([\w-]+):\s*(.+?)\s*$")
CELL_INFO_RE = re.compile(r"^\s*\{python\}\s*$")
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Prose:
    text: str  # dedented; for an f-string, its fields shown as ``{expr}`` placeholders
    line: int
    fstring: ast.JoinedStr | None = None  # set for an f-string: evaluated in the namespace rather than templated
    indent: str = ""  # what dedenting removed, to remove again from an f-string's evaluated text


@dataclass(frozen=True)
class Cell:
    source: str
    line: int

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
        """Whether cell source appears in the output: ``# code: show`` opts in; a report hides its plumbing by default."""
        return self.meta.get("code") == "show"


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
        self._in_cell = False  # whether the open fence is a cell

    def _closes(self, token: str, info: str) -> bool:
        f = self._fence
        return f is not None and token[0] == f[0] and len(token) >= len(f) and not info.strip()

    def _flush(self, as_cell: bool) -> None:
        if as_cell:
            self.segments.append(Cell("".join(self._buf), self._buf_line))
        elif self._buf:
            self.segments.append(Prose("".join(self._buf), self._buf_line))
        self._buf = []

    def feed(self, i: int, line: str) -> None:
        m = FENCE_RE.match(line)
        if m and self._fence is None:
            self._fence = m.group(1)
            if CELL_INFO_RE.match(m.group(2)):
                self._flush(as_cell=False)
                self._in_cell = True
                self._buf_line = i + 1
                return
        elif m and self._closes(m.group(1), m.group(2)):
            self._fence = None
            if self._in_cell:
                self._flush(as_cell=True)
                self._in_cell = False
                self._buf_line = i + 1
                return
        if not self._buf:
            self._buf_line = self._buf_line if self._in_cell else i
        self._buf.append(line)

    def finish(self) -> list[Prose | Cell]:
        self._flush(as_cell=self._in_cell)  # an unterminated cell is kept, as CommonMark keeps an unterminated fence
        return self.segments


def parse(path: Path | str, text: str | None = None) -> Document:
    """Split a document into prose and cells, with its metadata: the ``.py`` spelling or the ``.md`` one, by suffix."""
    path = Path(path)
    text = path.read_text() if text is None else text
    return _parse_md(path, text) if path.suffix == ".md" else _parse_py(path, text)


def _parse_py(path: Path, text: str) -> Document:
    """The Python spelling.

    Metadata is the run of ``# key: value`` comment lines before the first code. A top-level expression statement that is a string literal is prose, dedented, and the code between prose statements is a cell (comments alone are not one). A string anywhere else (a docstring in a function, a value) is code, so a *variable* docstring hung under a constant reads as prose here — write it as a comment.

    An f-string at the top level is prose too, evaluated rather than templated (see :meth:`Runner._prose`): its text here carries each field as a ``{expr}`` placeholder, which is also how it renders past a :func:`stop`.
    """
    lines = text.splitlines(keepends=True)
    segments: list[Prose | Cell] = []
    cursor = 1

    def prose(node: ast.expr, line: int) -> Prose | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Prose(_dedent(node.value), line)
        if isinstance(node, ast.JoinedStr):
            raw = _fstring_text(node)
            return Prose(_dedent(raw), line, fstring=node, indent=_indent_of(raw))
        return None

    def cell(start: int, end: int) -> None:  # the code on lines start..end, as a cell if there is any
        body = lines[start - 1 : end]
        while body and not body[0].strip():
            body, start = body[1:], start + 1
        while body and not body[-1].strip():
            body = body[:-1]
        if any(line.strip() and not line.lstrip().startswith("#") for line in body):
            segments.append(Cell("".join(body), start))

    for node in ast.parse(text, str(path)).body:
        if isinstance(node, ast.Expr) and (seg := prose(node.value, node.lineno)) is not None:
            cell(cursor, node.lineno - 1)
            segments.append(seg)
            cursor = (node.end_lineno or node.lineno) + 1
    cell(cursor, len(lines))
    return Document(path, _py_header(text), tuple(segments))


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def _indent_of(text: str) -> str:
    """The common leading whitespace :func:`textwrap.dedent` would remove from *text*."""
    indents = [m.group(1) for m in re.finditer(r"(?m)^([ \t]*)(?=\S)", text)]
    if not indents:
        return ""
    first = min(indents, key=len)
    return first if all(i.startswith(first) for i in indents) else ""


def _fstring_text(node: ast.JoinedStr) -> str:
    """An f-string's source text with each field as a ``{expr}`` placeholder (literal braces single, as they read)."""
    return "".join(_piece_text(v) for v in node.values)


def _piece_text(piece: ast.expr) -> str:
    if isinstance(piece, ast.FormattedValue):
        return "{" + _field_source(piece) + "}"
    assert isinstance(piece, ast.Constant), piece  # an f-string is constants and fields, nothing else
    return str(piece.value)


def _field_source(field: ast.FormattedValue) -> str:
    s = ast.unparse(field.value)
    if field.conversion != -1:
        s += "!" + chr(field.conversion)
    if isinstance(field.format_spec, ast.JoinedStr):
        s += ":" + _fstring_text(field.format_spec)
    return s


def is_literate_document(path: str | Path) -> bool:
    """Whether *path* is a literate ``.py`` rather than an ordinary module: it opens with a ``# title:`` header.

    The publishing checks use this to tell a document beside a report from one of the report's inputs. The ``.md`` spelling is not distinguished from any other Markdown file here.
    """
    p = Path(path)
    return p.suffix == ".py" and "title" in _py_header(p.read_text("utf-8", errors="ignore"))


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
    """An undefined name after :func:`stop`: renders as a mark, and absorbs whatever is done to it.

    Calls, attributes, items, arithmetic, and comparisons all give the same mark back, and a filter handed one returns it (see :func:`_absorbing`), so ``{{ "%.3f"|format(fig(res).x * 100) }}`` renders as one mark rather than raising. Iteration is empty, so a ``{% for %}`` table has no rows, and it is false in an ``{% if %}``.
    """

    def __str__(self) -> str:
        return f'<mark class="pending">{html.escape(self._undefined_name or "…")}</mark>'

    def _absorb(self, *a: Any, **k: Any) -> _Pending:
        return self

    __call__ = __getitem__ = _absorb  # ty: ignore[invalid-method-override]
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _absorb  # ty: ignore[invalid-method-override]
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pow__ = __rpow__ = __neg__ = __pos__ = _absorb  # ty: ignore[invalid-method-override]
    __lt__ = __le__ = __gt__ = __ge__ = _absorb  # ty: ignore[invalid-method-override]

    def __getattr__(self, name: str) -> _Pending:
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


def _absorbing(filter_fn: Callable[..., Any]) -> Callable[..., Any]:
    """*filter_fn*, returning the pending mark instead of running when any argument is one (``format`` would ask it for a float)."""

    @functools.wraps(filter_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for v in (*args, *kwargs.values()):
            if isinstance(v, _Pending):
                return v
        return filter_fn(*args, **kwargs)

    return wrapped


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
_LENIENT.filters = {name: _absorbing(f) for name, f in _LENIENT.filters.items()}


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
    running: Cell | None = None  # set on a partial weave: the cell about to run when this snapshot was taken

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


class _ThreadStdout:
    """Routes writes to a per-thread buffer while a cell runs on that thread, and to the real stream otherwise.

    ``contextlib.redirect_stdout`` swaps ``sys.stdout`` for the whole process, so with a build on a background thread (the live server), a status line printed from any other thread during a slow cell would be captured and published as that cell's output. This keeps the capture to the cell's own thread. Installed once per process, over whatever ``sys.stdout`` was.
    """

    def __init__(self, real: Any) -> None:
        self.real = real
        self.captures: dict[int, io.StringIO] = {}

    def write(self, s: str) -> int:
        return (self.captures.get(threading.get_ident()) or self.real).write(s)

    def flush(self) -> None:
        self.real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)

    @contextlib.contextmanager
    def capture(self) -> Iterator[io.StringIO]:
        tid = threading.get_ident()
        self.captures[tid] = buf = io.StringIO()
        try:
            yield buf
        finally:
            del self.captures[tid]


def _capture_stdout() -> contextlib.AbstractContextManager[io.StringIO]:
    if not isinstance(sys.stdout, _ThreadStdout):
        sys.stdout = _ThreadStdout(sys.stdout)
    return sys.stdout.capture()


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
            with _capture_stdout() as buf:
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
        if doc.show_code:
            parts.append(_code_block(cell.source))
        if out.stdout:
            parts.append(f'<pre class="stdout">{html.escape(out.stdout)}</pre>\n')
        if out.value:
            parts.append(f"\n{out.value}\n")
        if out.error:
            parts.append(_error_block(out.error))
        return parts

    def weave(self, doc: Document | None = None, *, partial: Callable[[Woven], None] | None = None) -> Woven:
        """Run what changed and weave the whole document into Markdown.

        With *partial*, it is called just before each cell that has to run, with the document as it stands: everything above woven, a running note in place of the cell, and the prose below rendered leniently (pending marks for what the cell has not yet defined), so a page can show the report while a slow cell downloads or computes. The snapshot is built synchronously, before the cell starts, so the callback may hand it to another thread.
        """
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
            for pos, seg in enumerate(doc.segments):
                if isinstance(seg, Prose):
                    parts.append(self._prose(seg, ns, lenient=stopped))
                    continue
                if stopped:
                    continue
                if index < keep:
                    out = self._outputs[index]
                else:
                    if partial is not None:
                        partial(self._snapshot(doc, pos, parts, outputs, ns, t0, cells_run))
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

    def _snapshot(
        self,
        doc: Document,
        pos: int,
        parts: list[str],
        outputs: list[CellOutput],
        ns: dict[str, Any],
        t0: float,
        cells_run: int,
    ) -> Woven:
        """The document with the cell at segment *pos* shown as running and the rest rendered leniently."""
        cell = doc.segments[pos]
        assert isinstance(cell, Cell)
        rest = [self._prose(s, ns, lenient=True) for s in doc.segments[pos + 1 :] if isinstance(s, Prose)]
        note = f'\n<pre class="running">Running the cell at line {cell.line}…</pre>\n'
        md = "\n".join([*parts, note, *rest])
        return Woven(doc, md, list(outputs), time.perf_counter() - t0, cells_run=cells_run, running=cell)

    def _prose(self, seg: Prose, ns: dict[str, Any], *, lenient: bool) -> str:
        """Prose rendered against the namespace: an f-string evaluated field by field, anything else as a Jinja template.

        *lenient* is the state past a :func:`stop`: a Jinja name that is missing renders as a pending mark (:class:`_Pending`), and an f-string field that fails to evaluate renders as one too, with its source as the label, while the fields that do evaluate still show their values.
        """
        try:
            if seg.fstring is not None:
                text = self._eval_fstring(seg.fstring, ns, lenient=lenient)
                return _dedent(re.sub(rf"(?m)^{re.escape(seg.indent)}", "", text) if seg.indent else text)
            return self._template(seg.text, _LENIENT if lenient else _STRICT).render(ns)
        except Exception as e:
            return _error_block(f"{self.path}:{seg.line}: {type(e).__name__}: {e}")

    def _eval_fstring(self, node: ast.JoinedStr, ns: dict[str, Any], *, lenient: bool) -> str:
        parts: list[str] = []
        for v in node.values:
            if not isinstance(v, ast.FormattedValue):
                parts.append(_piece_text(v))
                continue
            try:
                value = eval(compile(ast.Expression(v.value), str(self.path), "eval"), ns)
                if v.conversion != -1:
                    value = {"s": str, "r": repr, "a": ascii}[chr(v.conversion)](value)
                spec = (
                    self._eval_fstring(v.format_spec, ns, lenient=lenient)
                    if isinstance(v.format_spec, ast.JoinedStr)
                    else ""
                )
                parts.append(format(value, spec))
            except Exception:
                if not lenient:
                    raise
                parts.append(f'<mark class="pending">{html.escape(_field_source(v))}</mark>')
        return "".join(parts)


def _code_block(source: str) -> str:
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", source)), default=0) + 1)
    return f"\n{fence}python\n{source.rstrip()}\n{fence}\n"


def _error_block(text: str) -> str:
    return f'\n<pre class="error">{html.escape(text)}</pre>\n'
