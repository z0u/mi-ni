"""
Identity-keyed memoization for multi-step orchestration.

A task record answers two different questions, and the store keeps them apart:

- **Identity — which task is this?** The key: the fn's qualified name plus a fingerprint of its inputs. Stable across code edits, so a record (and its logs, results, history) keeps one address for the task's whole life.
- **Validity — is the cached result current?** The *evidence* stored on each attempt: a fingerprint of the fn's source plus the source of the project functions/classes it references (transitively), whatever it imports *inside its own body*, and the explicit ``version=``. Stale evidence re-runs the task **in place** — a new attempt under the same key, with the prior attempt compacted into the record's ``history``.

Both fingerprints must be **deterministic across processes** (every agent wake is a fresh process) — hashing ``cloudpickle.dumps(fn)`` fails that (its bytes vary run to run), so we fingerprint *source*, which also ignores library churn (site-packages and the mini framework itself are excluded).
"""

from __future__ import annotations

import ast
import dataclasses
import dis
import enum
import fcntl
import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import secrets
import sys
import textwrap
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any, Callable, Iterator, cast

import cloudpickle

from mini.runs import SETTLED, RunState, _atomic_write, _merge_json

__all__ = ["task_key", "task_key_parts", "RecordStore", "LocalRecordStore", "MemoStore", "PollCache", "META_KEY"]

log = logging.getLogger(__name__)

# Source under these roots is treated as an opaque, stable dependency: the
# stdlib, installed packages, and the mini framework itself (so editing mini
# doesn't invalidate every experiment's cache).
_MINI_DIR = str(Path(__file__).parent.resolve())

# Reserved control-plane key for run-level metadata (the wall-clock budget /
# deadline). It rides the same record store as the task records — a sidecar, so a
# detached run carries its budget with no new infra — but is excluded from
# ``records()`` so it never reads as a task or skews the aggregate state. A task
# fingerprint is ``{name}-{hex12}``, so ``__run__`` can never collide with one.
META_KEY = "__run__"


def _is_project_file(path: str | Path) -> bool:
    rf = str(Path(path).resolve())
    return "site-packages" not in rf and "/lib/python3" not in rf and not rf.startswith(_MINI_DIR)


def _is_project_source(obj: Any) -> bool:
    try:
        f = inspect.getsourcefile(obj)
    except TypeError, OSError:
        return False
    return bool(f) and _is_project_file(cast(str, f))


def _nested_codes(code: types.CodeType) -> Iterator[types.CodeType]:
    """*code* plus every code object nested in it (inner defs, lambdas, genexprs).

    A helper referenced only inside a nested function lives in the *inner* code object's ``co_names``; walking just the outer one would miss it.
    """
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _nested_codes(const)


def _attr_chain_refs(fn: Callable) -> list[Any]:
    """Objects reached through attribute chains rooted at a global (``utils.helper``).

    Bare names are resolved via ``co_names`` ∩ globals, but a helper called as a module attribute never appears in globals under its own name — so without this walk, ``import utils; utils.helper()`` would be *invisible* to the fingerprint and editing the helper would silently serve stale results. We scan the bytecode for ``LOAD_GLOBAL`` → ``LOAD_ATTR``… chains and resolve each step with ``getattr``, collecting any function/class the chain lands on (``pkg.mod.fn`` resolves through the intermediate modules).
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", {})
    if code is None:
        return []
    refs: list[Any] = []
    for c in _nested_codes(code):
        chain: Any = None
        for ins in dis.get_instructions(c):
            if ins.opname == "LOAD_GLOBAL" and ins.argval in g:
                chain = g[ins.argval]
            elif ins.opname == "LOAD_ATTR" and chain is not None:
                chain = getattr(chain, ins.argval, None)
                if callable(chain) or isinstance(chain, type):
                    refs.append(chain)
            else:
                chain = None  # any other instruction breaks the chain
    return refs


def _collect_class(cls: type, seen: dict[str, str]) -> None:
    """Collect a class's source, then traverse its methods' *references*.

    The class source already contains the method bodies textually (so editing a method invalidates); traversing the methods is what picks up the helpers and project bases they *call*, which the text alone doesn't reach.
    """
    if cls.__qualname__ in seen:
        return
    try:
        seen[cls.__qualname__] = _without_docstrings(inspect.getsource(cls))
    except TypeError, OSError:
        return
    for base in cls.__bases__:
        if _is_project_source(base):
            _collect_class(base, seen)
    for member in vars(cls).values():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if isinstance(member, types.FunctionType):
            _collect_sources(member, seen)
    # Usually redundant — a method would carry the same chain — but a class with no
    # methods of its own (a dataclass of fields, an enum) reaches this and nothing else.
    _collect_deferred(_import_time_chain(cls), seen)


def _value_json(obj: Any) -> str | None:
    """A stable JSON encoding of a plain value, or ``None`` if it has none.

    No ``default=`` fallback here: an exotic object's ``repr`` can embed a memory address, which would make the fingerprint differ every process — worse than not tracking the value at all. Stable-or-skip.
    """
    try:
        return json.dumps(_canonical(obj), sort_keys=True)
    except TypeError, ValueError:
        return None


def _named_refs(fn: Callable) -> list[tuple[str | None, Any]]:
    """Everything *fn* references, as ``(name, object)`` pairs.

    Bare globals (from every nested code object), closure cells (named via ``co_freevars``), and attribute-chain targets (unnamed — they're never treated as values, only as code).
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", {})
    names = [n for c in _nested_codes(code) for n in c.co_names] if code is not None else []
    refs: list[tuple[str | None, Any]] = [(n, g[n]) for n in names if n in g]
    freevars = code.co_freevars if code is not None else ()
    for name, cell in zip(freevars, getattr(fn, "__closure__", None) or (), strict=False):
        try:
            refs.append((name, cell.cell_contents))
        except ValueError:
            pass
    return refs + [(None, obj) for obj in _attr_chain_refs(fn)]


# ---------------------------------------------------------------------------
# Deferred (function-local) imports
#
# A task that keeps the driver light by importing inside its body —
# ``def eval_one(...): from sca.compute.geometry import probe_maps`` — reaches
# project code the reference walk above cannot see: the module never lands in
# the fn's globals, so editing it would serve a silent stale hit.
#
# The reference walk above resolves live *objects*; down here there are none,
# because importing is the very cost a deferred import exists to avoid. So we
# read source instead — find the module's file by searching ``sys.path``, parse
# it, take the top-level definitions the import actually named, then follow what
# *those* reference, through the same module and on into its own imports. The
# result matches the reference walk's granularity (a helper, not its module)
# while importing nothing and running no project code.
#
# Source doesn't always say what a name binds: a star-import, a name defined
# inside an ``if``, a module alias passed around as a value. Each of those falls
# back to folding in the whole module. Over-invalidation is the safe direction —
# a spurious re-run is visible and bounded, a stale hit isn't.
# ---------------------------------------------------------------------------

# Ceiling on how far the transitive walk spreads. A project's dependency graph
# is small; the cap just stops a pathological one from dominating a fingerprint.
_MAX_DEFERRED_SYMBOLS = 500

# What the walk resolves: a top-level name in a module, or — where narrowing
# isn't sound — the whole module, as ``(module, None)``.
type _Ref = tuple[str, str | None]

# The module's own import-time statements, as a pseudo-symbol: reserved because
# no Python identifier contains a dot.
_PRELUDE = "<module>"


@functools.cache
def _module_file(name: str) -> Path | None:
    """The source file for dotted module *name*, found by searching ``sys.path``.

    Deliberately not ``importlib.util.find_spec``: that imports parent packages, and a deferred import exists precisely because importing here is expensive. A plain path search reads nothing and executes nothing. Cached — ``sys.path`` doesn't meaningfully move within a process.
    """
    rel = Path(*name.split("."))
    for entry in sys.path:
        if not entry:
            continue
        for cand in (Path(entry) / rel.with_suffix(".py"), Path(entry) / rel / "__init__.py"):
            if cand.is_file():
                return cand
    return None


@functools.cache
def _installed_roots() -> frozenset[str]:
    """Top-level names supplied by an installed distribution.

    The reason a name can legitimately resolve to no *source* file: a C extension (``ujson``), or a wheel that ships no ``.py`` at its root. Read from installed metadata rather than by importing, and only ever consulted once a path search has already failed.
    """
    try:
        return frozenset(importlib.metadata.packages_distributions())
    except Exception:  # a diagnostic must never be the thing that breaks a run
        log.debug("couldn't enumerate installed distributions", exc_info=True)
        return frozenset()


def _is_namespace_portion(name: str) -> bool:
    """Whether *name* is a directory on ``sys.path``.

    Only asked once ``_module_file`` has already come back empty, and that pairing is what makes it an answer: a directory holding no ``__init__.py`` is a PEP 420 namespace package. Such a package has no source of its own, so finding none is its ordinary shape rather than a hole — its submodules still resolve and still join the evidence. ``_installed_roots`` covers the namespace packages that arrived in a wheel; a project-local one has no metadata to consult, and the directory is the only thing that says so.

    Named on the portion itself rather than its root, so a missing submodule of a namespace package (``nspkg.gone``) still reads as the hole it is. Uncached: reached once per module name, behind a path search that already failed.
    """
    rel = Path(*name.split("."))
    return any((Path(entry) / rel).is_dir() for entry in sys.path if entry)


def _should_have_resolved(name: str) -> bool:
    """Whether finding no source for *name* is a hole rather than an exclusion.

    ``_module_file`` returning ``None`` is the normal case for the stdlib and for extension modules, and the walk is right to skip those. It means something else when the name is project code: nothing about the module joins the evidence, so edits to it can't invalidate the cache. Told apart by the *root* package, which is what says whose code this is — a missing submodule of a project package counts, a missing submodule of ``numpy`` does not.
    """
    if _is_namespace_portion(name):
        return False
    root = name.partition(".")[0]
    if (path := _module_file(root)) is not None:
        return _is_project_file(path)
    return root not in sys.stdlib_module_names and root not in _installed_roots()


def _resolve_relative(pkg: str, module: str | None, level: int) -> str | None:
    """Absolute dotted name for ``from <level dots><module> import …`` inside *pkg*."""
    if level == 0:
        return module
    parts = pkg.split(".") if pkg else []
    if level - 1 > len(parts):
        return None  # climbs past the top — nothing to resolve against
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, *([module] if module else [])]) or None


@dataclasses.dataclass(frozen=True)
class _ModuleIndex:
    """A project module's top-level namespace, read from its source file.

    Enough to answer "what does this imported name bind, and what does *it* reach" without importing: the definitions by name, the module's own import bindings, the statements that are neither (they run on import, and can bind names no source read can see), and which names are ever used as something other than the root of an attribute access.
    """

    name: str
    pkg: str
    source: str
    tree: ast.Module
    defs: dict[str, ast.stmt]
    imports: dict[str, _Ref]
    prelude: tuple[ast.stmt, ...]
    bare: frozenset[str]
    opaque: bool  # a star-import: the namespace can't be enumerated, so narrowing is unsound


def _bare_names(tree: ast.AST) -> frozenset[str]:
    """Names loaded somewhere other than as the root of an attribute access.

    A module alias narrows to the attributes actually reached (``mv.lift``) only while *every* use is an attribute access. Once the bare name goes somewhere else — passed to a function, stored in a dict — what it reaches is anyone's guess, and the whole module has to count.
    """
    rooted = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in rooted)


def _segment(source: str, node: ast.stmt) -> str:
    """A definition's source text, decorators included.

    Whole lines from the first decorator (``ast``'s ``lineno`` for a decorated def points at the ``def`` itself, so a decorator — which changes what the name binds — would otherwise sit outside the text and its edits go unseen).
    """
    start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]) - 1
    return "".join(source.splitlines(keepends=True)[start : node.end_lineno or node.lineno])


def _stmt_defs(node: ast.stmt) -> dict[str, ast.stmt]:
    """The top-level names one statement defines, each mapped to the statement."""
    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
            return {node.name: node}
        case ast.TypeAlias(name=ast.Name(id=alias)):
            return {alias: node}
        case ast.Assign(targets=targets):
            return {t.id: node for t in targets if isinstance(t, ast.Name)}
        case ast.AnnAssign(target=ast.Name(id=target)):
            return {target: node}
        case _:
            return {}


def _stmt_imports(node: ast.stmt, pkg: str) -> dict[str, _Ref]:
    """The names one import statement binds, and what each of them reaches."""
    match node:
        case ast.Import(names=aliases):
            # ``import a.b.c`` binds ``a`` but runs all three; ``… as z`` binds ``z``
            # to the leaf. Either way the leaf is what the name reaches through.
            return {(a.asname or a.name.split(".")[0]): (a.name, None) for a in aliases}
        case ast.ImportFrom(module=mod, names=aliases, level=level) if base := _resolve_relative(pkg, mod, level):
            return {(a.asname or a.name): (base, a.name) for a in aliases if a.name != "*"}
        case _:
            return {}


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


_DOCSTRING_HOLDERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _without_docstrings(source: str) -> str:
    """*source* with every docstring blanked out, so it carries no weight in the evidence.

    A docstring is the one piece of source with nothing behind it: rewording one changes what a reader learns and not what the task computes, so it should not re-stamp the fingerprint and re-run the DAG. Comments stay in. A changed comment usually rides along with changed code, so the over-invalidating bias is the right one there, and this stays a narrow carve-out rather than a general "ignore the prose" rule.

    Blanked line by line rather than cut, so line numbers still line up with the file if anyone reads a manifest back, and a docstring sharing its line with code (``def f(): "doc"``) is left alone rather than taking the code with it. Text that won't parse comes back as it came.

    For *source* only. A value's JSON encoding (:func:`_value_json`) must not come through here: a value that encodes to a bare string parses as a module whose only statement is a docstring, and would blank to nothing.
    """
    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError, ValueError:
        return source
    lines = text.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_HOLDERS) or not node.body or not _is_docstring(doc := node.body[0]):
            continue
        end = doc.end_lineno or doc.lineno
        before, after = lines[doc.lineno - 1][: doc.col_offset], lines[end - 1][doc.end_col_offset :]
        if before.strip() or after.strip():
            continue  # shares its lines with code
        for i in range(doc.lineno - 1, end):
            lines[i] = "\n" if lines[i].endswith("\n") else ""
    return "".join(lines)


@functools.cache
def _module_index(name: str) -> _ModuleIndex | None:
    """Read *name*'s top-level namespace from source.

    ``None`` for anything that isn't project code — the stdlib, an installed package, ``mini`` itself, or a name that resolves to no file at all.

    That last case is the one worth hearing about, so it warns: a module the driver process can't see contributes nothing to the evidence, which reads as "no dependencies" and caches forever. Cached alongside the index, so each name says it once.
    """
    if (path := _module_file(name)) is None:
        if _should_have_resolved(name):
            log.warning(
                "no source found for %r on sys.path, and it is neither stdlib nor an installed package — nothing about it joins the evidence, so edits to it will not re-run the task. If it is project code, check that the driver process can see it (an editable install, or PYTHONPATH). If it is meant to be absent — an optional dependency behind try/except ImportError, or an `if TYPE_CHECKING:` import — there is nothing to fix.",
                name,
            )
        return None
    if not _is_project_file(path):
        return None
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except OSError, SyntaxError:
        return None
    pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
    defs: dict[str, ast.stmt] = {}
    imports: dict[str, _Ref] = {}
    prelude: list[ast.stmt] = []
    opaque = False
    for node in tree.body:
        opaque |= isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
        defs |= (bound := _stmt_defs(node))
        imports |= (imported := _stmt_imports(node, pkg))
        # Whatever binds no name is import-time behavior: it runs, and it can bind
        # names no source read can see. A docstring is text with nothing behind it.
        if not (bound or imported or isinstance(node, (ast.Import, ast.ImportFrom)) or _is_docstring(node)):
            prelude.append(node)
    return _ModuleIndex(name, pkg, source, tree, defs, imports, tuple(prelude), _bare_names(tree), opaque)


def _package_chain(module: str) -> list[_Ref]:
    """The packages executed on the way down to *module*.

    Their ``__init__`` runs before it does, and that can matter well beyond re-exports — ``sca/__init__.py`` sets ``XLA_FLAGS``, which changes what the task computes.
    """
    parts = module.split(".")[:-1]
    return [(".".join(parts[: i + 1]), None) for i in range(len(parts))]


def _import_time_chain(obj: Any) -> list[_Ref]:
    """The same import-time evidence for a live object that a deferred import gets from its name.

    The reference walk reaches a helper as an *object*, so the modules that ran to produce it leave no trace: a task importing ``probe_maps`` at module scope records the function and what it references, and no package source enters. ``sca/__init__.py`` could then change its ``XLA_FLAGS`` line — which changes what the task computes — without re-running anything. The deferred walk gets the chain for free, because it resolves a dotted name; here it's read off ``__module__`` instead, and folds the same two things: the parent packages whole, and the defining module's own import-time statements.

    Skipped when the module resolves to no file, which for a live object is ordinary rather than a hole: ``__main__``, a notebook cell module, something built by ``exec``. The deferred walk's missing-source warning is about a name that was *written down* and should have resolved; there's no such name here, so there's nothing to report.
    """
    module = getattr(obj, "__module__", None)
    # ``__module__`` is a str by convention, and assignable to anything at all — which
    # would reach the path search as a crash. A fingerprint must not be what breaks a run.
    if not isinstance(module, str) or not module or _module_file(module) is None:
        return []
    return [*_package_chain(module), (module, _PRELUDE)]


def _imports_within(node: ast.AST, idx: _ModuleIndex) -> list[_Ref]:
    """Every project reference the ``import`` statements inside *node* name.

    At any depth, so a module's own function-local imports are followed the same way the task's are.
    """
    out: list[_Ref] = []
    for n in ast.walk(node):
        match n:
            case ast.Import(names=aliases):
                out += [(a.name, None) for a in aliases]
            case ast.ImportFrom(module=mod, names=aliases, level=level):
                if base := _resolve_relative(idx.pkg, mod, level):
                    out += [(base, a.name) if a.name != "*" else (base, None) for a in aliases]
    return out


def _through_attr(idx: _ModuleIndex, root: str, attr: str) -> list[_Ref]:
    """The narrowest reference an ``root.attr`` access implies.

    Through a *module* alias it's the single attribute reached; through anything else — a class, a function, a name defined right here — it's that whole object, whose own source already contains the attribute.
    """
    if (ref := idx.imports.get(root)) is None:
        return [(idx.name, root)] if root in idx.defs else []
    module, symbol = ref
    target = f"{module}.{symbol}" if symbol else module
    return [(target, attr)] if _module_file(target) else [ref]


def _node_refs(node: ast.stmt, idx: _ModuleIndex) -> list[_Ref]:
    """What one definition reaches: names from its own module, attributes through a module alias, and whatever it imports inside its own body."""
    out = _imports_within(node, idx)
    for n in ast.walk(node):
        match n:
            # A name that is *only* ever an attribute root narrows to the attribute;
            # one used bare anywhere in the module can't, and falls to the case below.
            case ast.Attribute(value=ast.Name(id=root), attr=attr) if root not in idx.bare:
                out += _through_attr(idx, root, attr)
            case ast.Name(id=name) if name in idx.bare:
                out += [idx.imports[name]] if name in idx.imports else [(idx.name, name)] * (name in idx.defs)
    return out


def _whole_module(idx: _ModuleIndex, seen: dict[str, str]) -> list[_Ref]:
    """Fall back to a module's entire source — still following it narrowly."""
    seen[f"module:{idx.name}"] = _without_docstrings(idx.source)
    return _imports_within(idx.tree, idx)


def _resolve_ref(ref: _Ref, seen: dict[str, str]) -> list[_Ref]:
    """Record one reference's source in *seen*, and return what it reaches in turn."""
    module, symbol = ref
    idx = _module_index(module)
    if idx is None:
        return []  # stdlib, an installed package, mini itself, or off the path entirely
    chain = _package_chain(module)
    if symbol is None or idx.opaque:
        return chain + _whole_module(idx, seen)
    if symbol == _PRELUDE:
        if idx.prelude:  # most modules are all defs and imports — no entry rather than an empty one
            seen[f"{module}:{_PRELUDE}"] = "\n".join(_without_docstrings(_segment(idx.source, n)) for n in idx.prelude)
        return chain + [r for n in idx.prelude for r in _node_refs(n, idx)]
    if (target := idx.imports.get(symbol)) is not None:
        return [target, (module, _PRELUDE)]  # a re-export: follow it to where it's defined
    if (node := idx.defs.get(symbol)) is None:
        # Not a top-level name here: either a submodule (``from sca.data import
        # mixed_vocab``) or something no source read can see — bound inside an
        # ``if``, or by code that runs on import. The submodule is a reference of
        # its own; anything else means the whole module counts.
        return chain + (
            [(f"{module}.{symbol}", None)] if _module_file(f"{module}.{symbol}") else _whole_module(idx, seen)
        )
    seen[f"{module}:{symbol}"] = _without_docstrings(_segment(idx.source, node))
    return chain + [(module, _PRELUDE), *_node_refs(node, idx)]


def _collect_deferred(refs: list[_Ref], seen: dict[str, str]) -> None:
    """Fold the source behind each reference into *seen*, transitively.

    A work list rather than recursion: the graph reaches a project's whole import closure in the worst case, and the ceiling wants one place to live.
    """
    queue, done = list(refs), set[_Ref]()
    while queue:
        if (ref := queue.pop()) in done:
            continue
        done.add(ref)
        if len(done) > _MAX_DEFERRED_SYMBOLS:
            log.warning("deferred-import walk hit its %d-symbol ceiling at %s", _MAX_DEFERRED_SYMBOLS, ref)
            return
        queue += _resolve_ref(ref, seen)


# Loads whose operand is a local/global name; the fused pairs push two, the
# second of which is what a following ``LOAD_ATTR`` applies to.
_LOADS_FAST = ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
_LOADS = (*_LOADS_FAST, "LOAD_NAME", "LOAD_DEREF", "LOAD_GLOBAL")
_STORES = ("STORE_FAST", "STORE_NAME", "STORE_DEREF", "STORE_GLOBAL")


def _scan_code(
    code: types.CodeType, pkg: str, bound: dict[str, _Ref], attrs: dict[str, set[str]], bare: set[str]
) -> None:
    """Fold one code object's imports and name uses into the shared maps.

    ``IMPORT_NAME`` carries the module and the two values pushed before it are the relative level and the fromlist; the ``IMPORT_FROM``s that follow name what came out of it, and the ``STORE`` after each names the local it lands in. Tracking that local is what lets ``from sca.data import mixed_vocab as mv`` narrow to the ``mv.lift`` the body actually reaches, instead of taking the module whole.

    The maps are shared across a function's nested code objects because the two halves usually *are* in different ones: the import binds in the outer body while the attribute is reached from a closure or comprehension inside it. A name reused for something unrelated in a nested scope only ever reads as "used bare", which costs the whole module — the safe direction.
    """
    consts: list[Any] = []  # the last two values pushed (level, fromlist)
    module: str | None = None
    fromlist: tuple = ()
    pending: str | None = None  # the name a following STORE binds; "" is the module itself
    top: str | None = None  # the name most recently loaded, awaiting its next instruction

    for ins in dis.get_instructions(code):
        if top is not None and ins.opname == "LOAD_ATTR":
            attrs.setdefault(top, set()).add(ins.argval)
            top, consts = None, []
            continue
        if top is not None:  # loaded and then used as something other than an attribute root
            bare.add(top)
            top = None
        match ins.opname:
            case "LOAD_CONST" | "LOAD_SMALL_INT":
                consts = [*consts, ins.argval][-2:]
                continue
            case "IMPORT_NAME":
                level = next((v for v in consts if isinstance(v, int)), 0)
                fromlist = next((v for v in consts if isinstance(v, tuple)), ())
                module = _resolve_relative(pkg, ins.argval, level)
                pending = None if fromlist else ""
            case "IMPORT_FROM":
                pending = ins.argval
            case op if op in _STORES and module and pending is not None:
                bound[ins.argval] = (module, pending or None)
                pending = None
            case op if op in _LOADS:
                # A ``LOAD_FAST`` of a *cell* variable is closure plumbing — the cell
                # object on its way into an inner function, not a read of its value.
                # Counting it as a use would mark every closed-over import "used
                # bare" and cost the whole module; the inner code object is scanned
                # in its own right, where the real read shows up as ``LOAD_DEREF``.
                cells = code.co_cellvars if op in _LOADS_FAST else ()
                raw = cast("list[str]", list(ins.argval) if isinstance(ins.argval, tuple) else [ins.argval])
                names = [n for n in raw if n not in cells]
                # Whatever is left on the stack is what a following LOAD_ATTR applies
                # to — so if *that* one was plumbing, nothing here awaits an attribute.
                top = names[-1] if names and raw[-1] not in cells else None
                bare.update(names[:-1] if top else names)
        consts = []
    if top is not None:
        bare.add(top)


def _bytecode_imports(fn: Callable) -> list[_Ref]:
    """What *fn*'s own body imports, narrowed by what it does with each binding."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return []
    pkg = getattr(sys.modules.get(getattr(fn, "__module__", "") or ""), "__package__", "") or ""
    bound: dict[str, _Ref] = {}
    attrs: dict[str, set[str]] = {}
    bare: set[str] = set()
    for c in _nested_codes(code):
        _scan_code(c, pkg, bound, attrs, bare)

    out: list[_Ref] = []
    for local, (mod, name) in bound.items():
        target = f"{mod}.{name}" if name else mod
        if name is None:
            out.append((mod, None))  # a plain ``import``: the statement's own effect is the point
        elif local in bare or local not in attrs:
            out.append((mod, name))  # used bare, or never loaded — nothing to narrow to
        else:
            out += [(target, a) for a in attrs[local]] if _module_file(target) else [(mod, name)]
    return out


def _collect_sources(fn: Callable, seen: dict[str, str]) -> None:
    qualname = getattr(fn, "__qualname__", repr(fn))
    if qualname in seen:
        return
    try:
        seen[qualname] = _without_docstrings(inspect.getsource(fn))
    except TypeError, OSError:
        return
    for name, obj in _named_refs(fn):
        if isinstance(obj, types.MethodType):
            obj = obj.__func__
        if isinstance(obj, types.FunctionType) and _is_project_source(obj):
            _collect_sources(obj, seen)
        elif isinstance(obj, type) and _is_project_source(obj):
            _collect_class(obj, seen)
        elif name is not None and not isinstance(obj, types.ModuleType) and not callable(obj):
            # A plain value referenced by name (a module-level LR, a config table):
            # fold its canonical JSON in, so editing the *value* invalidates like
            # editing code. Skipped when it has no stable encoding (see _value_json).
            if (js := _value_json(obj)) is not None:
                seen[f"{qualname}::{name}"] = js
    _collect_deferred(_bytecode_imports(fn) + _import_time_chain(fn), seen)


@functools.lru_cache(maxsize=256)
def _sources_for(fn: Callable) -> tuple[tuple[str, str], ...]:
    """The (cached) sorted dependency manifest for *fn*: ``(name, source-or-value)``.

    Source never changes within one process (every wake is a fresh process), so this caches per fn object — a ``ctx.map`` fingerprints its fn once per wake instead of re-walking the reference graph for every cell.
    """
    seen: dict[str, str] = {}
    _collect_sources(fn, seen)
    return tuple(sorted(seen.items()))


def _code_fingerprint(fn: Callable) -> str:
    blob = "\n--\n".join(f"{k}:{v}" for k, v in _manifest(fn))
    return hashlib.sha256(blob.encode()).hexdigest()


def _canonical(o: Any) -> Any:
    """Normalize *o* into a JSON-stable structure — deterministic across processes.

    ``pickle.dumps`` is *not* stable run-to-run for values containing sets, and a Pydantic model carries one (``__pydantic_fields_set__``); set iteration order is hash-randomized per process, so the same config would fingerprint differently each wake and miss the cache (the same trap that ruled out cloudpickle for the *code* fingerprint). So we canonicalize first: models and dataclasses to their field dicts, sets to sorted lists, then JSON with sorted keys downstream.
    """
    dump = getattr(o, "model_dump", None)  # pydantic v2, duck-typed (no hard dep on pydantic)
    if callable(dump):
        try:
            return _canonical(dump(mode="json"))
        except TypeError:
            return _canonical(dump())
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _canonical(dataclasses.asdict(o))
    if isinstance(o, enum.Enum):
        return [type(o).__qualname__, _canonical(o.value)]
    if isinstance(o, Mapping):
        return {str(k): _canonical(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted((_canonical(x) for x in o), key=lambda v: json.dumps(v, sort_keys=True, default=repr))
    if isinstance(o, (list, tuple)):
        return [_canonical(x) for x in o]
    if isinstance(o, PurePath):
        return str(o)  # a path keys by *location* — prefer an Artifact, which keys by content
    if (code := _canonical_code(o)) is not None:
        return code
    return o


def _canonical_code(o: Any) -> list | None:
    """Canonical forms for *code passed as data* — keyed by source, not identity.

    A callable's repr embeds a memory address, so without these a callable input would produce a fresh key every process — relaunching the task on every wake. Returns ``None`` for non-code values (handled by :func:`_canonical`).
    """
    if isinstance(o, types.MethodType):
        return ["method", o.__func__.__qualname__, _code_fingerprint(o.__func__)[:12], _canonical(o.__self__)]
    if isinstance(o, types.FunctionType):
        return ["fn", o.__qualname__, _code_fingerprint(o)[:12]]
    if isinstance(o, type):
        return ["class", o.__qualname__, hashlib.sha256(_class_source(o).encode()).hexdigest()[:12]]
    if isinstance(o, functools.partial):
        return ["partial", _canonical(o.func), _canonical(o.args), _canonical(o.keywords)]
    return None


def _class_source(cls: type) -> str:
    try:
        return inspect.getsource(cls)
    except TypeError, OSError:
        return cls.__qualname__  # no source (builtin/C) — the name is the stable id


def _input_fingerprint(args: tuple) -> str:
    try:
        blob = json.dumps(_canonical(args), sort_keys=True, default=repr).encode()
    except Exception:
        blob = repr(args).encode()
    if b" at 0x" in blob:  # an object address leaked into the key
        log.warning(
            "task inputs have no stable encoding (repr contains an object address), so the memo "
            "key will differ every process and the task can never be a cache hit — it will relaunch "
            "on every wake. Pass plain data, dataclasses, or Artifact handles instead: %.200r",
            args,
        )
    return hashlib.sha256(blob).hexdigest()


# Guards fn-value cycles: a module-level container holding the fn that references
# it would recurse (collect fn → canonicalize the container → fingerprint fn → …).
# The marker is a constant per qualname, so the resulting manifest stays
# deterministic across processes.
_collecting: set[int] = set()


def _manifest(fn: Callable) -> tuple[tuple[str, str], ...]:
    if id(fn) in _collecting:
        return ((f"<recursive:{getattr(fn, '__qualname__', '?')}>", ""),)
    _collecting.add(id(fn))
    try:
        try:
            return _sources_for(fn)
        except TypeError:  # unhashable callable — compute uncached
            seen: dict[str, str] = {}
            _collect_sources(fn, seen)
            return tuple(sorted(seen.items()))
    finally:
        _collecting.discard(id(fn))


def task_key(fn: Callable, args: tuple) -> str:
    """The stable *identity* key for calling *fn* with *args*.

    Identity is which task this is — the fn's qualified name plus its inputs — deliberately excluding code: an edited fn re-runs under the **same** key (a new attempt on the same record) instead of orphaning it. Whether the cached result is still *valid* is judged against the evidence from :func:`task_key_parts`, stored per attempt.
    """
    return task_key_parts(fn, args)[0]


def task_key_parts(fn: Callable, args: tuple, version: str | None = None) -> tuple[str, dict[str, Any]]:
    """The identity key plus the validity evidence to stamp on its next attempt.

    Returns ``(key, parts)``. The key hashes only the fn's module-qualified name and the input fingerprint. *parts* carries what decides staleness — the code fingerprint, ``version=``, and ``deps``: a short hash per tracked dependency (the fn itself, each project helper/class it references, each plain-value global) — so ``mini explain`` can diff two attempts down to *which* dependency moved.
    """
    manifest = _manifest(fn)
    blob = "\n--\n".join(f"{k}:{v}" for k, v in manifest)
    code_fp = hashlib.sha256(blob.encode()).hexdigest()
    input_fp = _input_fingerprint(args)
    ident = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', 'task')}"
    h = hashlib.sha256(f"{ident}\n{input_fp}".encode())
    key = f"{getattr(fn, '__name__', 'task')}-{h.hexdigest()[:12]}"
    deps = {k: hashlib.sha256(v.encode()).hexdigest()[:8] for k, v in manifest}
    parts = {"code_fp": code_fp[:12], "input_fp": input_fp[:12], "deps": deps}
    if version:
        parts["version"] = version
    return key, parts


# What a finished attempt is worth keeping once a new one replaces it: the
# evidence and the outcome. Live/bulky fields (metrics, env, heartbeats, pids)
# describe a worker, not the attempt's identity in the run's story.
_ATTEMPT_KEEP = ("state", "gen", "code_fp", "input_fp", "version", "deps", "created_at", "error", "exc_type")


def _compact_attempt(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in _ATTEMPT_KEEP if k in rec}


class RecordStore(ABC):
    """A small, flat ``key -> record`` store: the memo's control plane.

    Records are tiny and hot (state, step, latest metrics, heartbeat), last-writer-wins. The local backend is JSON files; the Modal backend is a named ``modal.Dict`` (readable from the client with no remote function). The interface is deliberately minimal so a ``modal.Dict`` satisfies it directly.
    """

    @abstractmethod
    def read(self, key: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def write(self, key: str, record: dict[str, Any]) -> None:
        """Overwrite a record wholesale (resets stale fields, e.g. a prior error)."""

    @abstractmethod
    def merge(self, key: str, fields: dict[str, Any]) -> None:
        """Merge *fields* into the record (progress/heartbeat updates)."""

    @abstractmethod
    def keys(self) -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a record entirely (the GC verb) — a no-op if *key* is absent."""

    # Conditional writes, fenced on the record's attempt generation (``gen``).
    # These defaults are read-check-write — atomic only if the backend makes them
    # so (``LocalRecordStore`` overrides under a file lock; ``modal.Dict`` has no
    # compare-and-swap, so on Modal only the fresh-key claim is exact — via
    # insert-if-absent, see ``ModalRecordStore.write_if`` — and the rest is
    # best-effort with a tiny window — still a vast improvement over
    # unconditional last-writer-wins).

    def write_if(self, key: str, record: dict[str, Any], gen: str | None) -> bool:
        """Replace the record iff its current ``gen`` equals *gen* (``None`` = unclaimed)."""
        if (self.read(key) or {}).get("gen") != gen:
            return False
        self.write(key, record)
        return True

    def merge_if(self, key: str, fields: dict[str, Any], gen: str | None) -> bool:
        """Merge *fields* iff the record's current ``gen`` equals *gen*."""
        if (self.read(key) or {}).get("gen") != gen:
            return False
        self.merge(key, fields)
        return True


class LocalRecordStore(RecordStore):
    """``RecordStore`` backed by JSON files under a directory.

    All mutations serialize on one store-wide ``flock``: ``merge`` is read-modify-write, so without the lock two concurrent mergers (a worker's final DONE vs the reaper's FAILED, a heartbeat vs the tick's pid stamp) could each read the same base record and silently drop the other's fields. Reads stay lock-free — ``_atomic_write`` renames, so a reader never sees a half-written file. The lock also makes ``write_if``/``merge_if`` genuinely atomic (check and write under one critical section).
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield  # released when the file closes

    def read(self, key: str) -> dict[str, Any] | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, key: str, record: dict[str, Any]) -> None:
        with self._locked():
            _atomic_write(self.root / f"{key}.json", json.dumps(record))

    def merge(self, key: str, fields: dict[str, Any]) -> None:
        with self._locked():
            _merge_json(self.root / f"{key}.json", fields)

    def write_if(self, key: str, record: dict[str, Any], gen: str | None) -> bool:
        with self._locked():
            if (self.read(key) or {}).get("gen") != gen:
                return False
            _atomic_write(self.root / f"{key}.json", json.dumps(record))
            return True

    def merge_if(self, key: str, fields: dict[str, Any], gen: str | None) -> bool:
        with self._locked():
            if (self.read(key) or {}).get("gen") != gen:
                return False
            _merge_json(self.root / f"{key}.json", fields)
            return True

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.exists() else []

    def delete(self, key: str) -> None:
        with self._locked():
            (self.root / f"{key}.json").unlink(missing_ok=True)


class MemoStore:
    """Per-experiment content-addressed task store (the orchestration backend).

    Two planes: records (small: state, metrics, heartbeat) live on a ``RecordStore`` control plane; results and tracebacks (large) live on the I/O plane. Locally both are files under ``data_dir``; on Modal the records go to a ``modal.Dict`` and results to the Volume, so the same ``MemoStore`` serves the client (poll/gather) and the remote worker (write-back) without either touching the other's filesystem.

    The cloudpickled *call* is not part of either plane: locally it's staged to disk for the subprocess worker; on Modal it's passed straight to ``spawn``.
    """

    def __init__(self, data_dir: Path, records: RecordStore | None = None):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / ".control" / "memo"
        self.records_backend: RecordStore = records or LocalRecordStore(self.root)

    def _call(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def result_dir(self, key: str) -> Path:
        return self.data_dir / "_memo" / key

    def state(self, key: str) -> RunState | None:
        rec = self.records_backend.read(key)
        return RunState(rec["state"]) if rec and rec.get("state") else None

    def record(self, key: str) -> dict[str, Any]:
        return self.records_backend.read(key) or {"key": key, "state": None}

    def result_path(self, key: str, gen: str | None) -> Path:
        """Where attempt *gen* of *key* writes its result.

        Generation-qualified so a superseded worker that survives ``cancel`` physically *cannot* overwrite its successor's result — each attempt owns its own file, and readers resolve through the record's current ``gen``. (``None`` — a record from before generations — reads the legacy name.)
        """
        return self.result_dir(key) / (f"result-{gen}.pkl" if gen else "result.pkl")

    def error_path(self, key: str, gen: str | None) -> Path:
        return self.result_dir(key) / (f"error-{gen}.txt" if gen else "error.txt")

    def artifacts_path(self, key: str, gen: str | None) -> Path:
        """Where attempt *gen* records the blob shas its result references.

        The worker stamps this sidecar next to the result (see :func:`mini._taskworker.execute_task`), so the artifact GC can mark a result's references without unpickling it — no project imports, no arbitrary code, one small read per record however large the result.
        """
        return self.result_dir(key) / (f"result-{gen}.artifacts.json" if gen else "result.artifacts.json")

    def result_artifacts(self, key: str) -> list[str] | None:
        """Blob shas the current result references, or ``None`` for a record from before the sidecar existed (unpickle the result to find out)."""
        p = self.artifacts_path(key, self._gen(key))
        return json.loads(p.read_text()) if p.exists() else None

    def _gen(self, key: str) -> str | None:
        return (self.records_backend.read(key) or {}).get("gen")

    def result(self, key: str) -> Any:
        return cloudpickle.loads(self.result_path(key, self._gen(key)).read_bytes())

    def error(self, key: str) -> str:
        for p in (self.error_path(key, self._gen(key)), self.error_path(key, None)):
            if p.exists():
                return p.read_text()
        return "(no logs)"

    def update(self, key: str, **fields: Any) -> None:
        self.records_backend.merge(key, fields)

    def update_if(self, key: str, gen: str, **fields: Any) -> bool:
        """Merge *fields* only while attempt *gen* still owns the record.

        The worker-side fence: every write a worker makes passes through here, so once its record is claimed by a successor attempt (or released by ``cancel``/``reap_dead``), a lingering worker can no longer heartbeat, merge DONE over the new attempt's RUNNING, or resurrect cleared fields.
        """
        return self.records_backend.merge_if(key, fields, gen)

    def records(self) -> list[dict[str, Any]]:
        return [
            rec for key in self.records_backend.keys() if key != META_KEY and (rec := self.records_backend.read(key))
        ]

    def meta(self) -> dict[str, Any]:
        """Run-level metadata (the wall-clock budget / ``deadline_at``), or ``{}``.

        Stored under the reserved ``META_KEY`` so it shares the run's control plane (local JSON / Modal ``Dict``) without ever surfacing as a task.
        """
        return self.records_backend.read(META_KEY) or {}

    def set_meta(self, **fields: Any) -> None:
        """Merge run-level metadata (e.g. ``deadline_at``) into the reserved record."""
        self.records_backend.merge(META_KEY, fields)

    def requested_keys(self) -> list[str] | None:
        """The keys the DAG requested on its last tick, or ``None`` if never recorded.

        Records are content-keyed, so an edited fn or a removed config leaves its old record behind under a key no wake will request again. This manifest is what lets a read-only view (``status``/``ls``/``watch``) aggregate over the run's *current* records and mark the rest superseded — without re-running ``main`` (reads must never tick). ``None`` (a store written before the manifest, or a run never ticked) means "unknown": treat every record as current.
        """
        return self.meta().get("requested")

    def split_current(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split *records* into ``(current, superseded)`` against the requested set.

        With no manifest, everything is current (nothing to judge against).
        """
        requested = self.requested_keys()
        if requested is None:
            return records, []
        wanted = set(requested)
        current = [r for r in records if r["key"] in wanted]
        return current, [r for r in records if r["key"] not in wanted]

    def dag_complete(self) -> bool | None:
        """Did the last tick run ``main`` to the end, or suspend part-way?

        ``None`` when no tick has recorded it. The distinction a read-only view can't derive for itself: with every launched task DONE, a run whose DAG suspended still has stages to go and needs another ``run`` — while one whose ``main`` returned is finished. Both read "all tasks done".
        """
        return self.meta().get("complete")

    def deadline(self) -> float | None:
        """The run's wall-clock deadline (epoch seconds), or ``None`` if unbudgeted."""
        return self.meta().get("deadline_at")

    def budget_expired(self) -> bool:
        """Whether a budget is set *and* its deadline has passed.

        The gate both for tearing a run down (cancel in-flight tasks) and for refusing to launch new work past the deadline.
        """
        d = self.deadline()
        return d is not None and time.time() >= d

    def _with_history(self, key: str, rec: dict[str, Any]) -> dict[str, Any]:
        """Fold the record's prior attempt (if any ran) into *rec*'s ``history``.

        Keys are identity, so a re-run replaces the record in place; compacting the outgoing attempt first is what keeps the task's story — every attempt, its evidence, its outcome — on the one record (``mini explain``).
        """
        prior = self.records_backend.read(key) or {}
        history: list[dict[str, Any]] = list(prior.get("history") or ())
        if prior.get("state"):  # a reset placeholder (state None) is not an attempt
            history.append(_compact_attempt(prior))
        if history:
            rec["history"] = history
        return rec

    def reset(self, key: str) -> None:
        """Clear a record back to un-run (state → None) so the next tick reruns it.

        The retry primitive: a settled-but-not-DONE task is terminal, so re-running takes intent. The cleared attempt is kept in the record's history; stale result/error artifacts are overwritten on the rerun.
        """
        self.records_backend.write(key, self._with_history(key, {"key": key, "state": None}))

    def mark_running(
        self, fn: Callable, key: str, parts: dict[str, Any] | None = None, expect_gen: str | None = None
    ) -> str | None:
        """Claim the record for a fresh attempt: flip it to RUNNING (wholesale, clearing any prior error) under a new generation stamp.

        Called by ``Ctx`` before the apparatus spawns the worker, so a poll between stage and first heartbeat sees RUNNING rather than a stale state. *parts* is the validity evidence (``code_fp``/``version``/``deps`` from :func:`task_key_parts`) this attempt runs under; any prior attempt is compacted into ``history``, so ``mini explain`` can answer "why did this re-run" after the fact.

        The claim is conditional on *expect_gen* — the ``gen`` the caller read when it classified the record (``None`` = unclaimed). If another ticker claimed the key in between, nothing is written and ``None`` returns (don't spawn — theirs is running). On success, returns the new attempt's ``gen``: the fence every write from that worker must carry.
        """
        gen = secrets.token_hex(4)
        rec = self._with_history(
            key,
            {
                "key": key,
                "fn": getattr(fn, "__name__", "task"),
                "state": RunState.RUNNING,
                "gen": gen,
                "created_at": time.time(),
                **(parts or {}),
            },
        )
        return gen if self.records_backend.write_if(key, rec, expect_gen) else None

    def write_call(
        self,
        key: str,
        fn: Callable,
        args: tuple,
        hooks: list[Callable] | None = None,
        gen: str | None = None,
        watchdog_s: float | None = None,
        watchdog_grace_s: float | None = None,
    ) -> None:
        """Stage the cloudpickled call to disk for a local subprocess worker."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._call(key).write_bytes(cloudpickle.dumps((fn, args, hooks or [], gen, watchdog_s, watchdog_grace_s)))

    def read_call(self, key: str) -> tuple[Callable, tuple, list[Callable], str | None, float | None, float | None]:
        parts = cloudpickle.loads(self._call(key).read_bytes())
        # A staged call from an older client may be a shorter tuple; staging is
        # transient (spawn → worker start), but a worker must still run it.
        return cast(
            "tuple[Callable, tuple, list[Callable], str | None, float | None, float | None]",
            (*parts, *([None] * (6 - len(parts)))),
        )


class PollCache:
    """Cheap repeated polling of a ``MemoStore``'s records for large sweeps.

    A settled record (``DONE``/``FAILED``/``CANCELLED``) is immutable, so once seen it never needs re-reading. Each ``records`` call re-reads only the unsettled subset (plus any keys not seen yet); the settled tail is served from memory. On Modal every record read is a ``modal.Dict`` round-trip, so a long sweep that's mostly done stops paying for the part that can't change — the watch loops poll just the handful still in flight.

    A reaper may settle a stale ``RUNNING`` record out from under us. That key was unsettled (so not cached), and the reaper writes it through ``MemoStore``, so the next ``records`` re-reads it once and caches the now-terminal record — nothing stale lingers.

    A *tick* can relaunch a settled record in place (keys are identity; an edit makes a new attempt, it doesn't re-key), so a cache must not outlive a tick — ``drive_and_watch`` rebuilds its cache per stage. Between ticks, settled is settled.
    """

    def __init__(self) -> None:
        self._settled: dict[str, dict[str, Any]] = {}

    def records(self, store: MemoStore) -> list[dict[str, Any]]:
        backend = store.records_backend
        out: list[dict[str, Any]] = []
        for key in backend.keys():
            if key == META_KEY:  # run-level metadata, not a task
                continue
            if cached := self._settled.get(key):
                out.append(cached)
                continue
            if (rec := backend.read(key)) is None:
                continue
            if rec.get("state") in SETTLED:  # StrEnum members hash as their str value
                self._settled[key] = rec
            out.append(rec)
        return out
