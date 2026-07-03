"""Reclaim memo-store state that no current read path can reach (``mini gc``).

Collectibility is judged against the store's own invariants, not age or size:

- A **superseded record** (its key absent from the requested-keys manifest) is
  collectible once the manifest is trustworthy: the last tick ran the DAG to
  completion (``complete`` in the run meta) and nothing is still unsettled.
  A *current* record is never collectible — a DONE one is a future memo hit,
  and even a FAILED one is live state (deleting it would silently convert a
  terminal failure into a relaunch on the next wake).
- A **stale attempt file** (a ``result-<gen>.pkl``/``error-<gen>.txt`` under a
  generation the record no longer owns) is unreachable: readers resolve through
  the record's current ``gen``, and a fenced zombie writer can't make anything
  read it again. The one exception is the legacy ``error.txt``, which
  ``MemoStore.error`` still falls back to when the current attempt left no
  traceback — that stays live until the current generation writes its own.
- An **orphaned result dir** has no record at all. Records are claimed before
  the worker creates its dir, so this is debris, not a race.
- A **staged call** (``.control/memo/<key>.pkl``) is worker spawn input; it is
  dead once its task is off RUNNING (a relaunch rewrites it).

Local backend only for now. On Modal the control plane self-expires (Dict
entries lapse after 7 days idle) but Volume result dirs persist; that sweep,
and the artifact CAS (which has no reference index yet), are tracked in #15.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mini.memo import MemoStore
from mini.runs import SETTLED, RunState

__all__ = ['GcItem', 'GcPlan', 'plan_gc', 'apply_gc']

# The worker-written files an attempt owns; anything else in a result dir is
# unknown, and unknown is not garbage.
_ATTEMPT_FILE = re.compile(r'result(-\w+)?\.pkl|error(-\w+)?\.txt')


@dataclass
class GcItem:
    kind: str  # 'superseded' | 'attempt-files' | 'orphan-dir' | 'staged-call'
    key: str
    paths: list[Path]
    size: int


@dataclass
class GcPlan:
    items: list[GcItem] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # reasons collectible-looking state was left alone

    @property
    def size(self) -> int:
        return sum(i.size for i in self.items)

    def by_kind(self, kind: str) -> list[GcItem]:
        return [i for i in self.items if i.kind == kind]


def _size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        elif p.is_file():
            total += p.stat().st_size
    return total


def plan_gc(store: MemoStore, records: list[dict] | None = None) -> GcPlan:
    """What ``apply_gc`` would delete, and why the rest stays.

    Call ``reap_dead`` first so a vanished worker's RUNNING record doesn't read
    as alive. Pass *records* to reuse a snapshot already in hand.
    """
    records = store.records() if records is None else records
    plan = GcPlan()
    collected = _plan_superseded(store, records, plan)
    _plan_attempt_files(store, records, collected, plan)
    _plan_orphan_dirs(store, records, plan)
    _plan_staged_calls(store, records, collected, plan)
    return plan


def _plan_superseded(store: MemoStore, records: list[dict], plan: GcPlan) -> set[str]:
    """Whole superseded records — JSON, result dir, and staged call — gate permitting."""
    current, superseded = store.split_current(records)
    if not superseded:
        return set()
    unsettled = [r for r in current if r.get('state') not in SETTLED]
    if not store.meta().get('complete'):
        plan.kept.append(
            f'{len(superseded)} superseded record(s): the last tick did not run the DAG to '
            'completion, so the manifest may be missing keys a later stage still wants'
        )
        return set()
    if unsettled:
        plan.kept.append(f'{len(superseded)} superseded record(s): {len(unsettled)} task(s) still unsettled')
        return set()
    collected: set[str] = set()
    for rec in superseded:
        key = rec['key']
        if rec.get('state') == RunState.RUNNING:  # reap_dead left it: the worker is provably alive
            plan.kept.append(f'{key}: superseded, but its worker is still alive — cancel it first')
            continue
        paths = [p for p in (store.result_dir(key), store._call(key)) if p.exists()]
        plan.items.append(GcItem('superseded', key, paths, _size(paths)))
        collected.add(key)
    return collected


def _plan_attempt_files(store: MemoStore, records: list[dict], collected: set[str], plan: GcPlan) -> None:
    """Attempt files no read through the record's current gen can reach."""
    for rec in records:
        key = rec['key']
        d = store.result_dir(key)
        if key in collected or not d.is_dir():
            continue
        gen = rec.get('gen')
        live = {store.result_path(key, gen).name, store.error_path(key, gen).name}
        if gen and not store.error_path(key, gen).exists():
            live.add(store.error_path(key, None).name)  # error() falls back to the legacy name
        stale = [
            p for p in sorted(d.iterdir()) if p.is_file() and _ATTEMPT_FILE.fullmatch(p.name) and p.name not in live
        ]
        if stale:
            plan.items.append(GcItem('attempt-files', key, stale, _size(stale)))


def _plan_orphan_dirs(store: MemoStore, records: list[dict], plan: GcPlan) -> None:
    """Result dirs with no record at all (e.g. a control plane that expired out from under the volume)."""
    known = {r['key'] for r in records}
    memo_root = store.data_dir / '_memo'
    if not memo_root.is_dir():
        return
    for d in sorted(memo_root.iterdir()):
        if d.is_dir() and d.name not in known:
            plan.items.append(GcItem('orphan-dir', d.name, [d], _size([d])))


def _plan_staged_calls(store: MemoStore, records: list[dict], collected: set[str], plan: GcPlan) -> None:
    """Cloudpickled spawn inputs for tasks that are no longer running."""
    recs = {r['key']: r for r in records}
    if not store.root.is_dir():
        return
    for p in sorted(store.root.glob('*.pkl')):
        if p.stem in collected:
            continue
        if (recs.get(p.stem) or {}).get('state') != RunState.RUNNING:
            plan.items.append(GcItem('staged-call', p.stem, [p], _size([p])))


def apply_gc(store: MemoStore, plan: GcPlan) -> None:
    """Delete everything in *plan*.

    Record first, files second: a crash between the two leaves an orphaned dir,
    which the next gc collects — never the reverse (a record whose result dir
    is gone).
    """
    for item in plan.items:
        if item.kind == 'superseded':
            store.records_backend.delete(item.key)
        for p in item.paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
