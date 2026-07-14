"""CLI inspect/cancel commands over the memo store.

`ls`/`status` surface memo-orchestration experiments (state lives in the memo
store, addressed by name); `cancel` stops in-flight tasks. Commands are driven
against ``.mini`` under a tmp cwd, so DATA_ROOT (a relative path) resolves there.
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from pathlib import Path

import pytest

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick
from mini.runs import RunState


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = tick(exp, app)
        if done:
            return
        time.sleep(0.1)
    raise AssertionError("orchestration did not complete")


def test_data_root_anchors_at_project_root(tmp_path: Path, monkeypatch):
    """`.mini` follows the project root (a marker dir), not the cwd, so `mini`
    finds the same store from any subdirectory; with no marker it falls back to cwd."""
    from mini.runs import data_root

    proj = tmp_path / "proj"
    (proj / "sub" / "deep").mkdir(parents=True)
    (proj / "pyproject.toml").touch()  # the project marker
    monkeypatch.chdir(proj / "sub" / "deep")
    assert data_root() == proj.resolve() / ".mini"  # walked up past sub/deep to the marker

    bare = tmp_path / "bare"  # no marker anywhere above → fall back to cwd
    bare.mkdir()
    monkeypatch.chdir(bare)
    assert data_root() == bare.resolve() / ".mini"


def test_ls_and_status_surface_memo_experiments(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no project marker under /tmp → store resolves under cwd

    def train(x):
        return x * 2

    exp = Experiment(name="cli", main=lambda ctx: ctx.map(train, [1, 2]))
    _drive(exp, LocalApparatus("cli"))  # default data_dir → .mini/cli

    from mini.__main__ import cmd_ls, cmd_status

    cmd_ls(argparse.Namespace())
    ls_out = capsys.readouterr().out
    assert "cli" in ls_out and "tasks" in ls_out  # discovered via the memo store

    cmd_status(argparse.Namespace(name="cli", app="local"))
    status_out = capsys.readouterr().out
    assert "train-" in status_out and "2 tasks" in status_out  # per-task memo records


def test_status_and_ls_report_done_despite_superseded_failure(tmp_path: Path, monkeypatch, capsys):
    """The scenario a monitor agent hits: a task fails, the fn is *replaced* by
    one with a new name (a new identity — re-keying every cell), and the run
    completes under the new keys. ``status``/``ls`` must report the *run* as done
    — aggregating over the requested set — with the orphaned old records shown
    but marked, not poisoning the state a poller acts on."""
    monkeypatch.chdir(tmp_path)

    def bad(x):
        raise RuntimeError("bug")

    def good(x):
        return x

    def sweep(fn):
        return Experiment(name="super", main=lambda ctx: ctx.map(fn, [1]))

    app = LocalApparatus("super")  # default data_dir → .mini/super
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:  # drive the buggy version to its failure
        try:
            tick(sweep(bad), app)
        except ExceptionGroup:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("map never surfaced the failure")
    _drive(sweep(good), LocalApparatus("super"))  # the "hotfix": new source, new keys

    from mini.__main__ import cmd_ls, cmd_status

    cmd_status(argparse.Namespace(name="super", app="local"))
    status_out = capsys.readouterr().out
    assert "—  done  (1 tasks)" in status_out  # aggregate ignores the orphan
    assert "(superseded)" in status_out  # …but the orphan stays visible, marked

    cmd_ls(argparse.Namespace())
    ls_out = capsys.readouterr().out
    assert "done" in ls_out and "+1 superseded" in ls_out


def test_explain_walks_the_attempt_timeline_after_a_hotfix(tmp_path: Path, monkeypatch, capsys):
    """After a hotfix, ``explain <key>`` must answer "why did this re-run": the
    record healed *in place* (same identity), so the story is its attempt
    timeline — the failed attempt under the old code, then the current one,
    naming the dependency that moved."""
    monkeypatch.chdir(tmp_path)

    def make(fixed: bool):
        if fixed:

            def work(x):
                return x
        else:

            def work(x):
                raise RuntimeError("bug")

        return work

    def sweep(fn):
        return Experiment(name="explain", main=lambda ctx: ctx.map(fn, [1]))

    app = LocalApparatus("explain")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            tick(sweep(make(fixed=False)), app)
        except ExceptionGroup:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("map never surfaced the failure")
    _drive(sweep(make(fixed=True)), LocalApparatus("explain"))

    store = app.memo_store()
    (rec,) = store.records()  # same qualname + inputs — one identity across the fix
    assert RunState(rec["state"]) == RunState.DONE

    from mini.__main__ import cmd_explain

    cmd_explain(argparse.Namespace(name="explain", key=rec["key"], app="local"))
    out = capsys.readouterr().out
    assert "(superseded)" not in out  # healed in place — nothing orphaned
    assert "attempts (2):" in out
    assert "failed" in out and "!! RuntimeError: bug" in out  # the old attempt and its error
    assert "changed" in out  # …and the dependency that moved to heal it


def test_status_shows_queued_distinct_from_running(tmp_path: Path, monkeypatch, capsys):
    """A RUNNING record with no ``env`` is launched-but-unstarted (the worker
    writes ``env`` as its first action): ``status`` must read it as *queued*,
    not silently lump it in with tasks actually running on a worker."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "queuedexp")
    now = time.time()
    pid = os.getpid()  # a live pid, so reap_dead doesn't settle the records
    common = {"state": "running", "fn": "train", "pid": pid, "heartbeat_at": now}
    store.records_backend.merge("train-queued", {"key": "train-queued", **common})
    store.records_backend.merge("train-live", {"key": "train-live", "env": {"host": "worker.test"}, **common})

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="queuedexp", app="local"))
    out = capsys.readouterr().out
    lines = {line.split()[2]: line for line in out.splitlines() if "train-" in line}
    assert "◌" in lines["train-queued"] and "queued" in lines["train-queued"]
    assert "♥" not in lines["train-queued"]  # its heartbeat is just the launch stamp, not liveness
    assert "▸" in lines["train-live"] and "running" in lines["train-live"] and "♥" in lines["train-live"]


def test_status_badges_stale_heartbeat(tmp_path: Path, monkeypatch, capsys):
    """A RUNNING task whose heartbeat has gone quiet for minutes gets an advisory
    badge — the honest signal when a backend liveness probe has a blind spot
    (#20). Queued tasks never badge: their heartbeat is just the launch stamp."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "staleexp")
    now = time.time()
    pid = os.getpid()  # a live pid, so reap_dead doesn't settle the records
    env = {"env": {"host": "worker.test"}}
    common = {"state": "running", "fn": "train", "pid": pid}
    store.records_backend.merge("t-fresh", {"key": "t-fresh", "heartbeat_at": now, **env, **common})
    store.records_backend.merge(
        "t-stale", {"key": "t-stale", "heartbeat_at": now - STALE_HEARTBEAT_S - 60, **env, **common}
    )
    store.records_backend.merge("t-queued", {"key": "t-queued", "heartbeat_at": now - STALE_HEARTBEAT_S - 60, **common})

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="staleexp", app="local"))
    out = capsys.readouterr().out
    lines = {line.split()[2]: line for line in out.splitlines() if "t-" in line}
    assert "⚠ stale — worker may be dead" in lines["t-stale"]
    assert "stale" not in lines["t-fresh"]
    assert "stale" not in lines["t-queued"] and "⧖" in lines["t-queued"]


def test_status_json_is_machine_readable(tmp_path: Path, monkeypatch, capsys):
    """``status --json`` emits one stable JSON object — the agent-facing signal
    (#20): aggregate state, per-task state, and honest liveness hints
    (queued / heartbeat age / stale), instead of grepping the human lines."""
    import json
    from unittest.mock import ANY

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "jsonexp")
    now = time.time()
    store.records_backend.merge("t-done", {"key": "t-done", "state": "done", "fn": "train", "metrics": {"loss": 0.5}})
    store.records_backend.merge(
        "t-zombie",
        {
            "key": "t-zombie",
            "state": "running",
            "fn": "train",
            "pid": os.getpid(),  # a live pid, so reap_dead doesn't settle it
            "env": {"host": "worker.test", "gpu": "L4"},
            "heartbeat_at": now - STALE_HEARTBEAT_S - 60,
            "step": 3,
            "total": 10,
        },
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="jsonexp", app="local", json=True))
    payload = json.loads(capsys.readouterr().out)
    payload["tasks"].sort(key=lambda t: t["key"])
    assert payload == {
        "experiment": "jsonexp",
        "app": "local",
        "state": "running",
        "settled": False,
        "tasks": [
            {
                "key": "t-done",
                "fn": "train",
                "state": "done",
                "queued": False,
                "superseded": False,
                "metrics": {"loss": 0.5},
            },
            {
                "key": "t-zombie",
                "fn": "train",
                "state": "running",
                "queued": False,
                "superseded": False,
                "step": 3,
                "total": 10,
                "env": {"host": "worker.test", "gpu": "L4"},
                "heartbeat_age_s": ANY,
                "stale_heartbeat": True,
            },
        ],
    }
    assert payload["tasks"][1]["heartbeat_age_s"] > STALE_HEARTBEAT_S


def test_watch_exit_code_reflects_settle_outcome(tmp_path: Path, monkeypatch, capsys):
    """``watch`` is the wake trigger for scripts/agents (#20): it blocks until the
    run settles and its exit code carries the outcome — 0 iff DONE."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    from mini.__main__ import cmd_watch

    store = MemoStore(data_root() / "watchfail")
    store.records_backend.merge("t1", {"key": "t1", "state": "failed", "error": "boom"})
    with pytest.raises(SystemExit) as ei:
        cmd_watch(argparse.Namespace(name="watchfail", app="local", poll=0.01))
    assert ei.value.code == 1

    store = MemoStore(data_root() / "watchdone")
    store.records_backend.merge("t1", {"key": "t1", "state": "done"})
    cmd_watch(argparse.Namespace(name="watchdone", app="local", poll=0.01))  # no raise — exit 0
    assert "done" in capsys.readouterr().out


def test_app_resolution_precedence(tmp_path: Path, monkeypatch):
    """--app flag > launch marker > $MINI_APP > [tool.mini] app > local (#47)."""
    monkeypatch.chdir(tmp_path)
    from mini.__main__ import _resolve_app

    def ns(app: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(app=app)

    assert _resolve_app("exp", ns()) == "local"  # nothing configured
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\napp = "modal"\n')
    assert _resolve_app("exp", ns()) == "modal"  # project default travels with the repo
    monkeypatch.setenv("MINI_APP", "local")
    assert _resolve_app("exp", ns()) == "local"  # env overrides pyproject (one-off shell / CI)
    marker = tmp_path / ".mini" / "exp" / ".app"
    marker.parent.mkdir(parents=True)
    marker.write_text("modal\n")
    assert _resolve_app("exp", ns()) == "modal"  # the launch marker is per-experiment ground truth
    assert _resolve_app("exp", ns(app="local")) == "local"  # explicit flag beats everything


def test_run_stamps_backend_for_later_reads(tmp_path: Path, monkeypatch, capsys):
    """A launch remembers its backend (``.mini/<name>/.app``), so ``status`` with
    no ``--app`` reads the store the experiment actually lives on (#47)."""
    monkeypatch.chdir(tmp_path)
    exp_file = tmp_path / "stamp.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment
        def work(x):
            return x
        experiment = Experiment(name='stampexp', main=lambda ctx: ctx.map(work, [7]))
        """)
    )
    from mini.__main__ import cmd_run, cmd_status

    cmd_run(argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app=None, workers=1))
    assert (tmp_path / ".mini" / "stampexp" / ".app").read_text().strip() == "local"
    capsys.readouterr()

    cmd_status(argparse.Namespace(name="stampexp", app=None))  # no flag — resolved via the marker
    assert "done" in capsys.readouterr().out


def test_run_captures_lineage_and_lineage_command_reports_it(tmp_path: Path, monkeypatch, capsys):
    """A run stamps provenance into meta; ``mini lineage`` reads it back.

    Under ``/tmp`` there's no git repo, so the git block is absent — but the
    identity/driver/timeline half is always captured, which is what this asserts.
    """
    monkeypatch.chdir(tmp_path)
    exp_file = tmp_path / "prov.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment
        def work(x):
            return x
        experiment = Experiment(name='provexp', main=lambda ctx: ctx.map(work, [1, 2]))
        """)
    )
    from mini.__main__ import cmd_lineage, cmd_run

    cmd_run(argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app=None, workers=1))
    capsys.readouterr()

    cmd_lineage(argparse.Namespace(name="provexp", app="local", diff=False))
    out = capsys.readouterr().out
    assert "provexp — lineage" in out
    assert "when" in out and "driver" in out  # timeline + spawning environment always present


def test_lineage_snapshots_declared_upstreams(tmp_path: Path, monkeypatch):
    """An experiment that declares ``deps`` records each upstream's provenance, so a
    downstream run can trace which A produced its inputs."""
    monkeypatch.chdir(tmp_path)

    def work(x):
        return x

    from mini.__main__ import _stamp_lineage

    up = Experiment(name="prep", main=lambda ctx: ctx.map(work, [1]))
    _drive(up, LocalApparatus("prep"))
    up_args = argparse.Namespace(name="prep", app="local")
    _stamp_lineage(up, LocalApparatus("prep").memo_store(), up_args)

    down = Experiment(name="train", main=lambda ctx: ctx.map(work, [2]), deps=["prep"])
    _drive(down, LocalApparatus("train"))
    store = LocalApparatus("train").memo_store()
    _stamp_lineage(down, store, argparse.Namespace(name="train", app="local"))

    upstreams = store.meta()["lineage"]["upstreams"]
    assert [u["experiment"] for u in upstreams] == ["prep"]
    assert "run_at" in upstreams[0]  # carries when the upstream first ran


def test_lineage_detects_upstreams_from_resolved_refs(tmp_path: Path, monkeypatch, capsys):
    """A run that reads another experiment's ref records it as an upstream — no
    ``deps=`` declared. The producer is stamped onto the ref at ``set_ref`` time
    (in the upstream's worker), the consumer's worker records the resolution on
    its task record, and the driver rolls it up into ``lineage.upstreams``."""
    monkeypatch.chdir(tmp_path)
    # Workers must hit the tmp LocalStore: an ambient bucket would write the test ref
    # to the real shared store, and a publish-repo alone builds a CAS-less store.
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    (tmp_path / "prep.py").write_text(
        textwrap.dedent("""
        from mini import Experiment
        def produce(x):
            from mini.store import put, set_ref
            set_ref('shared/thing', put(b'payload', name='thing.bin'))
            return x
        experiment = Experiment(name='prep', main=lambda ctx: ctx.map(produce, [1]))
        """)
    )
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        from mini import Experiment
        def consume(x):
            from mini.store import get_ref
            assert get_ref('shared/thing') is not None
            return x
        experiment = Experiment(name='train', main=lambda ctx: ctx.map(consume, [2]))
        """)
    )
    from mini.__main__ import cmd_lineage, cmd_run

    ns = lambda f: argparse.Namespace(path=str(tmp_path / f), watch=True, poll=0.05, app=None, workers=1)  # noqa: E731
    cmd_run(ns("prep.py"))
    cmd_run(ns("train.py"))
    capsys.readouterr()

    store = LocalApparatus("train").memo_store()
    (rec,) = store.records()
    assert rec["upstream_refs"] == [{"ref": "shared/thing", "experiment": "prep"}]  # worker-side evidence
    upstreams = store.meta()["lineage"]["upstreams"]  # driver rollup (stamped at end of the watch)
    assert [u["experiment"] for u in upstreams] == ["prep"]
    assert upstreams[0]["refs"] == ["shared/thing"]
    assert "run_at" in upstreams[0]  # snapshotted from prep's own stored lineage

    cmd_lineage(argparse.Namespace(name="train", app="local", diff=False))
    out = capsys.readouterr().out
    assert "⇐ prep" in out and "via shared/thing" in out


def test_empty_read_names_backend_and_hints_at_the_other(tmp_path: Path, monkeypatch):
    """A read that finds nothing must say which backend it looked on and — when
    the run lives on the other one — name the flag to get there (#47)."""
    monkeypatch.chdir(tmp_path)
    import mini.__main__ as cli

    monkeypatch.setattr(cli, "_peek", lambda name, backend: 3 if backend == "modal" else 0)
    with pytest.raises(SystemExit) as e:
        cli.cmd_status(argparse.Namespace(name="ghost", app=None))
    assert "no tasks found for experiment 'ghost' on local" in str(e.value)
    assert "found 3 task(s) on modal — try: --app modal" in str(e.value)


def test_cancel_stops_running_task(tmp_path: Path):
    def slow(x):
        import time

        time.sleep(30)  # long enough that only a cancel ends it within the test
        return x

    app = LocalApparatus("cancelexp", data_dir=tmp_path / "cancelexp")
    tick(Experiment(name="cancelexp", main=lambda ctx: ctx.map(slow, [1])), app)  # launch + suspend

    store = app.memo_store()
    (rec,) = store.records()
    pid = rec["pid"]  # recorded synchronously at spawn
    assert pid and RunState(rec["state"]) == RunState.RUNNING

    assert app.cancel(store) == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())

    # the worker really took the SIGTERM (reap it to confirm + avoid a zombie)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("worker did not exit after cancel")


def test_retry_cli_heals_failed_task(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # DATA_ROOT + the experiment file resolve under here
    exp_file = tmp_path / "retrycli.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment, get_data_dir
        def flaky(x):
            f = get_data_dir() / 'attempts'
            n = int(f.read_text()) if f.exists() else 0
            f.write_text(str(n + 1))
            if n == 0:  # fail on the first attempt only
                raise RuntimeError('boom once')
            return x
        experiment = Experiment(name='retrycli', main=lambda ctx: ctx.map(flaky, [7]))
        """)
    )
    from mini.__main__ import cmd_retry, cmd_run

    def ns():  # run/retry share flags; --watch drives synchronously to settle
        return argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app="local", workers=1, key=None)

    with pytest.raises(SystemExit):  # FAILED is terminal — watch surfaces it and exits 1
        cmd_run(ns())
    capsys.readouterr()  # drop the failure output

    cmd_retry(ns())  # resets the failed task, then the rerun (attempt 2) succeeds
    out = capsys.readouterr().out
    assert "retrying 1 task" in out and "✓ complete" in out


def test_run_with_a_name_instead_of_a_file_hints_at_the_split(tmp_path: Path, monkeypatch):
    """``run``/``retry`` take a *file*; the read verbs take a *name*. Typing a
    known experiment name here must not die in a raw ``ImportError`` — it should
    name the mistake and point at the verbs that take names (#57)."""
    monkeypatch.chdir(tmp_path)  # no project marker under /tmp → store resolves under cwd
    _drive(Experiment(name="stale-probe", main=lambda ctx: ctx.map(lambda x: x, [1])), LocalApparatus("stale-probe"))

    from mini.__main__ import cmd_retry

    def ns(path: str) -> argparse.Namespace:
        return argparse.Namespace(path=path, watch=False, poll=0.05, app="local", workers=1, key=None)

    with pytest.raises(SystemExit) as e:  # the name of a real experiment, at a file-taking verb
        cmd_retry(ns("stale-probe"))
    assert "'stale-probe' is an experiment name" in str(e.value)
    assert "status/results/cancel take names" in str(e.value)

    with pytest.raises(SystemExit) as e:  # an unknown token → the plain missing-file error, no name hint
        cmd_retry(ns("nope"))
    assert "no experiment file at 'nope'" in str(e.value)
    assert "experiment name" not in str(e.value)
