"""Execution-environment capture and the Modal lineage/cost helpers.

``compute_env`` runs inside the worker to record what a task *actually* ran on; the
Modal helpers name worker functions for the dashboard and aggregate billing. All
pure enough to test without a live backend — the Modal env allowlist and cost
aggregation are the security- and correctness-sensitive bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from mini import runs
from mini.modal_apparatus import _aggregate_cost, _worker_fn_name


def test_compute_env_has_core_fields():
    env = runs.compute_env()
    assert {"host", "platform", "python", "cpu_count"} <= env.keys()


def test_compute_env_records_modal_container_ids_but_never_secrets(monkeypatch):
    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
    monkeypatch.setenv("MODAL_REGION", "us-west-2")
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_AWS")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-abc")
    # Credentials Modal also sets in every container — these must NOT be captured.
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN", "eyJ-secret-jwt")
    monkeypatch.setenv("MODAL_TASK_SECRET", "shhh")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "also-secret")
    env = runs.compute_env()
    assert env["modal_task_id"] == "ta-123"
    assert env["region"] == "us-west-2"
    assert env["cloud"] == "CLOUD_PROVIDER_AWS"
    assert env["modal_image_id"] == "im-abc"
    blob = str(env)
    assert "eyJ-secret-jwt" not in blob
    assert "shhh" not in blob
    assert "also-secret" not in blob


def test_worker_fn_name_is_readable_and_disambiguated():
    def train(x):
        return x

    name = _worker_fn_name(train)
    assert name.startswith("train-")
    assert _worker_fn_name(train) == name  # stable across calls


def test_worker_fn_name_distinguishes_same_named_functions():
    def make(tag):
        def run(x):  # both have __name__ == "run" but distinct qualnames
            return x

        run.__qualname__ = f"make.<locals>.run.{tag}"
        return run

    a, b = make("a"), make("b")
    assert a.__name__ == b.__name__ == "run"
    assert _worker_fn_name(a) != _worker_fn_name(b)  # hash suffix keeps them apart


def test_worker_fn_name_sanitizes_unsafe_characters():
    def fn():
        pass

    fn.__name__ = "weird name/with:chars"
    assert _worker_fn_name(fn).startswith("weird-name-with-chars-")


@dataclass
class _Item:
    object_id: str
    cost: Decimal
    cost_by_resource: dict[str, Decimal]


def test_aggregate_cost_sums_only_wanted_apps_with_breakdown():
    items = [
        _Item("ap-1", Decimal("1.00"), {"L4": Decimal("0.90"), "CPU": Decimal("0.10")}),
        _Item("ap-1", Decimal("0.50"), {"CPU": Decimal("0.50")}),  # a second daily interval
        _Item("ap-9", Decimal("99.0"), {"CPU": Decimal("99.0")}),  # not ours — ignored
    ]
    out = _aggregate_cost(items, {"ap-1"})
    assert out["total"] == Decimal("1.50")
    assert out["by_resource"] == {"L4": Decimal("0.90"), "CPU": Decimal("0.60")}
    assert out["intervals"] == 2


def test_aggregate_cost_empty_when_no_match():
    out = _aggregate_cost([_Item("ap-x", Decimal("1"), {})], {"ap-y"})
    assert out == {"total": Decimal(0), "by_resource": {}, "intervals": 0}
