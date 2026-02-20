"""Tests for snapshot-based regression detection."""

from __future__ import annotations

import json
from typing import Any

import pytest

from forge_world.core.metrics import ConfusionMatrix
from forge_world.core.runner import (
    AggregateMetrics,
    BenchmarkReport,
    ItemResult,
    MultiBenchmarkReport,
    SeedReport,
)
from forge_world.core.snapshots import SnapshotManager


def _make_report(
    items: list[dict[str, Any]],
    run_id: str = "test123",
) -> BenchmarkReport:
    """Build a BenchmarkReport from simple item specs."""
    item_results = []
    for item in items:
        item_results.append(
            ItemResult(
                item_id=item["id"],
                category=item.get("category", "test"),
                expected_label=item.get("expected", "findings"),
                passed=item["passed"],
                risk_level=item.get("risk", "medium"),
                confidence=item.get("confidence", 0.7),
                converging_evidence=False,
                findings=[],
                methods_flagged=item.get("methods", []),
            )
        )

    cm = ConfusionMatrix(
        true_positives=sum(1 for r in item_results if r.passed and r.expected_label == "findings"),
        true_negatives=sum(1 for r in item_results if r.passed and r.expected_label == "clean"),
        false_positives=sum(1 for r in item_results if not r.passed and r.expected_label == "clean"),
        false_negatives=sum(1 for r in item_results if not r.passed and r.expected_label == "findings"),
    )

    return BenchmarkReport(
        run_id=run_id,
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        item_results=item_results,
        confusion_matrix=cm,
        category_metrics=[],
        method_metrics={},
    )


def _make_multi_report(
    stable_items_by_seed: dict[int, list[dict[str, Any]]],
    exploration_items_by_seed: dict[int, list[dict[str, Any]]] | None = None,
) -> MultiBenchmarkReport:
    """Build a MultiBenchmarkReport from per-seed item specs."""
    seed_reports = []
    for seed, items in stable_items_by_seed.items():
        report = _make_report(items, run_id=f"multi_{seed}")
        seed_reports.append(SeedReport(seed=seed, seed_kind="stable", report=report))

    if exploration_items_by_seed:
        for seed, items in exploration_items_by_seed.items():
            report = _make_report(items, run_id=f"multi_exp_{seed}")
            seed_reports.append(SeedReport(seed=seed, seed_kind="exploration", report=report))

    # Compute aggregate
    pass_rates = [sr.report.pass_rate for sr in seed_reports]
    sensitivities = [sr.report.sensitivity for sr in seed_reports]
    fprs = [sr.report.fpr for sr in seed_reports]
    f1s = [sr.report.f1 for sr in seed_reports]

    from collections import Counter
    item_pass_counts: Counter[str] = Counter()
    item_total_counts: Counter[str] = Counter()
    for sr in seed_reports:
        for r in sr.report.item_results:
            item_total_counts[r.item_id] += 1
            if r.passed:
                item_pass_counts[r.item_id] += 1
    item_stability = {
        iid: item_pass_counts[iid] / item_total_counts[iid]
        for iid in item_total_counts
    }

    aggregate = AggregateMetrics(
        mean_pass_rate=sum(pass_rates) / len(pass_rates),
        min_pass_rate=min(pass_rates),
        max_pass_rate=max(pass_rates),
        mean_sensitivity=sum(sensitivities) / len(sensitivities),
        min_sensitivity=min(sensitivities),
        worst_case_fpr=max(fprs),
        mean_f1=sum(f1s) / len(f1s),
        item_stability=item_stability,
    )

    return MultiBenchmarkReport(
        run_id="multi_run",
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        seed_reports=seed_reports,
        aggregate_metrics=aggregate,
    )


class TestSnapshotManager:
    def test_lock_and_load(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([
            {"id": "item1", "passed": True, "risk": "high"},
            {"id": "item2", "passed": False, "risk": "low"},
        ])

        snapshot = sm.lock(report, name="v1")
        assert snapshot.name == "v1"
        assert "item1" in snapshot.item_outcomes
        assert "item2" in snapshot.item_outcomes

        loaded = sm.load("v1")
        assert loaded is not None
        assert loaded.name == "v1"
        assert loaded.item_outcomes["item1"]["passed"] is True
        assert loaded.item_outcomes["item2"]["passed"] is False

    def test_load_nonexistent(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        assert sm.load("nonexistent") is None

    def test_list_snapshots(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([{"id": "item1", "passed": True}])

        sm.lock(report, name="baseline")
        sm.lock(report, name="v2")

        names = sm.list_snapshots()
        assert "baseline" in names
        assert "v2" in names

    def test_check_no_regressions(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")

        baseline = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item2", "passed": False},
        ])
        sm.lock(baseline)

        current = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item2", "passed": False},
        ])

        regression = sm.check(current)
        assert regression.items_regressed == 0
        assert regression.items_improved == 0
        assert not regression.has_regressions

    def test_check_detects_regression(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")

        baseline = _make_report([
            {"id": "item1", "passed": True, "risk": "high"},
            {"id": "item2", "passed": True, "risk": "medium"},
        ])
        sm.lock(baseline)

        current = _make_report([
            {"id": "item1", "passed": True, "risk": "high"},
            {"id": "item2", "passed": False, "risk": "low"},
        ])

        regression = sm.check(current)
        assert regression.items_regressed == 1
        assert regression.has_regressions
        assert regression.regressions[0].item_id == "item2"

    def test_check_detects_improvement(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")

        baseline = _make_report([
            {"id": "item1", "passed": False, "risk": "low"},
        ])
        sm.lock(baseline)

        current = _make_report([
            {"id": "item1", "passed": True, "risk": "high"},
        ])

        regression = sm.check(current)
        assert regression.items_improved == 1
        assert regression.improvements[0].item_id == "item1"

    def test_check_detects_false_positive(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")

        baseline = _make_report([
            {"id": "clean1", "passed": True, "expected": "clean"},
        ])
        sm.lock(baseline)

        current = _make_report([
            {"id": "clean1", "passed": False, "expected": "clean", "risk": "high"},
        ])

        regression = sm.check(current)
        assert regression.new_false_positives == 1
        assert regression.has_new_false_positives

    def test_check_missing_baseline(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([{"id": "item1", "passed": True}])

        with pytest.raises(FileNotFoundError, match="No baseline"):
            sm.check(report)

    def test_check_new_items_ignored(self, tmp_path):
        """Items not in baseline should be ignored (not counted as regressions)."""
        sm = SnapshotManager(tmp_path / "snapshots")

        baseline = _make_report([
            {"id": "item1", "passed": True},
        ])
        sm.lock(baseline)

        current = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item_new", "passed": False},
        ])

        regression = sm.check(current)
        assert regression.items_regressed == 0

    def test_snapshot_json_is_valid(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([
            {"id": "item1", "passed": True, "risk": "high"},
        ])
        sm.lock(report, name="test")

        path = tmp_path / "snapshots" / "test.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test"
        assert "item_outcomes" in data

    def test_regression_report_to_dict(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        baseline = _make_report([{"id": "item1", "passed": True}])
        sm.lock(baseline)

        current = _make_report([{"id": "item1", "passed": False}])
        regression = sm.check(current)

        d = regression.to_dict()
        assert d["items_regressed"] == 1
        assert len(d["regressions"]) == 1


class TestGetFailedItemIds:
    def test_get_failed_single_seed(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item2", "passed": False},
            {"id": "item3", "passed": False},
        ])
        sm.lock(report)

        failed = sm.get_failed_item_ids()
        assert failed == {"item2", "item3"}

    def test_get_failed_multi_seed(self, tmp_path):
        """Returns items that failed on ANY seed."""
        sm = SnapshotManager(tmp_path / "snapshots")
        multi = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": True}, {"id": "item2", "passed": False}],
                137: [{"id": "item1", "passed": False}, {"id": "item2", "passed": True}],
            },
        )
        sm.lock(multi)

        failed = sm.get_failed_item_ids()
        # item1 failed on seed 137, item2 failed on seed 42
        assert failed == {"item1", "item2"}

    def test_no_failures_empty(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        report = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item2", "passed": True},
        ])
        sm.lock(report)

        failed = sm.get_failed_item_ids()
        assert failed == set()

    def test_missing_snapshot_raises(self, tmp_path):
        sm = SnapshotManager(tmp_path / "snapshots")
        with pytest.raises(FileNotFoundError, match="No snapshot"):
            sm.get_failed_item_ids("nonexistent")


class TestMultiSeedSnapshots:
    def test_lock_multi_seed(self, tmp_path):
        """Multi-seed lock stores per-seed outcomes and exploration metrics."""
        sm = SnapshotManager(tmp_path / "snapshots")
        multi = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": True}, {"id": "item2", "passed": False}],
                137: [{"id": "item1", "passed": True}, {"id": "item2", "passed": True}],
            },
            exploration_items_by_seed={
                999: [{"id": "item1", "passed": True}, {"id": "item2", "passed": True}],
            },
        )

        snapshot = sm.lock(multi, name="multi_baseline")
        assert snapshot.is_multi_seed
        assert "42" in snapshot.seed_outcomes
        assert "137" in snapshot.seed_outcomes
        assert snapshot.exploration_count == 1
        assert snapshot.exploration_baseline > 0

    def test_load_multi_seed(self, tmp_path):
        """Multi-seed snapshot round-trips through JSON correctly."""
        sm = SnapshotManager(tmp_path / "snapshots")
        multi = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": True}],
            },
            exploration_items_by_seed={
                999: [{"id": "item1", "passed": True}],
            },
        )
        sm.lock(multi, name="test")

        loaded = sm.load("test")
        assert loaded is not None
        assert loaded.is_multi_seed
        assert "42" in loaded.seed_outcomes
        assert loaded.exploration_count == 1

    def test_check_multi_seed_no_regression(self, tmp_path):
        """No regression when current matches baseline on all seeds."""
        sm = SnapshotManager(tmp_path / "snapshots")
        items = [{"id": "item1", "passed": True}, {"id": "item2", "passed": False}]
        baseline = _make_multi_report(
            stable_items_by_seed={42: items},
            exploration_items_by_seed={999: items},
        )
        sm.lock(baseline)

        current = _make_multi_report(
            stable_items_by_seed={42: items},
            exploration_items_by_seed={888: items},
        )
        regression = sm.check(current)
        assert not regression.has_regressions

    def test_check_multi_seed_detects_regression(self, tmp_path):
        """Regression detected when item flips from pass to fail on a seed."""
        sm = SnapshotManager(tmp_path / "snapshots")
        baseline = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": True, "risk": "high"}],
            },
        )
        sm.lock(baseline)

        current = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": False, "risk": "low"}],
            },
        )
        regression = sm.check(current)
        assert regression.items_regressed == 1
        assert regression.regressions[0].seed == 42

    def test_check_multi_seed_unstable_items(self, tmp_path):
        """Unstable items (pass on some seeds, fail on others) are reported."""
        sm = SnapshotManager(tmp_path / "snapshots")
        baseline = _make_multi_report(
            stable_items_by_seed={42: [{"id": "item1", "passed": True}]},
        )
        sm.lock(baseline)

        # item1 passes on seed 42 but fails on seed 137
        current = _make_multi_report(
            stable_items_by_seed={
                42: [{"id": "item1", "passed": True}],
                137: [{"id": "item1", "passed": False}],
            },
        )
        regression = sm.check(current)
        assert len(regression.unstable_items) == 1
        assert regression.unstable_items[0]["item_id"] == "item1"

    def test_exploration_baseline_update(self, tmp_path):
        """Exploration baseline is a running average across locks."""
        sm = SnapshotManager(tmp_path / "snapshots")

        # First lock: exploration pass rate = 0.5
        multi1 = _make_multi_report(
            stable_items_by_seed={42: [{"id": "item1", "passed": True}]},
            exploration_items_by_seed={
                999: [{"id": "item1", "passed": True}, {"id": "item2", "passed": False}],
            },
        )
        snap1 = sm.lock(multi1)
        assert snap1.exploration_baseline == snap1.exploration_metrics["mean_pass_rate"]
        assert snap1.exploration_count == 1

        # Second lock: different exploration pass rate — baseline should be averaged
        multi2 = _make_multi_report(
            stable_items_by_seed={42: [{"id": "item1", "passed": True}]},
            exploration_items_by_seed={
                888: [{"id": "item1", "passed": True}, {"id": "item2", "passed": True}],
            },
        )
        snap2 = sm.lock(multi2)
        assert snap2.exploration_count == 2
        # Baseline = avg(snap1 exploration rate, snap2 exploration rate)
        expected = (snap1.exploration_metrics["mean_pass_rate"] + snap2.exploration_metrics["mean_pass_rate"]) / 2
        assert abs(snap2.exploration_baseline - expected) < 0.001

    def test_exploration_regression_detected(self, tmp_path):
        """Exploration regression fires when current exploration drops below threshold."""
        sm = SnapshotManager(tmp_path / "snapshots")

        # Baseline: stable=100%, exploration=100%
        baseline = _make_multi_report(
            stable_items_by_seed={42: [
                {"id": "item1", "passed": True},
                {"id": "item2", "passed": True},
            ]},
            exploration_items_by_seed={999: [
                {"id": "item1", "passed": True},
                {"id": "item2", "passed": True},
            ]},
        )
        sm.lock(baseline)

        # Current: stable still good, but exploration drops to 0%
        current = _make_multi_report(
            stable_items_by_seed={42: [
                {"id": "item1", "passed": True},
                {"id": "item2", "passed": True},
            ]},
            exploration_items_by_seed={888: [
                {"id": "item1", "passed": False},
                {"id": "item2", "passed": False},
            ]},
        )
        regression = sm.check(current)
        assert regression.exploration_regressed
        assert regression.exploration_details["regressed"] is True

    def test_backward_compat_single_seed_load(self, tmp_path):
        """Single-seed snapshots load correctly with multi-seed manager."""
        sm = SnapshotManager(tmp_path / "snapshots")

        # Lock with a single-seed report
        single = _make_report([{"id": "item1", "passed": True}])
        sm.lock(single)

        loaded = sm.load()
        assert loaded is not None
        assert not loaded.is_multi_seed
        assert "item1" in loaded.item_outcomes

    def test_multi_seed_check_against_single_seed_baseline(self, tmp_path):
        """Multi-seed check works against a single-seed baseline (backward compat)."""
        sm = SnapshotManager(tmp_path / "snapshots")

        # Lock single-seed baseline
        single = _make_report([
            {"id": "item1", "passed": True},
            {"id": "item2", "passed": True},
        ])
        sm.lock(single)

        # Check with multi-seed report
        multi = _make_multi_report(
            stable_items_by_seed={
                42: [
                    {"id": "item1", "passed": True},
                    {"id": "item2", "passed": False, "risk": "low"},
                ],
            },
        )
        regression = sm.check(multi)
        assert regression.items_regressed == 1
