"""Tests for the diff module."""

from __future__ import annotations

from typing import Any

from forge_world.core.diff import DiffReport, DiffSummary, RiskShift, compute_diff
from forge_world.core.metrics import ConfusionMatrix
from forge_world.core.runner import BenchmarkReport, ItemResult
from forge_world.core.snapshots import RegressionItem, RegressionReport, Snapshot


def _make_report(items: list[dict[str, Any]]) -> BenchmarkReport:
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
                methods_flagged=[],
            )
        )
    cm = ConfusionMatrix(
        true_positives=sum(1 for r in item_results if r.passed and r.expected_label == "findings"),
        true_negatives=sum(1 for r in item_results if r.passed and r.expected_label == "clean"),
        false_positives=sum(
            1 for r in item_results if not r.passed and r.expected_label == "clean"
        ),
        false_negatives=sum(
            1 for r in item_results if not r.passed and r.expected_label == "findings"
        ),
    )
    return BenchmarkReport(
        run_id="test",
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        item_results=item_results,
        confusion_matrix=cm,
        category_metrics=[],
        method_metrics={},
    )


def _make_snapshot(items: list[dict[str, Any]], name: str = "baseline") -> Snapshot:
    outcomes = {}
    for item in items:
        outcomes[item["id"]] = {
            "passed": item["passed"],
            "risk_level": item.get("risk", "medium"),
            "confidence": item.get("confidence", 0.7),
            "category": item.get("category", "test"),
            "expected_label": item.get("expected", "findings"),
            "methods_flagged": [],
            "converging_evidence": False,
        }
    total = len(items)
    passed = sum(1 for i in items if i["passed"])
    return Snapshot(
        name=name,
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        item_outcomes=outcomes,
        summary={"pass_rate": f"{passed}/{total}"},
    )


class TestDiff:
    def test_no_changes(self):
        """Same baseline and current = empty diff."""
        items = [
            {"id": "item1", "passed": True, "risk": "high"},
            {"id": "item2", "passed": False, "risk": "low"},
        ]
        report = _make_report(items)
        snapshot = _make_snapshot(items)
        regression = RegressionReport(
            baseline_name="baseline",
            items_regressed=0,
            items_improved=0,
            baseline_pass_rate="1/2",
            current_pass_rate="1/2",
        )

        diff = compute_diff(report, snapshot, regression)
        assert diff.summary.net_change == 0
        assert len(diff.new_failures) == 0
        assert len(diff.new_passes) == 0
        assert len(diff.risk_shifts) == 0

    def test_new_failure_detected(self):
        """Item that regressed appears in new_failures."""
        baseline_items = [
            {"id": "item1", "passed": True, "risk": "high"},
        ]
        current_items = [
            {"id": "item1", "passed": False, "risk": "low"},
        ]
        report = _make_report(current_items)
        snapshot = _make_snapshot(baseline_items)
        regression = RegressionReport(
            baseline_name="baseline",
            items_regressed=1,
            regressions=[
                RegressionItem(
                    item_id="item1",
                    category="test",
                    expected_label="findings",
                    baseline_passed=True,
                    current_passed=False,
                    baseline_risk="high",
                    current_risk="low",
                    baseline_confidence=0.7,
                    current_confidence=0.3,
                ),
            ],
            baseline_pass_rate="1/1",
            current_pass_rate="0/1",
        )

        diff = compute_diff(report, snapshot, regression)
        assert len(diff.new_failures) == 1
        assert diff.new_failures[0]["item_id"] == "item1"
        assert diff.summary.net_change == -1

    def test_new_pass_detected(self):
        """Item that improved appears in new_passes."""
        baseline_items = [
            {"id": "item1", "passed": False, "risk": "low"},
        ]
        current_items = [
            {"id": "item1", "passed": True, "risk": "high"},
        ]
        report = _make_report(current_items)
        snapshot = _make_snapshot(baseline_items)
        regression = RegressionReport(
            baseline_name="baseline",
            items_improved=1,
            improvements=[
                RegressionItem(
                    item_id="item1",
                    category="test",
                    expected_label="findings",
                    baseline_passed=False,
                    current_passed=True,
                    baseline_risk="low",
                    current_risk="high",
                    baseline_confidence=0.3,
                    current_confidence=0.9,
                ),
            ],
            baseline_pass_rate="0/1",
            current_pass_rate="1/1",
        )

        diff = compute_diff(report, snapshot, regression)
        assert len(diff.new_passes) == 1
        assert diff.summary.net_change == 1

    def test_risk_shift_up(self):
        """Item that stayed passing but risk increased."""
        baseline_items = [
            {"id": "item1", "passed": True, "risk": "medium"},
        ]
        current_items = [
            {"id": "item1", "passed": True, "risk": "high"},
        ]
        report = _make_report(current_items)
        snapshot = _make_snapshot(baseline_items)
        regression = RegressionReport(baseline_name="baseline")

        diff = compute_diff(report, snapshot, regression)
        assert len(diff.risk_shifts) == 1
        assert diff.risk_shifts[0].direction == "up"
        assert diff.risk_shifts[0].baseline_risk == "medium"
        assert diff.risk_shifts[0].current_risk == "high"

    def test_risk_shift_down(self):
        """Item that stayed passing but risk decreased."""
        baseline_items = [
            {"id": "item1", "passed": True, "risk": "high"},
        ]
        current_items = [
            {"id": "item1", "passed": True, "risk": "medium"},
        ]
        report = _make_report(current_items)
        snapshot = _make_snapshot(baseline_items)
        regression = RegressionReport(baseline_name="baseline")

        diff = compute_diff(report, snapshot, regression)
        assert len(diff.risk_shifts) == 1
        assert diff.risk_shifts[0].direction == "down"

    def test_to_markdown_format(self):
        """to_markdown produces valid markdown output."""
        summary = DiffSummary(
            baseline_pass_rate="140/168",
            current_pass_rate="145/168",
            net_change=5,
        )
        diff = DiffReport(
            summary=summary,
            new_failures=[
                {"item_id": "x", "category": "a", "baseline_risk": "h", "current_risk": "l"}
            ],
            risk_shifts=[RiskShift("y", "b", "medium", "high", "up")],
        )
        md = diff.to_markdown()
        assert "# Diff: current vs baseline" in md
        assert "## Summary" in md
        assert "+5 net" in md
        assert "## New Failures (1)" in md
        assert "## Risk Shifts (1)" in md

    def test_summary_net_change(self):
        """net_change reflects improvements minus regressions."""
        items = [
            {"id": "item1", "passed": True, "risk": "high"},
        ]
        report = _make_report(items)
        snapshot = _make_snapshot(items)
        regression = RegressionReport(
            baseline_name="baseline",
            items_regressed=2,
            items_improved=5,
        )
        diff = compute_diff(report, snapshot, regression)
        assert diff.summary.net_change == 3

    def test_boundary_approaching(self):
        """Items moving closer to the pass/fail boundary are detected."""
        # Baseline: item at medium risk (distance 0 from boundary for findings items)
        baseline_items = [
            {
                "id": "item1",
                "passed": True,
                "risk": "high",
                "expected": "findings",
                "confidence": 0.9,
                "category": "test",
            },
        ]
        # Current: item at medium risk (closer to boundary)
        current_items = [
            {
                "id": "item1",
                "passed": True,
                "risk": "medium",
                "expected": "findings",
                "confidence": 0.5,
                "category": "test",
            },
        ]
        report = _make_report(current_items)
        snapshot = _make_snapshot(baseline_items)
        regression = RegressionReport(baseline_name="baseline")

        diff = compute_diff(report, snapshot, regression)
        # The risk shift should be detected (high -> medium = "down")
        # since pass/fail didn't flip
        assert len(diff.risk_shifts) == 1
        assert diff.risk_shifts[0].direction == "down"

    def test_multi_seed_diff(self):
        """Unstable items from regression appear in diff."""
        items = [{"id": "item1", "passed": True, "risk": "high"}]
        report = _make_report(items)
        snapshot = _make_snapshot(items)
        regression = RegressionReport(
            baseline_name="baseline",
            unstable_items=[{"item_id": "item1", "stability": 0.5}],
        )
        diff = compute_diff(report, snapshot, regression)
        assert len(diff.unstable_items) == 1
