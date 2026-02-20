"""Tests for the AI agent feedback interface."""

from __future__ import annotations

from forge_world.core.agent_interface import build_evolution_context
from forge_world.core.metrics import ConfusionMatrix, CategoryMetrics, MethodEffectiveness
from forge_world.core.runner import (
    AggregateMetrics,
    BenchmarkReport,
    ItemResult,
    MultiBenchmarkReport,
    SeedReport,
)
from forge_world.core.snapshots import RegressionReport, RegressionItem


def _make_report() -> BenchmarkReport:
    """Build a test report."""
    item_results = [
        ItemResult(
            item_id="img1.png",
            category="anomalous",
            expected_label="findings",
            passed=True,
            risk_level="high",
            confidence=0.85,
            converging_evidence=True,
            findings=[
                {"method": "ela", "confidence": 0.8, "severity": "high", "title": "ELA anomaly"},
                {"method": "clone_detection", "confidence": 0.9, "severity": "high", "title": "Clone"},
            ],
            methods_flagged=["ela", "clone_detection"],
        ),
        ItemResult(
            item_id="img2.png",
            category="anomalous",
            expected_label="findings",
            passed=False,
            risk_level="low",
            confidence=0.3,
            converging_evidence=False,
            findings=[
                {"method": "ela", "confidence": 0.3, "severity": "low", "title": "Weak ELA"},
            ],
            methods_flagged=["ela"],
        ),
        ItemResult(
            item_id="clean1.pdf",
            category="clean",
            expected_label="clean",
            passed=True,
            risk_level="clean",
            confidence=0.0,
            converging_evidence=False,
            findings=[],
            methods_flagged=[],
        ),
    ]

    return BenchmarkReport(
        run_id="test_run",
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        item_results=item_results,
        confusion_matrix=ConfusionMatrix(
            true_positives=1,
            true_negatives=1,
            false_positives=0,
            false_negatives=1,
        ),
        category_metrics=[
            CategoryMetrics(
                category="anomalous",
                total=2,
                passed=1,
                failed=1,
                confusion=ConfusionMatrix(true_positives=1, false_negatives=1),
            ),
            CategoryMetrics(
                category="clean",
                total=1,
                passed=1,
                failed=0,
                confusion=ConfusionMatrix(true_negatives=1),
            ),
        ],
        method_metrics={
            "ela": MethodEffectiveness(
                method="ela",
                times_fired=2,
                true_detections=2,
                false_detections=0,
                items_with_findings=2,
                avg_confidence=0.55,
            ),
            "clone_detection": MethodEffectiveness(
                method="clone_detection",
                times_fired=1,
                true_detections=1,
                false_detections=0,
                items_with_findings=1,
                avg_confidence=0.9,
            ),
        },
        config_snapshot={"threshold": 0.5, "ela": {"quality": 80}},
    )


def _make_multi_report() -> MultiBenchmarkReport:
    """Build a test multi-seed report."""
    report1 = _make_report()
    report2 = _make_report()

    seed_reports = [
        SeedReport(seed=42, seed_kind="stable", report=report1),
        SeedReport(seed=999, seed_kind="exploration", report=report2),
    ]

    return MultiBenchmarkReport(
        run_id="multi_test",
        timestamp="2025-01-01T00:00:00Z",
        git_sha="abc1234",
        config_hash="cfg_hash",
        seed_reports=seed_reports,
        aggregate_metrics=AggregateMetrics(
            mean_pass_rate=2 / 3,
            min_pass_rate=2 / 3,
            max_pass_rate=2 / 3,
            mean_sensitivity=0.5,
            min_sensitivity=0.5,
            worst_case_fpr=0.0,
            mean_f1=0.5,
            item_stability={
                "img1.png": 1.0,
                "img2.png": 0.0,
                "clean1.pdf": 1.0,
            },
        ),
        config_snapshot={"threshold": 0.5},
    )


class TestBuildEvolutionContext:
    def test_basic(self):
        report = _make_report()
        ctx = build_evolution_context(report)

        assert ctx.current_metrics["pass_rate"] == "2/3"
        assert len(ctx.category_breakdown) == 2
        assert len(ctx.failing_items) == 1
        assert ctx.failing_items[0]["item_id"] == "img2.png"

    def test_method_effectiveness(self):
        report = _make_report()
        ctx = build_evolution_context(report)

        assert len(ctx.method_effectiveness) == 2
        methods = {m["method"] for m in ctx.method_effectiveness}
        assert "ela" in methods
        assert "clone_detection" in methods

    def test_with_regression(self):
        report = _make_report()
        regression = RegressionReport(
            baseline_name="baseline",
            items_regressed=1,
            items_improved=0,
            new_false_positives=0,
            regressions=[
                RegressionItem(
                    item_id="img2.png",
                    category="anomalous",
                    expected_label="findings",
                    baseline_passed=True,
                    current_passed=False,
                    baseline_risk="medium",
                    current_risk="low",
                    baseline_confidence=0.6,
                    current_confidence=0.3,
                )
            ],
        )
        ctx = build_evolution_context(report, regression=regression)
        assert len(ctx.items_regressed) == 1
        assert ctx.items_regressed[0]["item_id"] == "img2.png"

    def test_constraints(self):
        report = _make_report()
        ctx = build_evolution_context(
            report,
            hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
            optimization_target={"metric": "sensitivity", "direction": "max"},
        )
        assert len(ctx.hard_constraints) == 1
        assert ctx.optimization_target["metric"] == "sensitivity"

    def test_sample_size_passthrough(self):
        report = _make_report()
        ctx = build_evolution_context(report, sample_size=50)
        assert ctx.sample_size == 50


class TestMultiSeedContext:
    def test_multi_seed_context(self):
        multi = _make_multi_report()
        ctx = build_evolution_context(multi)

        assert "seeds_evaluated" in ctx.current_metrics
        assert ctx.current_metrics["seeds_evaluated"] == 2
        assert ctx.seed_variance  # Should have seed variance info
        assert "per_seed" in ctx.seed_variance
        assert len(ctx.seed_variance["per_seed"]) == 2

    def test_item_stability_reported(self):
        multi = _make_multi_report()
        ctx = build_evolution_context(multi)

        # img2.png has stability 0.0 (always fails) -- it's unstable since 0 < 0.0 is false
        # So only items with 0 < stab < 1 appear
        # All items are either 0.0 or 1.0 in our test, so no unstable items
        assert len(ctx.item_stability) == 0

    def test_multi_seed_prompt_context(self):
        multi = _make_multi_report()
        ctx = build_evolution_context(multi)
        prompt = ctx.to_prompt_context()

        assert "# Evolution Context" in prompt
        assert "## Seed Variance" in prompt
        assert "## Current Performance" in prompt

    def test_multi_seed_to_dict(self):
        multi = _make_multi_report()
        ctx = build_evolution_context(multi)
        d = ctx.to_dict()

        assert "seed_variance" in d
        assert "current_metrics" in d
        assert "failing_items" in d


class TestEvolutionContextPrompt:
    def test_to_prompt_context(self):
        report = _make_report()
        ctx = build_evolution_context(
            report,
            hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
            optimization_target={"metric": "sensitivity", "direction": "max"},
        )
        prompt = ctx.to_prompt_context()

        assert "# Evolution Context" in prompt
        assert "## Current Performance" in prompt
        assert "Sensitivity" in prompt
        assert "## Failing Items" in prompt
        assert "img2.png" in prompt
        assert "## Method Effectiveness" in prompt
        assert "ela" in prompt
        assert "## Hard Constraints" in prompt
        assert "fpr" in prompt

    def test_to_prompt_context_no_failures(self):
        """Should not crash when there are no failures."""
        report = BenchmarkReport(
            run_id="ok",
            timestamp="2025-01-01T00:00:00Z",
            git_sha=None,
            config_hash="x",
            item_results=[],
            confusion_matrix=ConfusionMatrix(),
            category_metrics=[],
            method_metrics={},
        )
        ctx = build_evolution_context(report)
        prompt = ctx.to_prompt_context()
        assert "# Evolution Context" in prompt
        # Should not contain failure section
        assert "Failing Items" not in prompt

    def test_sample_size_in_prompt(self):
        report = _make_report()
        ctx = build_evolution_context(report, sample_size=50)
        prompt = ctx.to_prompt_context()
        assert "Sample size (M): 50" in prompt


class TestDiagnosticsInContext:
    def test_diagnostics_in_context(self):
        report = _make_report()
        diagnostics = [
            {
                "label": "Undetectable retracted papers",
                "description": "No image evidence",
                "item_ids": ["paper1.pdf"],
                "suggested_action": "Skip these",
                "achievable": False,
            },
        ]
        ctx = build_evolution_context(report, diagnostics=diagnostics)
        assert len(ctx.diagnostic_clusters) == 1
        assert ctx.diagnostic_clusters[0]["label"] == "Undetectable retracted papers"

    def test_diagnostics_in_prompt(self):
        report = _make_report()
        diagnostics = [
            {
                "label": "Test cluster",
                "description": "Test desc",
                "item_ids": ["a", "b"],
                "suggested_action": "Fix it",
                "achievable": True,
            },
        ]
        ctx = build_evolution_context(report, diagnostics=diagnostics)
        prompt = ctx.to_prompt_context()
        assert "## Domain-Specific Diagnosis" in prompt
        assert "Test cluster" in prompt
        assert "ACHIEVABLE" in prompt

    def test_no_diagnostics_omits_section(self):
        report = _make_report()
        ctx = build_evolution_context(report)
        prompt = ctx.to_prompt_context()
        assert "Domain-Specific Diagnosis" not in prompt

    def test_diagnostics_in_dict(self):
        report = _make_report()
        diagnostics = [
            {
                "label": "Test",
                "description": "desc",
                "item_ids": [],
                "suggested_action": "action",
                "achievable": True,
            },
        ]
        ctx = build_evolution_context(report, diagnostics=diagnostics)
        d = ctx.to_dict()
        assert "diagnostic_clusters" in d
        assert len(d["diagnostic_clusters"]) == 1


class TestEvolutionContextDict:
    def test_to_dict(self):
        report = _make_report()
        ctx = build_evolution_context(report)
        d = ctx.to_dict()

        assert "current_metrics" in d
        assert "failing_items" in d
        assert "method_effectiveness" in d
        assert "current_config" in d
        assert d["current_config"]["threshold"] == 0.5
