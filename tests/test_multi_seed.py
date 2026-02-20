"""End-to-end multi-seed benchmark tests with synthetic pipeline.

Verifies aggregate metrics, item stability, exploration seed derivation,
and analysis caching across seeds.
"""

from __future__ import annotations

from typing import Any

from forge_world.core.protocols import (
    AggregatedResult,
    Finding,
    LabeledItem,
    PassFailRule,
    Severity,
)
from forge_world.core.runner import (
    BenchmarkRunner,
    MultiBenchmarkReport,
    SeedStrategy,
)


class SeedAwarePipeline:
    """Pipeline where findings depend on the item's score."""

    def __init__(self):
        self._config = {"threshold": 0.5}

    def analyze(self, item: Any) -> list[Finding]:
        if isinstance(item, dict):
            score = item.get("score", 0.0)
            if score >= self._config["threshold"]:
                return [
                    Finding(
                        title="Anomaly",
                        method=item.get("method", "test"),
                        severity=Severity.HIGH if score > 0.8 else Severity.MEDIUM,
                        confidence=score,
                    )
                ]
        return []

    def get_config(self):
        return dict(self._config)

    def set_config(self, config):
        self._config = config

    def get_config_schema(self):
        return {}


class SimpleAggregator:
    def aggregate(self, findings: list[Finding]) -> AggregatedResult:
        if not findings:
            return AggregatedResult(
                risk_level=Severity.CLEAN,
                overall_confidence=0.0,
                converging_evidence=False,
                total_findings=0,
            )
        return AggregatedResult(
            risk_level=max(f.severity for f in findings),
            overall_confidence=sum(f.confidence for f in findings) / len(findings),
            converging_evidence=len({f.method for f in findings}) >= 2,
            total_findings=len(findings),
            methods_flagged={f.method for f in findings},
            findings=findings,
        )


class SeedAwareDataset:
    """Dataset that returns different items depending on seed.

    Fixed items (seed=None): 3 items
    With seed: fixed + extra items specific to each seed
    """

    def __init__(self):
        self._fixed = [
            LabeledItem(
                id="fixed_pass",
                category="test",
                expected_label="findings",
                data={"score": 0.9, "method": "ela"},
            ),
            LabeledItem(
                id="fixed_fail",
                category="test",
                expected_label="findings",
                data={"score": 0.3, "method": "ela"},
            ),
            LabeledItem(
                id="fixed_clean",
                category="clean",
                expected_label="clean",
                data={"score": 0.1, "method": "ela"},
            ),
        ]
        # Seed-specific items with different characteristics
        self._seed_items = {
            42: [
                LabeledItem(
                    id="s42_easy",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.85, "method": "ela"},
                ),
                LabeledItem(
                    id="s42_hard",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.4, "method": "ela"},
                ),
            ],
            137: [
                LabeledItem(
                    id="s137_easy",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.95, "method": "ela"},
                ),
                LabeledItem(
                    id="s137_borderline",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.5, "method": "ela"},
                ),
            ],
            256: [
                LabeledItem(
                    id="s256_all_pass",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.9, "method": "ela"},
                ),
                LabeledItem(
                    id="s256_all_pass2",
                    category="seeded",
                    expected_label="findings",
                    data={"score": 0.8, "method": "ela"},
                ),
            ],
        }

    def items(self, seed: int | None = None, sample_size: int | None = None) -> list[LabeledItem]:
        result = list(self._fixed)
        if seed is not None and seed in self._seed_items:
            extra = self._seed_items[seed]
            if sample_size is not None:
                extra = extra[:sample_size]
            result.extend(extra)
        return result

    def categories(self) -> list[str]:
        return ["clean", "seeded", "test"]


class SimpleRules:
    def get_rule(self, expected_label: str) -> PassFailRule:
        if expected_label == "findings":
            return PassFailRule(expected_label="findings", min_risk_for_pass=Severity.MEDIUM)
        elif expected_label == "clean":
            return PassFailRule(expected_label="clean", max_risk_for_pass=Severity.MEDIUM)
        return PassFailRule(expected_label=expected_label)


class TestMultiSeedEndToEnd:
    def _make_runner(self) -> BenchmarkRunner:
        return BenchmarkRunner(
            pipeline=SeedAwarePipeline(),
            aggregator=SimpleAggregator(),
            dataset=SeedAwareDataset(),
            rules=SimpleRules(),
        )

    def test_basic_multi_seed(self):
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42, 137], n_exploration_seeds=1)
        multi = runner.run_multi(strategy)

        assert isinstance(multi, MultiBenchmarkReport)
        assert len(multi.stable_reports) == 2
        assert len(multi.exploration_reports) == 1

    def test_aggregate_metrics_computed(self):
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42, 137, 256], n_exploration_seeds=0)
        multi = runner.run_multi(strategy)

        am = multi.aggregate_metrics
        assert am.min_pass_rate <= am.mean_pass_rate <= am.max_pass_rate
        assert am.worst_case_fpr == 0  # No false positives
        assert len(am.item_stability) > 0

    def test_item_stability_across_seeds(self):
        """Items present in all seeds should have stability values."""
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42, 137, 256], n_exploration_seeds=0)
        multi = runner.run_multi(strategy)

        stability = multi.aggregate_metrics.item_stability
        # Fixed items appear in all 3 seeds
        assert stability["fixed_pass"] == 1.0  # Always passes
        assert stability["fixed_fail"] == 0.0  # Always fails (score 0.3 < 0.5)
        assert stability["fixed_clean"] == 1.0  # Always passes (clean, no findings)

    def test_exploration_seeds_differ_per_run(self):
        """Different run_ids produce different exploration seeds."""
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42], n_exploration_seeds=2)

        multi1 = runner.run_multi(strategy)
        multi2 = runner.run_multi(strategy)

        exp1 = [sr.seed for sr in multi1.exploration_reports]
        exp2 = [sr.seed for sr in multi2.exploration_reports]
        # Different run_ids -> different exploration seeds (overwhelmingly likely)
        # There's an astronomically small chance they collide
        assert exp1 != exp2 or multi1.run_id == multi2.run_id

    def test_sample_size_limits_seeded_items(self):
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42], n_exploration_seeds=0)

        multi_full = runner.run_multi(strategy)
        multi_limited = runner.run_multi(strategy, sample_size=1)

        full_count = multi_full.stable_reports[0].report.total_count
        limited_count = multi_limited.stable_reports[0].report.total_count
        # Full: 3 fixed + 2 seed items = 5
        # Limited: 3 fixed + 1 seed item = 4
        assert full_count == 5
        assert limited_count == 4

    def test_worst_case_fpr_zero(self):
        """Across all seeds, no false positives should occur."""
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42, 137, 256], n_exploration_seeds=2)
        multi = runner.run_multi(strategy)

        assert multi.aggregate_metrics.worst_case_fpr == 0
        for sr in multi.seed_reports:
            assert sr.report.fpr == 0

    def test_seed_none_returns_only_fixed(self):
        """When no seed is given, only fixed items are returned."""
        runner = self._make_runner()
        report = runner.run()  # No seed
        assert report.total_count == 3  # Only fixed items

    def test_multi_report_summary(self):
        runner = self._make_runner()
        strategy = SeedStrategy(stable_seeds=[42, 137], n_exploration_seeds=1)
        multi = runner.run_multi(strategy)

        s = multi.summary()
        assert s["seeds_evaluated"] == 3
        assert len(s["stable_seeds"]) == 2
        assert len(s["exploration_seeds"]) == 1
        assert "mean_pass_rate" in s
        assert "worst_case_fpr" in s

    def test_multi_report_to_dict_roundtrip(self):
        runner = self._make_runner()
        multi = runner.run_multi()
        d = multi.to_dict()

        assert "seed_reports" in d
        assert "aggregate_metrics" in d
        assert d["aggregate_metrics"]["worst_case_fpr"] >= 0
        assert "item_stability" in d["aggregate_metrics"]
