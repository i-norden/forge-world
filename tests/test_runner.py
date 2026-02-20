"""Tests for BenchmarkRunner with a synthetic pipeline."""

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
    ItemResult,
    MultiBenchmarkReport,
    PerformanceMetrics,
    SeedStrategy,
    _compute_performance_metrics,
    _derive_exploration_seeds,
)


# --- Synthetic test implementations ---


class FakePipeline:
    """Pipeline that returns configurable findings based on item data."""

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {"threshold": 0.5}

    def analyze(self, item: Any) -> list[Finding]:
        # Item data is a dict with 'score' and 'method' fields
        if isinstance(item, dict):
            score = item.get("score", 0.0)
            method = item.get("method", "test_method")
            if score >= self._config["threshold"]:
                severity = Severity.HIGH if score > 0.8 else Severity.MEDIUM
                return [
                    Finding(
                        title=f"Anomaly detected (score={score})",
                        method=method,
                        severity=severity,
                        confidence=score,
                        item_id=item.get("id", ""),
                    )
                ]
        return []

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def set_config(self, config: dict[str, Any]) -> None:
        self._config = config

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }


class FakeAggregator:
    def aggregate(self, findings: list[Finding]) -> AggregatedResult:
        if not findings:
            return AggregatedResult(
                risk_level=Severity.CLEAN,
                overall_confidence=0.0,
                converging_evidence=False,
                total_findings=0,
            )
        max_severity = max(f.severity for f in findings)
        avg_conf = sum(f.confidence for f in findings) / len(findings)
        methods = {f.method for f in findings}
        return AggregatedResult(
            risk_level=max_severity,
            overall_confidence=avg_conf,
            converging_evidence=len(methods) >= 2,
            total_findings=len(findings),
            methods_flagged=methods,
            findings=findings,
        )


class FakeDataset:
    """Dataset that returns fixed items plus optional seed-sampled items."""

    def __init__(
        self, items: list[LabeledItem], seed_items: dict[int, list[LabeledItem]] | None = None
    ):
        self._items = items
        self._seed_items = seed_items or {}

    def items(self, seed: int | None = None, sample_size: int | None = None) -> list[LabeledItem]:
        result = list(self._items)
        if seed is not None and seed in self._seed_items:
            extra = self._seed_items[seed]
            if sample_size is not None:
                extra = extra[:sample_size]
            result.extend(extra)
        return result

    def categories(self) -> list[str]:
        return sorted({i.category for i in self._items})


class FakeRules:
    def get_rule(self, expected_label: str) -> PassFailRule:
        if expected_label == "findings":
            return PassFailRule(expected_label="findings", min_risk_for_pass=Severity.MEDIUM)
        elif expected_label == "clean":
            return PassFailRule(expected_label="clean", max_risk_for_pass=Severity.MEDIUM)
        return PassFailRule(expected_label=expected_label)


# --- Tests ---


def _make_items() -> list[LabeledItem]:
    """Create a set of labeled test items."""
    return [
        # Anomalous items (should detect)
        LabeledItem(
            id="anomaly_high",
            category="anomalous",
            expected_label="findings",
            data={"id": "anomaly_high", "score": 0.9, "method": "ela"},
        ),
        LabeledItem(
            id="anomaly_medium",
            category="anomalous",
            expected_label="findings",
            data={"id": "anomaly_medium", "score": 0.6, "method": "ela"},
        ),
        LabeledItem(
            id="anomaly_missed",
            category="anomalous",
            expected_label="findings",
            data={"id": "anomaly_missed", "score": 0.3, "method": "ela"},
        ),
        # Clean items (should not flag)
        LabeledItem(
            id="clean_1",
            category="clean",
            expected_label="clean",
            data={"id": "clean_1", "score": 0.1, "method": "ela"},
        ),
        LabeledItem(
            id="clean_2",
            category="clean",
            expected_label="clean",
            data={"id": "clean_2", "score": 0.2, "method": "ela"},
        ),
    ]


def _make_seed_items() -> dict[int, list[LabeledItem]]:
    """Create seed-specific extra items."""
    return {
        42: [
            LabeledItem(
                id="seed42_item1",
                category="seeded",
                expected_label="findings",
                data={"id": "seed42_item1", "score": 0.7, "method": "ela"},
            ),
            LabeledItem(
                id="seed42_item2",
                category="seeded",
                expected_label="findings",
                data={"id": "seed42_item2", "score": 0.4, "method": "ela"},
            ),
        ],
        137: [
            LabeledItem(
                id="seed137_item1",
                category="seeded",
                expected_label="findings",
                data={"id": "seed137_item1", "score": 0.8, "method": "ela"},
            ),
        ],
    }


class TestBenchmarkRunner:
    def test_basic_run(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()

        assert report.total_count == 5
        assert report.run_id
        assert report.timestamp
        assert report.config_hash

        # anomaly_high (0.9) -> HIGH -> pass
        # anomaly_medium (0.6) -> MEDIUM -> pass
        # anomaly_missed (0.3 < 0.5 threshold) -> CLEAN -> fail
        # clean_1 (0.1) -> CLEAN -> pass
        # clean_2 (0.2) -> CLEAN -> pass
        assert report.pass_count == 4
        assert report.confusion_matrix.false_negatives == 1
        assert report.confusion_matrix.false_positives == 0

    def test_run_with_seed(self):
        """Run with a seed should include seed-sampled items."""
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        report = runner.run(seed=42)
        # 5 fixed + 2 seed-42 items = 7 total
        assert report.total_count == 7

    def test_run_with_sample_size(self):
        """sample_size limits seed-sampled items."""
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        report = runner.run(seed=42, sample_size=1)
        # 5 fixed + 1 seed-42 item (sample_size=1) = 6 total
        assert report.total_count == 6

    def test_run_without_seed(self):
        """Run without seed returns only fixed items."""
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        report = runner.run()
        assert report.total_count == 5  # Only fixed items

    def test_category_metrics(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        categories = {c.category: c for c in report.category_metrics}

        assert "anomalous" in categories
        assert "clean" in categories
        assert categories["anomalous"].total == 3
        assert categories["anomalous"].passed == 2
        assert categories["clean"].total == 2
        assert categories["clean"].passed == 2

    def test_method_metrics(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()

        # Only items with score >= 0.5 produce findings, so:
        # anomaly_high (0.9) and anomaly_medium (0.6) fire 'ela'
        assert "ela" in report.method_metrics
        assert report.method_metrics["ela"].times_fired == 2

    def test_config_sensitivity(self):
        """Lowering the threshold should detect more items."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(config={"threshold": 0.2}),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        assert report.pass_count >= 4

    def test_on_item_complete_callback(self):
        completed: list[ItemResult] = []

        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
            on_item_complete=completed.append,
        )
        runner.run()
        assert len(completed) == 5

    def test_report_to_dict(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        d = report.to_dict()

        assert "run_id" in d
        assert "item_results" in d
        assert len(d["item_results"]) == 5
        assert "confusion_matrix" in d
        assert "category_metrics" in d
        assert "method_metrics" in d
        assert "summary" in d

    def test_summary(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        s = report.summary()
        assert s["pass_rate"] == "4/5"
        assert "sensitivity" in s
        assert "fpr" in s

    def test_zero_fpr(self):
        """With default threshold, no clean items should be falsely flagged."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        assert report.fpr == 0.0
        assert report.confusion_matrix.false_positives == 0

    def test_analysis_cache(self):
        """When cache is provided, repeated item_ids reuse cached findings."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        cache: dict = {}
        report1 = runner.run(_analysis_cache=cache)
        # Cache should have entries for all 5 items
        assert len(cache) == 5

        # Running again with same cache should reuse findings
        report2 = runner.run(_analysis_cache=cache)
        assert report1.pass_count == report2.pass_count


class TestRunMulti:
    def test_basic_run_multi(self):
        """run_multi with default SeedStrategy produces a MultiBenchmarkReport."""
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        multi = runner.run_multi()

        assert isinstance(multi, MultiBenchmarkReport)
        # Default: 1 stable seed (42) + 1 exploration seed
        assert len(multi.seed_reports) == 2
        assert len(multi.stable_reports) == 1
        assert len(multi.exploration_reports) == 1
        assert multi.stable_reports[0].seed == 42
        assert multi.run_id
        assert multi.timestamp

    def test_custom_seed_strategy(self):
        items = _make_items()
        seed_items = _make_seed_items()
        strategy = SeedStrategy(stable_seeds=[42, 137], n_exploration_seeds=2)
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        multi = runner.run_multi(strategy)

        assert len(multi.stable_reports) == 2
        assert len(multi.exploration_reports) == 2
        assert multi.stable_reports[0].seed == 42
        assert multi.stable_reports[1].seed == 137

    def test_aggregate_metrics(self):
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        multi = runner.run_multi()
        am = multi.aggregate_metrics

        assert 0 <= am.mean_pass_rate <= 1
        assert am.min_pass_rate <= am.mean_pass_rate <= am.max_pass_rate
        assert am.worst_case_fpr >= 0
        assert isinstance(am.item_stability, dict)

    def test_multi_report_to_dict(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        multi = runner.run_multi()
        d = multi.to_dict()

        assert "run_id" in d
        assert "seed_reports" in d
        assert "aggregate_metrics" in d
        assert "summary" in d

    def test_multi_summary(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        multi = runner.run_multi()
        s = multi.summary()

        assert "seeds_evaluated" in s
        assert "stable_seeds" in s
        assert "exploration_seeds" in s
        assert "mean_pass_rate" in s

    def test_sample_size_passed_through(self):
        items = _make_items()
        seed_items = _make_seed_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        multi = runner.run_multi(
            SeedStrategy(stable_seeds=[42]),
            sample_size=1,
        )
        # Stable seed 42: 5 fixed + 1 seed-42 item = 6
        stable_report = multi.stable_reports[0].report
        assert stable_report.total_count == 6

    def test_analysis_cache_reuse(self):
        """Fixed items should be analyzed once across seeds, not per-seed."""
        call_count = 0
        original_analyze = FakePipeline.analyze

        def counting_analyze(self, item):
            nonlocal call_count
            call_count += 1
            return original_analyze(self, item)

        items = _make_items()
        seed_items = _make_seed_items()
        pipeline = FakePipeline()
        pipeline.analyze = counting_analyze.__get__(pipeline, FakePipeline)

        runner = BenchmarkRunner(
            pipeline=pipeline,
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items, seed_items),
            rules=FakeRules(),
        )
        # 2 stable seeds (42 and 137), no exploration
        runner.run_multi(SeedStrategy(stable_seeds=[42, 137], n_exploration_seeds=0))

        # 5 fixed items analyzed once (cached) + 2 seed-42 items + 1 seed-137 item = 8
        # Without caching it would be (5+2) + (5+1) = 13
        assert call_count == 8


class TestSeedStrategy:
    def test_defaults(self):
        s = SeedStrategy()
        assert s.stable_seeds == [42]
        assert s.n_exploration_seeds == 1

    def test_custom(self):
        s = SeedStrategy(stable_seeds=[1, 2, 3], n_exploration_seeds=5)
        assert len(s.stable_seeds) == 3
        assert s.n_exploration_seeds == 5


class TieredFakeDataset(FakeDataset):
    """FakeDataset with tiers support."""

    def tiers(self) -> dict[str, list[str]]:
        return {
            "smoke": ["anomalous"],
            "full": ["anomalous", "clean"],
        }


class TestTierFiltering:
    def test_tier_filters_categories(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=TieredFakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run(tier="smoke")
        # Only anomalous items
        categories = {r.category for r in report.item_results}
        assert categories == {"anomalous"}
        assert report.total_count == 3

    def test_unknown_tier_raises(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=TieredFakeDataset(items),
            rules=FakeRules(),
        )
        import pytest

        with pytest.raises(ValueError, match="Unknown tier"):
            runner.run(tier="nonexistent")

    def test_no_tiers_method_raises(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),  # No tiers() method
            rules=FakeRules(),
        )
        import pytest

        with pytest.raises(ValueError, match="does not support tiers"):
            runner.run(tier="smoke")

    def test_no_tier_processes_all(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=TieredFakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()  # No tier = all items
        assert report.total_count == 5


class TestItemFilter:
    def test_item_filter_reduces_items(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run(item_filter={"anomaly_high", "clean_1"})
        assert report.total_count == 2
        ids = {r.item_id for r in report.item_results}
        assert ids == {"anomaly_high", "clean_1"}

    def test_filter_with_tier_composes(self):
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=TieredFakeDataset(items),
            rules=FakeRules(),
        )
        # tier=smoke (anomalous only) + item_filter={anomaly_high}
        report = runner.run(tier="smoke", item_filter={"anomaly_high"})
        assert report.total_count == 1
        assert report.item_results[0].item_id == "anomaly_high"

    def test_filter_with_tier_intersection(self):
        """item_filter for a clean item + tier=smoke (anomalous only) = no items."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=TieredFakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run(tier="smoke", item_filter={"clean_1"})
        assert report.total_count == 0


class TestDiskCacheIntegration:
    def test_disk_cache_stores_results(self):
        from forge_world.core.cache import AnalysisCache
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AnalysisCache(os.path.join(tmpdir, "cache"))
            items = _make_items()
            runner = BenchmarkRunner(
                pipeline=FakePipeline(),
                aggregator=FakeAggregator(),
                dataset=FakeDataset(items),
                rules=FakeRules(),
                analysis_cache=cache,
            )
            runner.run()
            # Cache should have entries (for items that produced findings)
            assert cache.stats["misses"] > 0  # First run = all misses

    def test_disk_cache_avoids_reanalysis(self):
        from forge_world.core.cache import AnalysisCache
        import tempfile
        import os

        call_count = 0
        original_analyze = FakePipeline.analyze

        def counting_analyze(self, item):
            nonlocal call_count
            call_count += 1
            return original_analyze(self, item)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AnalysisCache(os.path.join(tmpdir, "cache"))
            items = _make_items()
            pipeline = FakePipeline()
            pipeline.analyze = counting_analyze.__get__(pipeline, FakePipeline)

            runner = BenchmarkRunner(
                pipeline=pipeline,
                aggregator=FakeAggregator(),
                dataset=FakeDataset(items),
                rules=FakeRules(),
                analysis_cache=cache,
            )

            # First run: all items analyzed
            runner.run()
            first_count = call_count

            # Second run: should use cache, zero analyze() calls
            runner.run()
            assert call_count == first_count  # No additional calls

    def test_config_change_invalidates(self):
        from forge_world.core.cache import AnalysisCache
        import tempfile
        import os

        call_count = 0
        original_analyze = FakePipeline.analyze

        def counting_analyze(self, item):
            nonlocal call_count
            call_count += 1
            return original_analyze(self, item)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AnalysisCache(os.path.join(tmpdir, "cache"))
            items = _make_items()

            # First run with config A
            pipeline_a = FakePipeline(config={"threshold": 0.5})
            pipeline_a.analyze = counting_analyze.__get__(pipeline_a, FakePipeline)
            runner_a = BenchmarkRunner(
                pipeline=pipeline_a,
                aggregator=FakeAggregator(),
                dataset=FakeDataset(items),
                rules=FakeRules(),
                analysis_cache=cache,
            )
            runner_a.run()
            count_after_a = call_count

            # Second run with config B (different threshold = different config_hash)
            pipeline_b = FakePipeline(config={"threshold": 0.3})
            pipeline_b.analyze = counting_analyze.__get__(pipeline_b, FakePipeline)
            runner_b = BenchmarkRunner(
                pipeline=pipeline_b,
                aggregator=FakeAggregator(),
                dataset=FakeDataset(items),
                rules=FakeRules(),
                analysis_cache=cache,
            )
            runner_b.run()
            # Should have analyzed all items again (different config_hash)
            assert call_count == count_after_a + len(items)

    def test_three_layer_ordering(self):
        """in-memory cache > disk cache > pipeline.analyze(), in that order."""
        from forge_world.core.cache import AnalysisCache
        import tempfile
        import os

        analyze_calls = 0
        original_analyze = FakePipeline.analyze

        def counting_analyze(self, item):
            nonlocal analyze_calls
            analyze_calls += 1
            return original_analyze(self, item)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AnalysisCache(os.path.join(tmpdir, "cache"))
            items = _make_items()
            pipeline = FakePipeline()
            pipeline.analyze = counting_analyze.__get__(pipeline, FakePipeline)

            runner = BenchmarkRunner(
                pipeline=pipeline,
                aggregator=FakeAggregator(),
                dataset=FakeDataset(items),
                rules=FakeRules(),
                analysis_cache=cache,
            )

            # Run 1: all items hit analyze (layer 3), populate disk cache
            in_memory: dict = {}
            runner.run(_analysis_cache=in_memory)
            assert analyze_calls == 5  # All 5 items analyzed
            assert len(in_memory) == 5  # All in in-memory cache

            # Run 2 with same in-memory cache: should use layer 1 (in-memory), zero analyze
            analyze_calls = 0
            runner.run(_analysis_cache=in_memory)
            assert analyze_calls == 0  # All from in-memory
            assert cache.stats["hits"] == 0  # Disk cache not even consulted

            # Run 3 with empty in-memory but populated disk: should use layer 2 (disk)
            analyze_calls = 0
            fresh_in_memory: dict = {}
            runner.run(_analysis_cache=fresh_in_memory)
            assert analyze_calls == 0  # All from disk cache
            assert cache.stats["hits"] == 5  # Disk cache served all 5
            # Items should have been promoted to in-memory
            assert len(fresh_in_memory) == 5


class TestDeriveExplorationSeeds:
    def test_deterministic(self):
        seeds1 = _derive_exploration_seeds("run123", 3)
        seeds2 = _derive_exploration_seeds("run123", 3)
        assert seeds1 == seeds2

    def test_different_run_ids(self):
        seeds1 = _derive_exploration_seeds("run_a", 3)
        seeds2 = _derive_exploration_seeds("run_b", 3)
        assert seeds1 != seeds2

    def test_count(self):
        seeds = _derive_exploration_seeds("run1", 5)
        assert len(seeds) == 5
        # All should be non-negative
        assert all(s >= 0 for s in seeds)


class TestPerformanceMetrics:
    def test_to_dict(self):
        perf = PerformanceMetrics(
            latency_mean_ms=10.5,
            latency_p50_ms=9.0,
            latency_p95_ms=20.0,
            latency_p99_ms=30.0,
            total_time_ms=100.0,
            throughput_items_per_sec=50.0,
            item_count=5,
        )
        d = perf.to_dict()
        assert d["latency_mean_ms"] == 10.5
        assert d["latency_p50_ms"] == 9.0
        assert d["latency_p95_ms"] == 20.0
        assert d["latency_p99_ms"] == 30.0
        assert d["item_count"] == 5

    def test_compute_known_latencies(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        perf = _compute_performance_metrics(latencies, 200.0)
        assert perf is not None
        assert perf.latency_mean_ms == 30.0
        assert perf.item_count == 5
        assert perf.total_time_ms == 200.0
        assert perf.throughput_items_per_sec == 25.0  # 5 / 200ms * 1000

    def test_compute_empty_returns_none(self):
        perf = _compute_performance_metrics([], 100.0)
        assert perf is None

    def test_compute_single_latency(self):
        perf = _compute_performance_metrics([42.0], 50.0)
        assert perf is not None
        assert perf.latency_mean_ms == 42.0
        assert perf.latency_p50_ms == 42.0
        assert perf.latency_p95_ms == 42.0
        assert perf.latency_p99_ms == 42.0
        assert perf.item_count == 1


class TestBenchmarkRunnerPerformance:
    def test_run_produces_performance(self):
        """BenchmarkRunner.run() should produce performance metrics."""
        items = [
            LabeledItem(
                id="item1",
                category="test",
                expected_label="findings",
                data={"id": "item1", "score": 0.9, "method": "test"},
            ),
            LabeledItem(
                id="item2",
                category="clean",
                expected_label="clean",
                data={"id": "item2", "score": 0.1, "method": "test"},
            ),
        ]
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        assert report.performance is not None
        assert report.performance.item_count == 2
        assert report.performance.latency_mean_ms > 0
        assert report.performance.total_time_ms > 0

    def test_cached_items_excluded_from_latency(self):
        """Items served from cache should NOT be included in latency stats."""
        items = [
            LabeledItem(
                id="item1",
                category="test",
                expected_label="findings",
                data={"id": "item1", "score": 0.9, "method": "test"},
            ),
            LabeledItem(
                id="item2",
                category="clean",
                expected_label="clean",
                data={"id": "item2", "score": 0.1, "method": "test"},
            ),
        ]
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )

        # First run: all items analyzed
        report1 = runner.run()
        assert report1.performance is not None
        assert report1.performance.item_count == 2

        # Second run with in-memory cache: all cached
        cache = {"item1": [], "item2": []}
        report2 = runner.run(_analysis_cache=cache)
        # Performance should be None (no items analyzed)
        assert report2.performance is None

    def test_performance_in_to_dict(self):
        """Performance metrics should appear in report.to_dict()."""
        items = [
            LabeledItem(
                id="item1",
                category="test",
                expected_label="findings",
                data={"id": "item1", "score": 0.9, "method": "test"},
            ),
        ]
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = runner.run()
        d = report.to_dict()
        assert "performance" in d
        assert d["performance"]["item_count"] == 1
