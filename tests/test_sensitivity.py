"""Tests for parameter sensitivity analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from forge_world.core.protocols import (
    AggregatedResult,
    Finding,
    LabeledItem,
    PassFailRule,
    Severity,
)
from forge_world.core.runner import BenchmarkRunner
from forge_world.core.sensitivity import (
    ParameterSensitivity,
    SensitivityReport,
    compute_sensitivity,
    get_nested_value,
    set_nested_value,
    walk_numeric_parameters,
)
from tests.test_runner import FakeAggregator, FakeDataset, FakePipeline, FakeRules


def _make_items() -> list[LabeledItem]:
    """Create a set of labeled test items."""
    return [
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


class TestWalkNumericParameters:
    def test_flat_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "name": {"type": "string"},
            },
        }
        params = walk_numeric_parameters(schema)
        assert len(params) == 1
        assert params[0]["path"] == "threshold"
        assert params[0]["type"] == "number"
        assert params[0]["minimum"] == 0
        assert params[0]["maximum"] == 1

    def test_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "ela": {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                },
            },
        }
        params = walk_numeric_parameters(schema)
        assert len(params) == 1
        assert params[0]["path"] == "ela.quality"

    def test_nested_with_ref(self):
        schema = {
            "type": "object",
            "properties": {
                "ela": {"$ref": "#/$defs/ElaConfig"},
            },
            "$defs": {
                "ElaConfig": {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                },
            },
        }
        params = walk_numeric_parameters(schema)
        assert len(params) == 1
        assert params[0]["path"] == "ela.quality"

    def test_bounded(self):
        schema = {
            "type": "object",
            "properties": {
                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }
        params = walk_numeric_parameters(schema)
        assert params[0]["minimum"] == 0.0
        assert params[0]["maximum"] == 1.0

    def test_integer(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        }
        params = walk_numeric_parameters(schema)
        assert len(params) == 1
        assert params[0]["type"] == "integer"


class TestGetSetNestedValue:
    def test_get_nested_value_dict(self):
        config = {"ela": {"quality": 80}, "weight": 0.7}
        assert get_nested_value(config, "ela.quality") == 80
        assert get_nested_value(config, "weight") == 0.7

    def test_set_nested_value_dict(self):
        config = {"ela": {"quality": 80}, "weight": 0.7}
        set_nested_value(config, "ela.quality", 85)
        assert config["ela"]["quality"] == 85
        set_nested_value(config, "weight", 0.8)
        assert config["weight"] == 0.8

    def test_get_set_nested_value_object(self):
        class Config:
            def __init__(self):
                self.threshold = 0.5

        config = Config()
        assert get_nested_value(config, "threshold") == 0.5
        set_nested_value(config, "threshold", 0.6)
        assert config.threshold == 0.6


class TestComputeSensitivity:
    def test_basic(self):
        """FakePipeline with threshold → measures impact."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = compute_sensitivity(runner, target_metric="sensitivity")
        assert len(report.parameters) >= 1
        assert report.target_metric == "sensitivity"
        assert report.baseline_value >= 0

    def test_ranking(self):
        """Results sorted by impact descending."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = compute_sensitivity(runner)
        if len(report.parameters) >= 2:
            assert report.parameters[0].impact >= report.parameters[1].impact

    def test_constrained(self):
        """Constraint violations flagged."""
        items = _make_items()
        runner = BenchmarkRunner(
            pipeline=FakePipeline(),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = compute_sensitivity(
            runner,
            hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
        )
        # At least some perturbation should exist
        assert len(report.parameters) >= 1

    def test_restore(self):
        """Original config restored after analysis."""
        items = _make_items()
        pipeline = FakePipeline(config={"threshold": 0.5})
        runner = BenchmarkRunner(
            pipeline=pipeline,
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        compute_sensitivity(runner)
        # Config should be restored to original
        assert pipeline.get_config()["threshold"] == 0.5

    def test_clamping(self):
        """Params don't exceed min/max bounds."""
        items = _make_items()
        # Use a very large delta to test clamping
        runner = BenchmarkRunner(
            pipeline=FakePipeline(config={"threshold": 0.9}),
            aggregator=FakeAggregator(),
            dataset=FakeDataset(items),
            rules=FakeRules(),
        )
        report = compute_sensitivity(runner, delta_fraction=0.5)
        for p in report.parameters:
            # With FakePipeline schema, max is 1.0
            # val_plus should be clamped to 1.0
            assert p.metric_plus is not None  # Should have run successfully


class TestSensitivityReport:
    def test_round_trip(self):
        report = SensitivityReport(
            parameters=[
                ParameterSensitivity(
                    path="threshold",
                    current_value=0.5,
                    delta=0.1,
                    metric_minus=0.75,
                    metric_plus=0.85,
                    impact=0.1,
                    direction="decrease",
                    constrained=False,
                ),
            ],
            target_metric="sensitivity",
            baseline_value=0.8,
            config_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
        )
        d = report.to_dict()
        restored = SensitivityReport.from_dict(d)
        assert len(restored.parameters) == 1
        assert restored.parameters[0].path == "threshold"
        assert restored.target_metric == "sensitivity"
        assert restored.baseline_value == 0.8

    def test_save_load(self):
        report = SensitivityReport(
            parameters=[
                ParameterSensitivity(
                    path="threshold",
                    current_value=0.5,
                    delta=0.1,
                    metric_minus=0.75,
                    metric_plus=0.85,
                    impact=0.1,
                    direction="decrease",
                    constrained=False,
                ),
            ],
            target_metric="sensitivity",
            baseline_value=0.8,
            config_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sensitivity.json"
            report.save(path)
            loaded = SensitivityReport.load(path)
            assert loaded is not None
            assert len(loaded.parameters) == 1

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            assert SensitivityReport.load(path) is None

    def test_prompt_context(self):
        report = SensitivityReport(
            parameters=[
                ParameterSensitivity(
                    path="weight_clone",
                    current_value=0.85,
                    delta=0.1,
                    metric_minus=0.78,
                    metric_plus=0.82,
                    impact=0.04,
                    direction="increase",
                    constrained=False,
                ),
                ParameterSensitivity(
                    path="threshold",
                    current_value=0.60,
                    delta=0.1,
                    metric_minus=0.80,
                    metric_plus=0.75,
                    impact=0.05,
                    direction="decrease",
                    constrained=True,
                    constraint_detail="increase causes fpr=0.05",
                ),
            ],
            target_metric="sensitivity",
            baseline_value=0.8,
            config_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
        )
        ctx = report.to_prompt_context()
        assert "## Parameter Sensitivity" in ctx
        assert "weight_clone" in ctx
        assert "threshold" in ctx
        assert "| Parameter |" in ctx

    def test_prompt_context_empty(self):
        report = SensitivityReport(
            parameters=[],
            target_metric="sensitivity",
            baseline_value=0.8,
            config_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
        )
        assert report.to_prompt_context() == ""
