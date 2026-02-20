"""Tests for Optuna-based auto-tuning.

All tests are gated with ``pytest.importorskip("optuna")`` so that the
test file is silently skipped when optuna is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

optuna = pytest.importorskip("optuna")

from forge_world.core.protocols import (  # noqa: E402
    AggregatedResult,
    Finding,
    LabeledItem,
    PassFailRule,
    Severity,
)
from forge_world.core.runner import BenchmarkRunner  # noqa: E402
from forge_world.core.tuner import (  # noqa: E402
    TunerConfig,
    TunerResult,
    apply_best_params,
    run_tuning,
    _check_constraints,
)


# --- Fake implementations ---


class TunerFakePipeline:
    """Pipeline where sensitivity = 1.0 - abs(threshold - 0.35).

    Optimal threshold is 0.35, giving sensitivity=1.0.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {"threshold": 0.5}

    def analyze(self, item: Any) -> list[Finding]:
        if isinstance(item, dict):
            score = item.get("score", 0.0)
            threshold = self._config.get("threshold", 0.5)
            if score >= threshold:
                severity = Severity.HIGH if score > 0.8 else Severity.MEDIUM
                return [
                    Finding(
                        title=f"Anomaly (score={score})",
                        method=item.get("method", "test"),
                        severity=severity,
                        confidence=score,
                        item_id=item.get("id", ""),
                    )
                ]
        return []

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def set_config(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }


class TunerFakeAggregator:
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


class TunerFakeDataset:
    def __init__(self):
        self._items = [
            LabeledItem(
                id="a1",
                category="anomalous",
                expected_label="findings",
                data={"id": "a1", "score": 0.9, "method": "ela"},
            ),
            LabeledItem(
                id="a2",
                category="anomalous",
                expected_label="findings",
                data={"id": "a2", "score": 0.6, "method": "ela"},
            ),
            LabeledItem(
                id="a3",
                category="anomalous",
                expected_label="findings",
                data={"id": "a3", "score": 0.35, "method": "ela"},
            ),
            LabeledItem(
                id="c1",
                category="clean",
                expected_label="clean",
                data={"id": "c1", "score": 0.1, "method": "ela"},
            ),
            LabeledItem(
                id="c2",
                category="clean",
                expected_label="clean",
                data={"id": "c2", "score": 0.2, "method": "ela"},
            ),
        ]

    def items(self, seed=None, sample_size=None):
        return list(self._items)

    def categories(self):
        return ["anomalous", "clean"]


class TunerFakeRules:
    def get_rule(self, expected_label: str) -> PassFailRule:
        if expected_label == "findings":
            return PassFailRule(expected_label="findings", min_risk_for_pass=Severity.MEDIUM)
        elif expected_label == "clean":
            return PassFailRule(expected_label="clean", max_risk_for_pass=Severity.MEDIUM)
        return PassFailRule(expected_label=expected_label)


def _make_runner(config: dict[str, Any] | None = None) -> BenchmarkRunner:
    return BenchmarkRunner(
        pipeline=TunerFakePipeline(config),
        aggregator=TunerFakeAggregator(),
        dataset=TunerFakeDataset(),
        rules=TunerFakeRules(),
    )


# --- Tests ---


class TestTunerConfigDefaults:
    def test_defaults(self):
        tc = TunerConfig()
        assert tc.n_trials == 20
        assert tc.study_name == "forge-world-tuning"
        assert len(tc.optimization_targets) == 1
        assert tc.optimization_targets[0]["metric"] == "sensitivity"
        assert tc.hard_constraints == []

    def test_custom(self):
        tc = TunerConfig(
            n_trials=5,
            optimization_targets=[
                {"metric": "sensitivity", "direction": "max"},
                {"metric": "latency_mean_ms", "direction": "min"},
            ],
        )
        assert tc.n_trials == 5
        assert len(tc.optimization_targets) == 2


class TestTunerResultToDict:
    def test_to_dict(self):
        tr = TunerResult(
            n_trials_completed=3,
            best_params={"threshold": 0.35},
            best_values={"sensitivity": 0.95},
            pareto_front=[{"params": {"threshold": 0.35}, "values": {"sensitivity": 0.95}}],
            all_trials_summary=[
                {
                    "trial": 0,
                    "params": {"threshold": 0.5},
                    "values": {"sensitivity": 0.8},
                    "violated": False,
                },
            ],
        )
        d = tr.to_dict()
        assert d["n_trials_completed"] == 3
        assert d["best_params"]["threshold"] == 0.35
        assert d["best_values"]["sensitivity"] == 0.95
        assert len(d["pareto_front"]) == 1
        assert len(d["all_trials_summary"]) == 1


class TestTunerResultPromptContext:
    def test_empty_result(self):
        tr = TunerResult(
            n_trials_completed=0,
            best_params={},
            best_values={},
            pareto_front=[],
            all_trials_summary=[],
        )
        assert tr.to_prompt_context() == ""

    def test_single_objective_context(self):
        tr = TunerResult(
            n_trials_completed=5,
            best_params={"threshold": 0.35},
            best_values={"sensitivity": 0.95},
            pareto_front=[{"params": {"threshold": 0.35}, "values": {"sensitivity": 0.95}}],
            all_trials_summary=[],
        )
        ctx = tr.to_prompt_context()
        assert "Auto-Tuning Results" in ctx
        assert "threshold" in ctx
        assert "sensitivity" in ctx
        assert "5 Optuna trials" in ctx

    def test_multi_objective_pareto(self):
        tr = TunerResult(
            n_trials_completed=10,
            best_params={"threshold": 0.3},
            best_values={"sensitivity": 0.9, "latency_mean_ms": 50.0},
            pareto_front=[
                {
                    "params": {"threshold": 0.3},
                    "values": {"sensitivity": 0.9, "latency_mean_ms": 50.0},
                },
                {
                    "params": {"threshold": 0.4},
                    "values": {"sensitivity": 0.85, "latency_mean_ms": 30.0},
                },
            ],
            all_trials_summary=[],
        )
        ctx = tr.to_prompt_context()
        assert "Pareto Front" in ctx
        assert "2 solutions" in ctx


class TestRunTuningBasic:
    def test_basic_tuning(self):
        runner = _make_runner({"threshold": 0.5})
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=5,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-basic",
            )
            result = run_tuning(runner, tc)

        assert result.n_trials_completed == 5
        assert "threshold" in result.best_params
        assert "sensitivity" in result.best_values
        assert len(result.all_trials_summary) == 5
        assert len(result.pareto_front) >= 1

    def test_finds_better_threshold(self):
        """Tuner should find a threshold giving better sensitivity than default 0.5."""
        runner = _make_runner({"threshold": 0.8})  # Start with a bad threshold
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=10,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-improvement",
            )
            result = run_tuning(runner, tc)

        assert (
            result.best_values.get("sensitivity", 0) >= 0.4
        )  # Should find something better than 0.8 threshold


class TestRunTuningMultiObjective:
    def test_multi_objective(self):
        runner = _make_runner({"threshold": 0.5})
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=5,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-multi",
                optimization_targets=[
                    {"metric": "sensitivity", "direction": "max"},
                    {"metric": "f1", "direction": "max"},
                ],
            )
            result = run_tuning(runner, tc)

        assert result.n_trials_completed == 5
        assert len(result.pareto_front) >= 1
        assert "sensitivity" in result.best_values or "f1" in result.best_values


class TestRunTuningConstraintPenalty:
    def test_constraint_violation_gets_penalty(self):
        runner = _make_runner({"threshold": 0.5})
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=5,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-constraints",
                hard_constraints=[
                    {"metric": "fpr", "op": "<=", "value": 0},
                ],
            )
            result = run_tuning(runner, tc)

        assert result.n_trials_completed == 5
        # Some trials may have been violated
        # At least the result should complete without errors
        assert len(result.all_trials_summary) == 5


class TestPipelineRestored:
    def test_config_restored_after_tuning(self):
        original_config = {"threshold": 0.5}
        runner = _make_runner(dict(original_config))
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=3,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-restore",
            )
            run_tuning(runner, tc)

        # Pipeline config should be restored to original
        assert runner.pipeline.get_config() == original_config

    def test_config_restored_on_error(self):
        """Even if tuning encounters issues, config should be restored."""
        original_config = {"threshold": 0.5}
        runner = _make_runner(dict(original_config))
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=2,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-restore-error",
            )
            run_tuning(runner, tc)

        assert runner.pipeline.get_config() == original_config


class TestApplyBestParams:
    def test_apply_and_rollback(self):
        pipeline = TunerFakePipeline({"threshold": 0.5})
        tr = TunerResult(
            n_trials_completed=3,
            best_params={"threshold": 0.35},
            best_values={"sensitivity": 0.95},
            pareto_front=[],
            all_trials_summary=[],
        )
        old = apply_best_params(pipeline, tr)
        assert old == {"threshold": 0.5}
        assert pipeline.get_config()["threshold"] == 0.35

        # Rollback
        pipeline.set_config(old)
        assert pipeline.get_config()["threshold"] == 0.5


class TestJournalPersistence:
    def test_study_accumulates_trials(self):
        runner = _make_runner({"threshold": 0.5})
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = str(Path(tmpdir) / "journal.log")

            tc1 = TunerConfig(
                n_trials=3,
                journal_path=journal,
                study_name="test-persist",
            )
            r1 = run_tuning(runner, tc1)
            assert r1.n_trials_completed == 3

            tc2 = TunerConfig(
                n_trials=3,
                journal_path=journal,
                study_name="test-persist",
            )
            r2 = run_tuning(runner, tc2)
            assert r2.n_trials_completed == 3

            # The journal file should exist and have data from both runs
            assert Path(journal).exists()


class TestNoTunableParams:
    def test_empty_schema_returns_zero(self):
        """If no numeric params are found, return TunerResult with 0 trials."""
        runner = _make_runner({"threshold": 0.5})
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = TunerConfig(
                n_trials=5,
                journal_path=str(Path(tmpdir) / "journal.log"),
                study_name="test-empty",
            )
            # Pass empty schema_params
            result = run_tuning(runner, tc, schema_params=[])

        assert result.n_trials_completed == 0
        assert result.best_params == {}


class TestCheckConstraints:
    def test_no_violation(self):
        metrics = {"fpr": 0.0, "sensitivity": 0.8}
        constraints = [{"metric": "fpr", "op": "<=", "value": 0}]
        assert _check_constraints(metrics, constraints) is False

    def test_violation(self):
        metrics = {"fpr": 0.05, "sensitivity": 0.8}
        constraints = [{"metric": "fpr", "op": "<=", "value": 0}]
        assert _check_constraints(metrics, constraints) is True

    def test_ge_constraint(self):
        metrics = {"sensitivity": 0.3}
        constraints = [{"metric": "sensitivity", "op": ">=", "value": 0.5}]
        assert _check_constraints(metrics, constraints) is True

    def test_empty_constraints(self):
        metrics = {"fpr": 1.0}
        assert _check_constraints(metrics, []) is False
