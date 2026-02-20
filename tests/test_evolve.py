"""Tests for the evolution loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forge_world.core.evolve import (
    EvolutionConfig,
    EvolutionLoop,
    EvolutionResult,
    IterationResult,
    _extract_metrics,
)
from forge_world.core.metrics import ConfusionMatrix
from forge_world.core.runner import BenchmarkReport, ItemResult
from forge_world.core.snapshots import SnapshotManager


def _make_report(pass_rate: float = 0.8, sensitivity: float = 0.8, fpr: float = 0.0):
    """Build a mock report with controllable metrics."""
    # Build items that produce desired metrics
    tp = 8
    fn = 2
    tn = 2
    fp = 0
    if fpr > 0:
        fp = 1
        tn = 1

    results = []
    for i in range(tp):
        results.append(ItemResult(
            item_id=f"findings_{i}", category="test", expected_label="findings",
            passed=True, risk_level="high", confidence=0.9,
            converging_evidence=False, findings=[], methods_flagged=[],
        ))
    for i in range(fn):
        results.append(ItemResult(
            item_id=f"missed_{i}", category="test", expected_label="findings",
            passed=False, risk_level="low", confidence=0.2,
            converging_evidence=False, findings=[], methods_flagged=[],
        ))
    for i in range(tn):
        results.append(ItemResult(
            item_id=f"clean_{i}", category="clean", expected_label="clean",
            passed=True, risk_level="clean", confidence=0.0,
            converging_evidence=False, findings=[], methods_flagged=[],
        ))
    for i in range(fp):
        results.append(ItemResult(
            item_id=f"fp_{i}", category="clean", expected_label="clean",
            passed=False, risk_level="high", confidence=0.8,
            converging_evidence=False, findings=[], methods_flagged=[],
        ))

    cm = ConfusionMatrix(
        true_positives=tp, true_negatives=tn,
        false_positives=fp, false_negatives=fn,
    )
    return BenchmarkReport(
        run_id="test", timestamp="2025-01-01T00:00:00Z", git_sha="abc",
        config_hash="cfg", item_results=results, confusion_matrix=cm,
        category_metrics=[], method_metrics={},
    )


class TestConstraintChecking:
    def test_fpr_zero_passes(self):
        config = EvolutionConfig(agent_command="echo test")
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        metrics = {"fpr": 0.0, "sensitivity": 0.8}
        violations = loop._check_constraints(metrics)
        assert violations == []

    def test_fpr_nonzero_fails(self):
        config = EvolutionConfig(
            agent_command="echo test",
            hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
        )
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        metrics = {"fpr": 0.05, "sensitivity": 0.9}
        violations = loop._check_constraints(metrics)
        assert len(violations) == 1
        assert "fpr" in violations[0]

    def test_ge_constraint(self):
        config = EvolutionConfig(
            agent_command="echo test",
            hard_constraints=[{"metric": "sensitivity", "op": ">=", "value": 0.5}],
        )
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        # Passes
        assert loop._check_constraints({"sensitivity": 0.6}) == []
        # Fails
        assert len(loop._check_constraints({"sensitivity": 0.4})) == 1


class TestIsImproved:
    def test_higher_sensitivity_is_improvement(self):
        config = EvolutionConfig(
            agent_command="echo test",
            optimization_target={"metric": "sensitivity", "direction": "max"},
        )
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        before = {"sensitivity": 0.8}
        after = {"sensitivity": 0.85}
        assert loop._is_improved(before, after) is True

    def test_lower_sensitivity_not_improvement(self):
        config = EvolutionConfig(
            agent_command="echo test",
            optimization_target={"metric": "sensitivity", "direction": "max"},
        )
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        before = {"sensitivity": 0.8}
        after = {"sensitivity": 0.75}
        assert loop._is_improved(before, after) is False

    def test_min_direction(self):
        config = EvolutionConfig(
            agent_command="echo test",
            optimization_target={"metric": "fpr", "direction": "min"},
        )
        loop = EvolutionLoop(
            config=config,
            runner=MagicMock(),
            snapshot_manager=MagicMock(),
        )
        before = {"fpr": 0.05}
        after = {"fpr": 0.02}
        assert loop._is_improved(before, after) is True


class TestEvolutionLoop:
    @patch("forge_world.core.evolve.subprocess.run")
    def test_stops_at_max_iterations(self, mock_run):
        """Loop stops after max_iterations even if changes keep happening."""
        report = _make_report()
        runner = MagicMock()
        runner.run.return_value = report
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        # Agent always succeeds but makes no improvement
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = EvolutionConfig(
            agent_command="echo test",
            max_iterations=3,
            convergence_patience=10,  # High so we hit max_iterations
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        # Patch _check_for_changes to always return False (no changes)
        with patch.object(loop, "_check_for_changes", return_value=False):
            result = loop.run()

        assert result.total_iterations == 3
        assert result.convergence_reason == "max_iterations"

    @patch("forge_world.core.evolve.subprocess.run")
    def test_stops_on_convergence(self, mock_run):
        """Loop stops after patience rounds with no improvement."""
        report = _make_report()
        runner = MagicMock()
        runner.run.return_value = report
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = EvolutionConfig(
            agent_command="echo test",
            max_iterations=20,
            convergence_patience=2,
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        with patch.object(loop, "_check_for_changes", return_value=False):
            result = loop.run()

        assert result.total_iterations == 2
        assert result.convergence_reason == "converged"

    @patch("forge_world.core.evolve.subprocess.run")
    def test_rejects_constraint_violation(self, mock_run):
        """Changes that violate constraints are rolled back."""
        report_good = _make_report(fpr=0.0)
        report_bad = _make_report(fpr=0.05)

        runner = MagicMock()
        # First call = initial bench, second call = after agent changes
        runner.run.side_effect = [report_good, report_bad]
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        mock_run.return_value = MagicMock(returncode=0, stdout="M file.py", stderr="")

        config = EvolutionConfig(
            agent_command="echo test",
            max_iterations=1,
            hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        with patch.object(loop, "_check_for_changes", return_value=True):
            result = loop.run()

        assert result.iterations[0].constraint_violated is True
        assert result.iterations[0].accepted is False

    @patch("forge_world.core.evolve.subprocess.run")
    def test_accepts_improvement(self, mock_run):
        """Changes that improve the target metric are committed."""
        report_before = _make_report()
        # Make a report that has better sensitivity
        report_after = _make_report()
        # Override the confusion matrix for better sensitivity
        report_after._improved_cm = True

        runner = MagicMock()
        runner.run.side_effect = [report_before, report_after]
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = EvolutionConfig(
            agent_command="echo test",
            max_iterations=1,
            optimization_target={"metric": "pass_rate", "direction": "max"},
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        # Mock: has changes and improved
        with (
            patch.object(loop, "_check_for_changes", return_value=True),
            patch.object(loop, "_is_improved", return_value=True),
            patch.object(loop, "_commit_changes"),
        ):
            result = loop.run()

        assert result.iterations[0].accepted is True
        assert result.accepted_iterations == 1

    @patch("forge_world.core.evolve.subprocess.run")
    def test_agent_no_changes(self, mock_run):
        """Agent producing no changes counts as no-improvement."""
        report = _make_report()
        runner = MagicMock()
        runner.run.return_value = report
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = EvolutionConfig(
            agent_command="echo test",
            max_iterations=1,
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        with patch.object(loop, "_check_for_changes", return_value=False):
            result = loop.run()

        assert result.iterations[0].had_changes is False
        assert result.iterations[0].accepted is False

    @patch("forge_world.core.evolve.subprocess.run")
    def test_agent_error(self, mock_run):
        """Non-zero agent exit is handled gracefully."""
        report = _make_report()
        runner = MagicMock()
        runner.run.return_value = report
        sm = MagicMock(spec=SnapshotManager)
        sm.check.side_effect = FileNotFoundError("no baseline")

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        config = EvolutionConfig(
            agent_command="false",
            max_iterations=5,
        )
        loop = EvolutionLoop(config=config, runner=runner, snapshot_manager=sm)

        result = loop.run()
        assert result.convergence_reason == "agent_error"
        assert result.total_iterations == 1


class TestEvolutionResult:
    def test_to_markdown(self):
        result = EvolutionResult(
            iterations=[
                IterationResult(iteration=1, had_changes=True, constraint_violated=False, accepted=True),
                IterationResult(iteration=2, had_changes=False, constraint_violated=False),
            ],
            total_iterations=2,
            accepted_iterations=1,
            starting_metrics={"sensitivity": 0.8, "fpr": 0.0},
            final_metrics={"sensitivity": 0.85, "fpr": 0.0},
            convergence_reason="converged",
        )
        md = result.to_markdown()
        assert "# Evolution Result" in md
        assert "Total iterations" in md
        assert "converged" in md

    def test_to_dict(self):
        result = EvolutionResult(
            total_iterations=3,
            accepted_iterations=2,
            starting_metrics={"sensitivity": 0.8},
            final_metrics={"sensitivity": 0.9},
            convergence_reason="max_iterations",
        )
        d = result.to_dict()
        assert d["total_iterations"] == 3
        assert d["convergence_reason"] == "max_iterations"


class TestExtractMetrics:
    def test_single_report(self):
        report = _make_report()
        metrics = _extract_metrics(report)
        assert "pass_rate" in metrics
        assert "sensitivity" in metrics
        assert "fpr" in metrics
        assert "f1" in metrics
