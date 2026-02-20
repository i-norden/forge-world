"""Tests for iteration memory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_world.core.memory import (
    EvolutionMemory,
    MemoryEntry,
    ParameterChange,
    compute_parameter_diff,
)


class TestParameterChange:
    def test_round_trip(self):
        pc = ParameterChange(path="weight_ela", old_value=0.7, new_value=0.75)
        d = pc.to_dict()
        restored = ParameterChange.from_dict(d)
        assert restored.path == "weight_ela"
        assert restored.old_value == 0.7
        assert restored.new_value == 0.75


class TestMemoryEntry:
    def test_round_trip(self):
        entry = MemoryEntry(
            iteration=1,
            timestamp="2025-01-01T00:00:00Z",
            files_changed=["config.py"],
            parameter_changes=[ParameterChange(path="threshold", old_value=0.5, new_value=0.6)],
            metrics_before={"sensitivity": 0.8, "fpr": 0.0},
            metrics_after={"sensitivity": 0.82, "fpr": 0.0},
            accepted=True,
            reason="improved",
            constraint_violations=[],
            agent_reasoning="Increased threshold to improve sensitivity",
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.iteration == 1
        assert restored.accepted is True
        assert restored.reason == "improved"
        assert len(restored.parameter_changes) == 1
        assert restored.parameter_changes[0].path == "threshold"
        assert restored.agent_reasoning == "Increased threshold to improve sensitivity"


class TestEvolutionMemory:
    def _make_entry(
        self,
        iteration: int = 1,
        accepted: bool = True,
        reason: str = "improved",
        sensitivity_before: float = 0.8,
        sensitivity_after: float = 0.82,
        fpr_after: float = 0.0,
        parameter_changes: list[ParameterChange] | None = None,
        constraint_violations: list[str] | None = None,
    ) -> MemoryEntry:
        return MemoryEntry(
            iteration=iteration,
            timestamp="2025-01-01T00:00:00Z",
            files_changed=["config.py"],
            parameter_changes=parameter_changes or [],
            metrics_before={"sensitivity": sensitivity_before, "fpr": 0.0},
            metrics_after={"sensitivity": sensitivity_after, "fpr": fpr_after},
            accepted=accepted,
            reason=reason,
            constraint_violations=constraint_violations or [],
        )

    def test_add_entry_updates_best(self):
        mem = EvolutionMemory()
        entry1 = self._make_entry(iteration=1, sensitivity_after=0.82)
        mem.add_entry(entry1)
        assert mem.best_metrics["sensitivity"] == 0.82
        assert mem.best_iteration == 1

        entry2 = self._make_entry(iteration=2, sensitivity_after=0.85)
        mem.add_entry(entry2)
        assert mem.best_metrics["sensitivity"] == 0.85
        assert mem.best_iteration == 2

    def test_add_entry_rejected_does_not_update_best(self):
        mem = EvolutionMemory()
        entry = self._make_entry(accepted=False, reason="not_improved")
        mem.add_entry(entry)
        assert mem.best_metrics == {}
        assert mem.best_iteration == 0

    def test_save_and_load(self):
        mem = EvolutionMemory()
        entry = self._make_entry()
        mem.add_entry(entry)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem.save(path)
            assert path.exists()

            loaded = EvolutionMemory.load(path)
            assert len(loaded.entries) == 1
            assert loaded.entries[0].iteration == 1
            assert loaded.best_metrics["sensitivity"] == 0.82
            assert loaded.best_iteration == 1

    def test_load_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            mem = EvolutionMemory.load(path)
            assert len(mem.entries) == 0
            assert mem.best_metrics == {}

    def test_to_prompt_context_with_entries(self):
        mem = EvolutionMemory()
        mem.add_entry(
            self._make_entry(
                iteration=1,
                accepted=False,
                reason="constraint_violated:fpr=0.05>0",
                sensitivity_after=0.85,
                fpr_after=0.05,
                parameter_changes=[
                    ParameterChange(path="convergence_threshold", old_value=0.60, new_value=0.40)
                ],
                constraint_violations=["fpr=0.05>0"],
            )
        )
        mem.add_entry(
            self._make_entry(
                iteration=2,
                accepted=True,
                reason="improved",
                sensitivity_before=0.80,
                sensitivity_after=0.82,
                parameter_changes=[
                    ParameterChange(path="weight_clone", old_value=0.85, new_value=0.90)
                ],
            )
        )

        ctx = mem.to_prompt_context()
        assert "## Iteration History" in ctx
        assert "REJECTED" in ctx
        assert "ACCEPTED" in ctx
        assert "## What NOT to Try Again" in ctx
        assert "## Best So Far" in ctx

    def test_to_prompt_context_empty(self):
        mem = EvolutionMemory()
        ctx = mem.to_prompt_context()
        assert ctx == ""

    def test_what_not_to_try_constraint(self):
        mem = EvolutionMemory()
        mem.add_entry(
            self._make_entry(
                accepted=False,
                reason="constraint_violated:fpr=0.05>0",
                parameter_changes=[
                    ParameterChange(path="threshold", old_value=0.60, new_value=0.40)
                ],
                constraint_violations=["fpr=0.05>0"],
            )
        )
        ctx = mem.to_prompt_context()
        assert "Do NOT decrease threshold" in ctx
        assert "constraint_violated" in ctx

    def test_what_not_to_try_no_impact(self):
        mem = EvolutionMemory()
        mem.add_entry(
            self._make_entry(
                accepted=False,
                reason="not_improved",
                parameter_changes=[
                    ParameterChange(path="weight_benford", old_value=0.3, new_value=0.5)
                ],
            )
        )
        ctx = mem.to_prompt_context()
        assert "Do NOT increase weight_benford" in ctx
        assert "no impact" in ctx

    def test_to_dict_from_dict_round_trip(self):
        mem = EvolutionMemory()
        mem.add_entry(self._make_entry(iteration=1))
        mem.add_entry(self._make_entry(iteration=2, sensitivity_after=0.85))

        d = mem.to_dict()
        restored = EvolutionMemory.from_dict(d)
        assert len(restored.entries) == 2
        assert restored.best_iteration == mem.best_iteration
        assert restored.best_metrics == mem.best_metrics


class TestComputeParameterDiff:
    def test_flat_diff(self):
        before = {"threshold": 0.5, "weight": 0.7}
        after = {"threshold": 0.6, "weight": 0.7}
        changes = compute_parameter_diff(before, after)
        assert len(changes) == 1
        assert changes[0].path == "threshold"
        assert changes[0].old_value == 0.5
        assert changes[0].new_value == 0.6

    def test_nested_diff(self):
        before = {"ela": {"quality": 80}, "weight": 0.7}
        after = {"ela": {"quality": 85}, "weight": 0.7}
        changes = compute_parameter_diff(before, after)
        assert len(changes) == 1
        assert changes[0].path == "ela.quality"
        assert changes[0].old_value == 80
        assert changes[0].new_value == 85

    def test_no_changes(self):
        config = {"threshold": 0.5, "weight": 0.7}
        changes = compute_parameter_diff(config, dict(config))
        assert len(changes) == 0

    def test_added_key(self):
        before = {"threshold": 0.5}
        after = {"threshold": 0.5, "weight": 0.7}
        changes = compute_parameter_diff(before, after)
        assert len(changes) == 1
        assert changes[0].path == "weight"
        assert changes[0].old_value is None
        assert changes[0].new_value == 0.7

    def test_removed_key(self):
        before = {"threshold": 0.5, "weight": 0.7}
        after = {"threshold": 0.5}
        changes = compute_parameter_diff(before, after)
        assert len(changes) == 1
        assert changes[0].path == "weight"
        assert changes[0].old_value == 0.7
        assert changes[0].new_value is None

    def test_deeply_nested(self):
        before = {"a": {"b": {"c": 1}}}
        after = {"a": {"b": {"c": 2}}}
        changes = compute_parameter_diff(before, after)
        assert len(changes) == 1
        assert changes[0].path == "a.b.c"


class TestBestTrackingTargetMetric:
    def _make_entry(
        self,
        iteration: int = 1,
        accepted: bool = True,
        metrics_after: dict[str, float] | None = None,
    ) -> MemoryEntry:
        return MemoryEntry(
            iteration=iteration,
            timestamp="2025-01-01T00:00:00Z",
            files_changed=[],
            parameter_changes=[],
            metrics_before={"sensitivity": 0.8, "f1": 0.7, "fpr": 0.0},
            metrics_after=metrics_after or {"sensitivity": 0.82, "f1": 0.72, "fpr": 0.0},
            accepted=accepted,
            reason="improved" if accepted else "not_improved",
            constraint_violations=[],
        )

    def test_best_tracking_uses_target_metric(self):
        """f1-targeted memory tracks f1, not sensitivity."""
        mem = EvolutionMemory()
        mem.target_metric = "f1"
        mem.target_direction = "max"

        mem.add_entry(
            self._make_entry(
                iteration=1,
                metrics_after={"sensitivity": 0.9, "f1": 0.70, "fpr": 0.0},
            )
        )
        mem.add_entry(
            self._make_entry(
                iteration=2,
                metrics_after={"sensitivity": 0.82, "f1": 0.85, "fpr": 0.0},
            )
        )
        # Best should be iteration 2 because f1 is higher
        assert mem.best_iteration == 2
        assert mem.best_metrics["f1"] == 0.85

    def test_best_tracking_min_direction(self):
        """target_direction="min" tracks lowest."""
        mem = EvolutionMemory()
        mem.target_metric = "fpr"
        mem.target_direction = "min"

        mem.add_entry(
            self._make_entry(
                iteration=1,
                metrics_after={"sensitivity": 0.8, "fpr": 0.05},
            )
        )
        mem.add_entry(
            self._make_entry(
                iteration=2,
                metrics_after={"sensitivity": 0.8, "fpr": 0.02},
            )
        )
        # Best should be iteration 2 because fpr is lower
        assert mem.best_iteration == 2
        assert mem.best_metrics["fpr"] == 0.02

    def test_target_metric_serialized(self):
        """target_metric and target_direction survive round-trip."""
        mem = EvolutionMemory()
        mem.target_metric = "f1"
        mem.target_direction = "min"
        d = mem.to_dict()
        restored = EvolutionMemory.from_dict(d)
        assert restored.target_metric == "f1"
        assert restored.target_direction == "min"
