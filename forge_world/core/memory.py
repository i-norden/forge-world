"""Iteration memory for the evolution loop.

Tracks parameter changes, outcomes, and constraints across iterations.
Provides "What NOT to Try Again" guidance to prevent the agent from
repeating failed experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParameterChange:
    """A single parameter that was modified between iterations."""

    path: str  # JSON path, e.g. "ela.quality" or "weight_ela"
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterChange:
        return cls(
            path=data["path"],
            old_value=data["old_value"],
            new_value=data["new_value"],
        )


@dataclass
class MemoryEntry:
    """Record of a single evolution iteration."""

    iteration: int
    timestamp: str
    files_changed: list[str]  # from git diff --name-only
    parameter_changes: list[ParameterChange]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float] | None
    accepted: bool
    reason: str  # "improved", "not_improved", "constraint_violated:fpr=0.05>0",
    # "no_changes", "agent_error", "exploration", "pareto_improved"
    constraint_violations: list[str]
    agent_reasoning: str = ""  # from proposal file's agent_notes, or ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "files_changed": self.files_changed,
            "parameter_changes": [pc.to_dict() for pc in self.parameter_changes],
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "accepted": self.accepted,
            "reason": self.reason,
            "constraint_violations": self.constraint_violations,
            "agent_reasoning": self.agent_reasoning,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            iteration=data["iteration"],
            timestamp=data["timestamp"],
            files_changed=data.get("files_changed", []),
            parameter_changes=[
                ParameterChange.from_dict(pc)
                for pc in data.get("parameter_changes", [])
            ],
            metrics_before=data.get("metrics_before", {}),
            metrics_after=data.get("metrics_after"),
            accepted=data["accepted"],
            reason=data.get("reason", ""),
            constraint_violations=data.get("constraint_violations", []),
            agent_reasoning=data.get("agent_reasoning", ""),
        )


@dataclass
class EvolutionMemory:
    """Persistent log of all iterations in an evolution run."""

    entries: list[MemoryEntry] = field(default_factory=list)
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_iteration: int = 0

    def add_entry(self, entry: MemoryEntry) -> None:
        """Append entry, update best_metrics/best_iteration if accepted and improved."""
        self.entries.append(entry)
        if entry.accepted and entry.metrics_after is not None:
            if not self.best_metrics:
                self.best_metrics = dict(entry.metrics_after)
                self.best_iteration = entry.iteration
            else:
                # Update best if sensitivity improved (or first accepted)
                current_best = self.best_metrics.get("sensitivity", 0)
                new_val = entry.metrics_after.get("sensitivity", 0)
                if new_val >= current_best:
                    self.best_metrics = dict(entry.metrics_after)
                    self.best_iteration = entry.iteration

    def save(self, path: Path) -> None:
        """Persist to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> EvolutionMemory:
        """Load from disk, or return empty if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "best_metrics": self.best_metrics,
            "best_iteration": self.best_iteration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionMemory:
        mem = cls(
            entries=[MemoryEntry.from_dict(e) for e in data.get("entries", [])],
            best_metrics=data.get("best_metrics", {}),
            best_iteration=data.get("best_iteration", 0),
        )
        return mem

    def to_prompt_context(self) -> str:
        """Compact markdown for agent consumption."""
        if not self.entries:
            return ""

        lines: list[str] = []
        lines.append("## Iteration History")
        lines.append("")
        lines.append("| # | Outcome | Key Metrics | Key Changes |")
        lines.append("|---|---------|-------------|-------------|")

        for entry in self.entries:
            outcome = "ACCEPTED" if entry.accepted else "REJECTED"
            if entry.reason.startswith("constraint_violated"):
                outcome = f"REJECTED ({entry.reason})"
            elif entry.reason == "no_changes":
                outcome = "NO CHANGES"
            elif entry.reason == "exploration":
                outcome = "ACCEPTED (exploration)"
            elif entry.reason == "pareto_improved":
                outcome = "ACCEPTED (pareto)"

            # Build metrics summary
            metrics_parts = []
            if entry.metrics_before and entry.metrics_after:
                for key in ("sensitivity", "fpr"):
                    before_val = entry.metrics_before.get(key)
                    after_val = entry.metrics_after.get(key)
                    if before_val is not None and after_val is not None:
                        metrics_parts.append(f"{key}: {before_val:.4f}→{after_val:.4f}")
            metrics_str = ", ".join(metrics_parts) if metrics_parts else "—"

            # Build changes summary
            if entry.parameter_changes:
                changes = []
                for pc in entry.parameter_changes[:3]:  # Limit to 3
                    changes.append(f"{pc.path}: {pc.old_value}→{pc.new_value}")
                changes_str = "; ".join(changes)
                if len(entry.parameter_changes) > 3:
                    changes_str += f" (+{len(entry.parameter_changes) - 3} more)"
            elif entry.files_changed:
                changes_str = ", ".join(entry.files_changed[:3])
                if len(entry.files_changed) > 3:
                    changes_str += f" (+{len(entry.files_changed) - 3} more)"
            else:
                changes_str = "—"

            lines.append(f"| {entry.iteration} | {outcome} | {metrics_str} | {changes_str} |")

        lines.append("")

        # What NOT to Try Again
        warnings = _build_warnings(self.entries)
        if warnings:
            lines.append("## What NOT to Try Again")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Best so far
        if self.best_metrics:
            lines.append("## Best So Far")
            parts = [f"{k}={v:.4f}" for k, v in sorted(self.best_metrics.items())]
            lines.append(f"Iteration {self.best_iteration}: {', '.join(parts)}")
            lines.append("")

        return "\n".join(lines)


def _build_warnings(entries: list[MemoryEntry]) -> list[str]:
    """Build "What NOT to Try Again" warnings from rejected entries."""
    warnings: list[str] = []
    for entry in entries:
        if entry.accepted:
            continue
        if not entry.parameter_changes:
            continue

        for pc in entry.parameter_changes:
            if entry.reason.startswith("constraint_violated"):
                # Warn about the specific parameter/direction combo
                direction = "increase" if _is_increase(pc.old_value, pc.new_value) else "decrease"
                warnings.append(
                    f"Do NOT {direction} {pc.path} beyond {pc.new_value} "
                    f"(causes {entry.reason}, iteration {entry.iteration})"
                )
            elif entry.reason == "not_improved":
                direction = "increase" if _is_increase(pc.old_value, pc.new_value) else "decrease"
                warnings.append(
                    f"Do NOT {direction} {pc.path} from {pc.old_value} to {pc.new_value} "
                    f"(no impact, iteration {entry.iteration})"
                )

    return warnings


def _is_increase(old: Any, new: Any) -> bool:
    """Check if a value change represents an increase."""
    try:
        return float(new) > float(old)
    except (TypeError, ValueError):
        return False


def compute_parameter_diff(
    before: dict[str, Any], after: dict[str, Any], prefix: str = ""
) -> list[ParameterChange]:
    """Recursively diff two config dicts. Returns flat list of changes with dotted paths.

    Example: {"ela": {"quality": 80}} vs {"ela": {"quality": 85}}
    → [ParameterChange(path="ela.quality", old_value=80, new_value=85)]
    """
    changes: list[ParameterChange] = []
    all_keys = set(list(before.keys()) + list(after.keys()))

    for key in sorted(all_keys):
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        old_val = before.get(key)
        new_val = after.get(key)

        if old_val == new_val:
            continue

        # Recurse into nested dicts
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.extend(compute_parameter_diff(old_val, new_val, path))
        else:
            changes.append(ParameterChange(path=path, old_value=old_val, new_value=new_val))

    return changes
