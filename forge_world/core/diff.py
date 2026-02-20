"""Compact diff between current benchmark run and baseline.

Computes deltas beyond simple pass/fail flips: risk level shifts,
boundary proximity changes, and multi-seed instability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge_world.core.metrics import find_near_misses
from forge_world.core.runner import BenchmarkReport, MultiBenchmarkReport
from forge_world.core.snapshots import RegressionReport, Snapshot


_SEVERITY_SCORES = {"clean": 0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}


@dataclass
class DiffSummary:
    baseline_pass_rate: str
    current_pass_rate: str
    net_change: int  # positive = improved

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_pass_rate": self.baseline_pass_rate,
            "current_pass_rate": self.current_pass_rate,
            "net_change": self.net_change,
        }


@dataclass
class RiskShift:
    """Item whose risk changed but pass/fail didn't flip."""

    item_id: str
    category: str
    baseline_risk: str
    current_risk: str
    direction: str  # "up" or "down"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "baseline_risk": self.baseline_risk,
            "current_risk": self.current_risk,
            "direction": self.direction,
        }


@dataclass
class BoundaryItem:
    """Item that moved closer/farther from the pass/fail boundary."""

    item_id: str
    category: str
    passed: bool
    distance_to_boundary: float
    direction: str  # "approaching" or "receding"
    baseline_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "passed": self.passed,
            "distance_to_boundary": round(self.distance_to_boundary, 4),
            "direction": self.direction,
            "baseline_distance": round(self.baseline_distance, 4),
        }


@dataclass
class DiffReport:
    summary: DiffSummary
    new_failures: list[dict[str, Any]] = field(default_factory=list)
    new_passes: list[dict[str, Any]] = field(default_factory=list)
    risk_shifts: list[RiskShift] = field(default_factory=list)
    boundary_items: list[BoundaryItem] = field(default_factory=list)
    unstable_items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "new_failures": [r for r in self.new_failures],
            "new_passes": [i for i in self.new_passes],
            "risk_shifts": [rs.to_dict() for rs in self.risk_shifts],
            "boundary_items": [bi.to_dict() for bi in self.boundary_items],
            "unstable_items": self.unstable_items,
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Diff: current vs baseline")
        lines.append("")

        # Summary
        lines.append("## Summary")
        s = self.summary
        sign = "+" if s.net_change >= 0 else ""
        lines.append(
            f"Baseline: {s.baseline_pass_rate}, "
            f"Current: {s.current_pass_rate} "
            f"({sign}{s.net_change} net)"
        )
        lines.append("")

        # New failures
        if self.new_failures:
            lines.append(f"## New Failures ({len(self.new_failures)})")
            for item in self.new_failures:
                lines.append(
                    f"- `{item['item_id']}` ({item['category']}): "
                    f"risk {item['baseline_risk']} -> {item['current_risk']}"
                )
            lines.append("")

        # New passes
        if self.new_passes:
            lines.append(f"## New Passes ({len(self.new_passes)})")
            for item in self.new_passes:
                lines.append(
                    f"- `{item['item_id']}` ({item['category']}): "
                    f"risk {item['baseline_risk']} -> {item['current_risk']}"
                )
            lines.append("")

        # Risk shifts
        if self.risk_shifts:
            lines.append(f"## Risk Shifts ({len(self.risk_shifts)}) [pass/fail unchanged]")
            for rs in self.risk_shifts:
                lines.append(
                    f"- `{rs.item_id}`: {rs.baseline_risk} -> {rs.current_risk} "
                    f"({rs.direction.upper()})"
                )
            lines.append("")

        # Boundary items
        if self.boundary_items:
            lines.append(f"## Boundary Movement ({len(self.boundary_items)})")
            for bi in self.boundary_items:
                status = "PASSING" if bi.passed else "FAILING"
                lines.append(
                    f"- `{bi.item_id}` [{status}]: "
                    f"distance {bi.baseline_distance:.3f} -> {bi.distance_to_boundary:.3f} "
                    f"({bi.direction})"
                )
            lines.append("")

        # Unstable items
        if self.unstable_items:
            lines.append(f"## Unstable Items ({len(self.unstable_items)})")
            for item in self.unstable_items:
                lines.append(f"- `{item['item_id']}`: stability={item.get('stability', '?')}")
            lines.append("")

        if (
            not self.new_failures
            and not self.new_passes
            and not self.risk_shifts
            and not self.boundary_items
            and not self.unstable_items
        ):
            lines.append("No changes detected.")
            lines.append("")

        return "\n".join(lines)


def compute_diff(
    report: BenchmarkReport | MultiBenchmarkReport,
    snapshot: Snapshot,
    regression: RegressionReport,
) -> DiffReport:
    """Compute a detailed diff between the current report and baseline snapshot."""
    # Extract current results
    if isinstance(report, MultiBenchmarkReport):
        stable = report.stable_reports
        primary = stable[0].report if stable else report.seed_reports[0].report
    else:
        primary = report

    # Summary
    baseline_pr = regression.baseline_pass_rate or "?"
    current_pr = regression.current_pass_rate or "?"
    net_change = regression.items_improved - regression.items_regressed

    summary = DiffSummary(
        baseline_pass_rate=baseline_pr,
        current_pass_rate=current_pr,
        net_change=net_change,
    )

    # New failures/passes from regression report
    new_failures = [r.to_dict() for r in regression.regressions]
    new_passes = [i.to_dict() for i in regression.improvements]

    # Risk shifts: items where risk changed but pass/fail didn't flip
    risk_shifts: list[RiskShift] = []
    for result in primary.item_results:
        baseline_outcome = snapshot.item_outcomes.get(result.item_id)
        if baseline_outcome is None:
            continue
        baseline_passed = baseline_outcome["passed"]
        current_passed = result.passed
        # Only interested in items that didn't flip
        if baseline_passed != current_passed:
            continue
        baseline_risk = baseline_outcome["risk_level"]
        current_risk = result.risk_level
        if baseline_risk != current_risk:
            baseline_score = _SEVERITY_SCORES.get(baseline_risk, 0)
            current_score = _SEVERITY_SCORES.get(current_risk, 0)
            direction = "up" if current_score > baseline_score else "down"
            risk_shifts.append(
                RiskShift(
                    item_id=result.item_id,
                    category=result.category,
                    baseline_risk=baseline_risk,
                    current_risk=current_risk,
                    direction=direction,
                )
            )

    # Boundary items: compare current near-misses against baseline distances
    result_dicts = [r.to_dict() for r in primary.item_results]
    current_near_misses = find_near_misses(result_dicts)

    # Compute baseline near-miss distances for comparison
    baseline_distances = _compute_baseline_distances(snapshot.item_outcomes)

    boundary_items: list[BoundaryItem] = []
    for nm in current_near_misses:
        if nm.item_id in baseline_distances:
            baseline_dist = baseline_distances[nm.item_id]
            current_dist = nm.distance_to_boundary
            if abs(baseline_dist - current_dist) > 0.01:
                direction = "approaching" if current_dist < baseline_dist else "receding"
                boundary_items.append(
                    BoundaryItem(
                        item_id=nm.item_id,
                        category=nm.category,
                        passed=nm.passed,
                        distance_to_boundary=current_dist,
                        direction=direction,
                        baseline_distance=baseline_dist,
                    )
                )

    # Unstable items (from regression report, multi-seed only)
    unstable_items = regression.unstable_items

    return DiffReport(
        summary=summary,
        new_failures=new_failures,
        new_passes=new_passes,
        risk_shifts=risk_shifts,
        boundary_items=boundary_items,
        unstable_items=unstable_items,
    )


def _compute_baseline_distances(
    item_outcomes: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Compute distance-to-boundary for baseline items."""
    distances: dict[str, float] = {}
    for item_id, outcome in item_outcomes.items():
        risk_level = outcome.get("risk_level", "clean")
        expected = outcome.get("expected_label", "findings")
        expects_findings = expected not in ("clean", "informational")

        if expects_findings:
            score = _SEVERITY_SCORES.get(risk_level, 0)
            distance = abs(score - 0.5)  # medium boundary
        else:
            score = _SEVERITY_SCORES.get(risk_level, 0)
            distance = abs(score - 0.75)  # high boundary
        distances[item_id] = distance
    return distances
