"""Snapshot-based regression detection.

Locks exact per-item outputs as JSON files (git-committable) and detects
regressions by comparing current runs against a locked baseline.

Supports multi-seed snapshots: per-stable-seed item outcomes and exploration
baseline tracking for generalization regression detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_world.core.runner import BenchmarkReport, MultiBenchmarkReport


@dataclass
class RegressionItem:
    """A single item that changed between baseline and current run."""

    item_id: str
    category: str
    expected_label: str
    baseline_passed: bool
    current_passed: bool
    baseline_risk: str
    current_risk: str
    baseline_confidence: float
    current_confidence: float
    seed: int | None = None  # Which seed this regression was observed on

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "item_id": self.item_id,
            "category": self.category,
            "expected_label": self.expected_label,
            "baseline_passed": self.baseline_passed,
            "current_passed": self.current_passed,
            "baseline_risk": self.baseline_risk,
            "current_risk": self.current_risk,
            "baseline_confidence": round(self.baseline_confidence, 4),
            "current_confidence": round(self.current_confidence, 4),
        }
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass
class RegressionReport:
    """Comparison between current run and locked baseline."""

    baseline_name: str
    items_regressed: int = 0
    items_improved: int = 0
    new_false_positives: int = 0
    regressions: list[RegressionItem] = field(default_factory=list)
    improvements: list[RegressionItem] = field(default_factory=list)
    unstable_items: list[dict[str, Any]] = field(default_factory=list)
    baseline_pass_rate: str = ""
    current_pass_rate: str = ""
    exploration_regressed: bool = False
    exploration_details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        return self.items_regressed > 0

    @property
    def has_new_false_positives(self) -> bool:
        return self.new_false_positives > 0

    @property
    def has_exploration_regression(self) -> bool:
        return self.exploration_regressed

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "items_regressed": self.items_regressed,
            "items_improved": self.items_improved,
            "new_false_positives": self.new_false_positives,
            "baseline_pass_rate": self.baseline_pass_rate,
            "current_pass_rate": self.current_pass_rate,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "unstable_items": self.unstable_items,
            "exploration_regressed": self.exploration_regressed,
            "exploration_details": self.exploration_details,
        }


@dataclass
class Snapshot:
    """A locked baseline snapshot.

    For single-seed snapshots, ``item_outcomes`` contains the per-item results.
    For multi-seed snapshots, ``seed_outcomes`` maps seed -> item_id -> outcome
    and ``exploration_baseline`` tracks the running average of exploration
    pass rates for generalization regression detection.
    """

    name: str
    timestamp: str
    git_sha: str | None
    config_hash: str
    item_outcomes: dict[str, dict[str, Any]]  # backward compat (single-seed)
    summary: dict[str, Any]
    # Multi-seed fields
    seed_outcomes: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    exploration_metrics: dict[str, float] = field(default_factory=dict)
    exploration_baseline: float = 0.0
    exploration_count: int = 0

    @property
    def is_multi_seed(self) -> bool:
        return bool(self.seed_outcomes)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "item_outcomes": self.item_outcomes,
            "summary": self.summary,
        }
        if self.seed_outcomes:
            d["seed_outcomes"] = self.seed_outcomes
        if self.exploration_metrics:
            d["exploration_metrics"] = self.exploration_metrics
        if self.exploration_baseline > 0 or self.exploration_count > 0:
            d["exploration_baseline"] = self.exploration_baseline
            d["exploration_count"] = self.exploration_count
        return d


class SnapshotManager:
    """Manages baseline snapshots as JSON files."""

    def __init__(self, snapshots_dir: str | Path = ".forge-world/snapshots"):
        self.snapshots_dir = Path(snapshots_dir)

    def _snapshot_path(self, name: str) -> Path:
        return self.snapshots_dir / f"{name}.json"

    def lock(
        self,
        report: BenchmarkReport | MultiBenchmarkReport,
        name: str = "baseline",
    ) -> Snapshot:
        """Lock current benchmark outputs as a named snapshot.

        Accepts either a single-seed ``BenchmarkReport`` or a multi-seed
        ``MultiBenchmarkReport``.  For multi-seed, stores per-stable-seed
        item outcomes and updates the exploration baseline.
        """
        if isinstance(report, MultiBenchmarkReport):
            return self._lock_multi(report, name)
        return self._lock_single(report, name)

    def _lock_single(self, report: BenchmarkReport, name: str) -> Snapshot:
        """Lock a single-seed benchmark report."""
        item_outcomes = self._extract_item_outcomes(report)

        snapshot = Snapshot(
            name=name,
            timestamp=report.timestamp,
            git_sha=report.git_sha,
            config_hash=report.config_hash,
            item_outcomes=item_outcomes,
            summary=report.summary(),
        )

        self._save(snapshot, name)
        return snapshot

    def _lock_multi(self, report: MultiBenchmarkReport, name: str) -> Snapshot:
        """Lock a multi-seed benchmark report.

        Stores per-stable-seed item outcomes for deterministic regression
        detection.  Updates the exploration baseline running average.
        """
        # Use the first stable seed's results for backward-compat item_outcomes
        stable = report.stable_reports
        first_report = stable[0].report if stable else report.seed_reports[0].report
        item_outcomes = self._extract_item_outcomes(first_report)

        # Per-stable-seed outcomes
        seed_outcomes: dict[str, dict[str, dict[str, Any]]] = {}
        for sr in stable:
            seed_outcomes[str(sr.seed)] = self._extract_item_outcomes(sr.report)

        # Exploration metrics from this run
        exploration = report.exploration_reports
        exploration_metrics: dict[str, float] = {}
        if exploration:
            exp_pass_rates = [sr.report.pass_rate for sr in exploration]
            exp_sensitivities = [sr.report.sensitivity for sr in exploration]
            exp_fprs = [sr.report.fpr for sr in exploration]
            exploration_metrics = {
                "mean_pass_rate": sum(exp_pass_rates) / len(exp_pass_rates),
                "mean_sensitivity": sum(exp_sensitivities) / len(exp_sensitivities),
                "worst_fpr": max(exp_fprs),
            }

        # Update exploration baseline (running average across locks)
        existing = self.load(name)
        if existing and existing.exploration_count > 0 and exploration_metrics:
            old_baseline = existing.exploration_baseline
            old_count = existing.exploration_count
            new_value = exploration_metrics["mean_pass_rate"]
            exploration_baseline = (old_baseline * old_count + new_value) / (old_count + 1)
            exploration_count = old_count + 1
        elif exploration_metrics:
            exploration_baseline = exploration_metrics["mean_pass_rate"]
            exploration_count = 1
        else:
            exploration_baseline = 0.0
            exploration_count = 0

        snapshot = Snapshot(
            name=name,
            timestamp=report.timestamp,
            git_sha=report.git_sha,
            config_hash=report.config_hash,
            item_outcomes=item_outcomes,
            summary=report.summary(),
            seed_outcomes=seed_outcomes,
            exploration_metrics=exploration_metrics,
            exploration_baseline=exploration_baseline,
            exploration_count=exploration_count,
        )

        self._save(snapshot, name)
        return snapshot

    @staticmethod
    def _extract_item_outcomes(report: BenchmarkReport) -> dict[str, dict[str, Any]]:
        outcomes: dict[str, dict[str, Any]] = {}
        for result in report.item_results:
            outcomes[result.item_id] = {
                "passed": result.passed,
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "category": result.category,
                "expected_label": result.expected_label,
                "methods_flagged": result.methods_flagged,
                "converging_evidence": result.converging_evidence,
            }
        return outcomes

    def _save(self, snapshot: Snapshot, name: str) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        with open(self._snapshot_path(name), "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    def load(self, name: str = "baseline") -> Snapshot | None:
        """Load a named snapshot, or None if it doesn't exist."""
        path = self._snapshot_path(name)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return Snapshot(
            name=data["name"],
            timestamp=data["timestamp"],
            git_sha=data.get("git_sha"),
            config_hash=data["config_hash"],
            item_outcomes=data["item_outcomes"],
            summary=data.get("summary", {}),
            seed_outcomes=data.get("seed_outcomes", {}),
            exploration_metrics=data.get("exploration_metrics", {}),
            exploration_baseline=data.get("exploration_baseline", 0.0),
            exploration_count=data.get("exploration_count", 0),
        )

    def get_failed_item_ids(self, name: str = "baseline") -> set[str]:
        """Item IDs that failed in a snapshot.

        For multi-seed snapshots, returns items that failed on ANY seed.
        """
        snapshot = self.load(name)
        if snapshot is None:
            raise FileNotFoundError(f"No snapshot '{name}' found. Run 'forge lock' first.")

        failed: set[str] = set()

        if snapshot.is_multi_seed:
            # Items failed on ANY seed
            for _seed, outcomes in snapshot.seed_outcomes.items():
                for item_id, outcome in outcomes.items():
                    if not outcome.get("passed", True):
                        failed.add(item_id)
        else:
            for item_id, outcome in snapshot.item_outcomes.items():
                if not outcome.get("passed", True):
                    failed.add(item_id)

        return failed

    def list_snapshots(self) -> list[str]:
        """List available snapshot names."""
        if not self.snapshots_dir.exists():
            return []
        return sorted(p.stem for p in self.snapshots_dir.glob("*.json"))

    def check(
        self,
        report: BenchmarkReport | MultiBenchmarkReport,
        baseline_name: str = "baseline",
    ) -> RegressionReport:
        """Compare current report against locked baseline.

        For single-seed reports, compares item-by-item (legacy behavior).
        For multi-seed reports, performs per-stable-seed item comparison
        plus exploration regression detection.
        """
        baseline = self.load(baseline_name)
        if baseline is None:
            raise FileNotFoundError(
                f"No baseline snapshot '{baseline_name}' found. "
                f"Run 'forge lock' first to create one."
            )

        if isinstance(report, MultiBenchmarkReport):
            return self._check_multi(report, baseline)
        return self._check_single(report, baseline)

    def _check_single(self, report: BenchmarkReport, baseline: Snapshot) -> RegressionReport:
        """Compare a single-seed report against baseline (legacy path)."""
        regressions, improvements, new_fps = self._compare_items(
            report.item_results, baseline.item_outcomes
        )

        return RegressionReport(
            baseline_name=baseline.name,
            items_regressed=len(regressions),
            items_improved=len(improvements),
            new_false_positives=new_fps,
            regressions=regressions,
            improvements=improvements,
            baseline_pass_rate=baseline.summary.get("pass_rate", ""),
            current_pass_rate=f"{report.pass_count}/{report.total_count}",
        )

    def _check_multi(self, report: MultiBenchmarkReport, baseline: Snapshot) -> RegressionReport:
        """Compare a multi-seed report against baseline.

        Stable seeds: per-seed item-level regression check.
        Exploration seeds: aggregate metric regression check against baseline.
        """
        all_regressions: list[RegressionItem] = []
        all_improvements: list[RegressionItem] = []
        total_new_fps = 0

        # Per-stable-seed comparison
        for sr in report.stable_reports:
            seed_str = str(sr.seed)
            # Use per-seed baseline if available, fall back to item_outcomes
            if baseline.is_multi_seed and seed_str in baseline.seed_outcomes:
                baseline_outcomes = baseline.seed_outcomes[seed_str]
            else:
                baseline_outcomes = baseline.item_outcomes

            regs, imps, fps = self._compare_items(
                sr.report.item_results, baseline_outcomes, seed=sr.seed
            )
            all_regressions.extend(regs)
            all_improvements.extend(imps)
            total_new_fps += fps

        # Exploration regression check
        exploration_regressed = False
        exploration_details: dict[str, Any] = {}

        exploration = report.exploration_reports
        if exploration and baseline.exploration_count > 0:
            exp_pass_rates = [sr.report.pass_rate for sr in exploration]
            current_exploration_pr = sum(exp_pass_rates) / len(exp_pass_rates)

            # Threshold = avg(baseline_stable_pass_rate, exploration_baseline)
            baseline_stable_pr = _extract_stable_pass_rate(baseline)
            threshold = (baseline_stable_pr + baseline.exploration_baseline) / 2

            exploration_regressed = current_exploration_pr < threshold

            exploration_details = {
                "current_exploration_pass_rate": round(current_exploration_pr, 4),
                "baseline_stable_pass_rate": round(baseline_stable_pr, 4),
                "exploration_baseline": round(baseline.exploration_baseline, 4),
                "threshold": round(threshold, 4),
                "regressed": exploration_regressed,
            }

        # Unstable items: items that pass on some seeds but not others
        item_stability = report.aggregate_metrics.item_stability
        unstable_items = [
            {"item_id": iid, "stability": round(stab, 4)}
            for iid, stab in sorted(item_stability.items())
            if 0 < stab < 1
        ]

        # Current pass rate summary
        am = report.aggregate_metrics
        current_pr = (
            f"{am.mean_pass_rate:.1%} (range: {am.min_pass_rate:.1%}-{am.max_pass_rate:.1%})"
        )

        return RegressionReport(
            baseline_name=baseline.name,
            items_regressed=len(all_regressions),
            items_improved=len(all_improvements),
            new_false_positives=total_new_fps,
            regressions=all_regressions,
            improvements=all_improvements,
            unstable_items=unstable_items,
            baseline_pass_rate=baseline.summary.get("pass_rate", "")
            or baseline.summary.get("mean_pass_rate", ""),
            current_pass_rate=current_pr,
            exploration_regressed=exploration_regressed,
            exploration_details=exploration_details,
        )

    @staticmethod
    def _compare_items(
        current_results: list,
        baseline_outcomes: dict[str, dict[str, Any]],
        seed: int | None = None,
    ) -> tuple[list[RegressionItem], list[RegressionItem], int]:
        """Compare item results against baseline outcomes.

        Returns (regressions, improvements, new_false_positives).
        """
        regressions: list[RegressionItem] = []
        improvements: list[RegressionItem] = []
        new_fps = 0

        for result in current_results:
            baseline_item = baseline_outcomes.get(result.item_id)
            if baseline_item is None:
                continue

            baseline_passed = baseline_item["passed"]
            current_passed = result.passed

            if baseline_passed and not current_passed:
                item = RegressionItem(
                    item_id=result.item_id,
                    category=result.category,
                    expected_label=result.expected_label,
                    baseline_passed=True,
                    current_passed=False,
                    baseline_risk=baseline_item["risk_level"],
                    current_risk=result.risk_level,
                    baseline_confidence=baseline_item["confidence"],
                    current_confidence=result.confidence,
                    seed=seed,
                )
                regressions.append(item)
                if result.expected_label in ("clean", "informational"):
                    new_fps += 1

            elif not baseline_passed and current_passed:
                improvements.append(
                    RegressionItem(
                        item_id=result.item_id,
                        category=result.category,
                        expected_label=result.expected_label,
                        baseline_passed=False,
                        current_passed=True,
                        baseline_risk=baseline_item["risk_level"],
                        current_risk=result.risk_level,
                        baseline_confidence=baseline_item["confidence"],
                        current_confidence=result.confidence,
                        seed=seed,
                    )
                )

        return regressions, improvements, new_fps


def _extract_stable_pass_rate(baseline: Snapshot) -> float:
    """Extract the stable-seed pass rate from a snapshot's summary."""
    summary = baseline.summary
    # Multi-seed summary stores mean_pass_rate as a float
    if "mean_pass_rate" in summary and isinstance(summary["mean_pass_rate"], (int, float)):
        return float(summary["mean_pass_rate"])
    # Single-seed stores pass_rate as "N/M" string
    pr = summary.get("pass_rate", "")
    if isinstance(pr, str) and "/" in pr:
        num, denom = pr.split("/")
        return int(num) / int(denom) if int(denom) > 0 else 0.0
    return 0.0
