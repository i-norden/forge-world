"""Autonomous bench-modify-bench evolution loop.

Runs an iterative cycle: benchmark -> build context -> invoke agent ->
check constraints -> accept/reject -> repeat.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_world.core.agent_interface import build_evolution_context
from forge_world.core.runner import BenchmarkRunner, MultiBenchmarkReport
from forge_world.core.snapshots import SnapshotManager


@dataclass
class EvolutionConfig:
    agent_command: str  # e.g. "claude -p '{context_file}'"
    max_iterations: int = 10
    hard_constraints: list[dict[str, Any]] = field(
        default_factory=lambda: [{"metric": "fpr", "op": "<=", "value": 0}]
    )
    optimization_target: dict[str, Any] = field(
        default_factory=lambda: {"metric": "sensitivity", "direction": "max"}
    )
    convergence_patience: int = 3
    context_file: str = ".forge-world/evolution-context.md"


@dataclass
class IterationResult:
    iteration: int
    had_changes: bool
    constraint_violated: bool
    constraint_violations: list[str] = field(default_factory=list)
    improved: bool = False
    accepted: bool = False
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] | None = None
    rollback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "iteration": self.iteration,
            "had_changes": self.had_changes,
            "constraint_violated": self.constraint_violated,
            "constraint_violations": self.constraint_violations,
            "improved": self.improved,
            "accepted": self.accepted,
            "metrics_before": self.metrics_before,
        }
        if self.metrics_after is not None:
            d["metrics_after"] = self.metrics_after
        if self.rollback_reason is not None:
            d["rollback_reason"] = self.rollback_reason
        return d


@dataclass
class EvolutionResult:
    iterations: list[IterationResult] = field(default_factory=list)
    total_iterations: int = 0
    accepted_iterations: int = 0
    starting_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    convergence_reason: str = ""  # max_iterations | converged | constraint_violation | agent_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "accepted_iterations": self.accepted_iterations,
            "starting_metrics": self.starting_metrics,
            "final_metrics": self.final_metrics,
            "convergence_reason": self.convergence_reason,
            "iterations": [it.to_dict() for it in self.iterations],
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Evolution Result")
        lines.append("")
        lines.append(f"**Total iterations:** {self.total_iterations}")
        lines.append(f"**Accepted:** {self.accepted_iterations}")
        lines.append(f"**Convergence reason:** {self.convergence_reason}")
        lines.append("")

        lines.append("## Metrics")
        lines.append(f"- Starting: {_format_metrics(self.starting_metrics)}")
        lines.append(f"- Final: {_format_metrics(self.final_metrics)}")
        lines.append("")

        if self.iterations:
            lines.append("## Iteration Log")
            for it in self.iterations:
                status = "accepted" if it.accepted else "rejected"
                if it.constraint_violated:
                    status = "CONSTRAINT VIOLATED"
                elif not it.had_changes:
                    status = "no changes"
                lines.append(f"- **Iteration {it.iteration}**: {status}")
                if it.rollback_reason:
                    lines.append(f"  Reason: {it.rollback_reason}")
            lines.append("")

        return "\n".join(lines)


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for k, v in sorted(metrics.items()):
        parts.append(f"{k}={v:.4f}")
    return ", ".join(parts) if parts else "N/A"


def _extract_metrics(report) -> dict[str, float]:
    """Extract key metrics from a report."""
    if isinstance(report, MultiBenchmarkReport):
        am = report.aggregate_metrics
        return {
            "pass_rate": am.mean_pass_rate,
            "sensitivity": am.mean_sensitivity,
            "fpr": am.worst_case_fpr,
            "f1": am.mean_f1,
        }
    return {
        "pass_rate": report.pass_rate,
        "sensitivity": report.sensitivity,
        "fpr": report.fpr,
        "f1": report.f1,
    }


class EvolutionLoop:
    """Runs the bench-modify-bench evolution cycle."""

    def __init__(
        self,
        config: EvolutionConfig,
        runner: BenchmarkRunner,
        snapshot_manager: SnapshotManager,
        pipeline_config_schema: dict[str, Any] | None = None,
        diagnostics: Any | None = None,
        on_iteration_start: Any | None = None,
        on_iteration_complete: Any | None = None,
        run_kwargs: dict[str, Any] | None = None,
        baseline_name: str = "baseline",
    ):
        self.config = config
        self.runner = runner
        self.snapshot_manager = snapshot_manager
        self.pipeline_config_schema = pipeline_config_schema
        self.diagnostics = diagnostics
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete
        self.run_kwargs = run_kwargs or {}
        self.baseline_name = baseline_name

    def run(self) -> EvolutionResult:
        """Execute the evolution loop."""
        result = EvolutionResult()
        no_improvement_count = 0

        # Initial benchmark
        report = self._run_bench()
        metrics_before = _extract_metrics(report)
        result.starting_metrics = dict(metrics_before)

        for i in range(1, self.config.max_iterations + 1):
            if self.on_iteration_start:
                self.on_iteration_start(i)

            iteration = IterationResult(
                iteration=i,
                had_changes=False,
                constraint_violated=False,
                metrics_before=dict(metrics_before),
            )

            # Build and write context
            regression = None
            try:
                regression = self.snapshot_manager.check(report, baseline_name=self.baseline_name)
            except FileNotFoundError:
                pass

            diagnostics_data = None
            if self.diagnostics is not None:
                try:
                    if hasattr(report, "stable_reports") and report.stable_reports:
                        failing = [
                            r for r in report.stable_reports[0].report.item_results
                            if not r.passed
                        ]
                    elif hasattr(report, "item_results"):
                        failing = [r for r in report.item_results if not r.passed]
                    else:
                        failing = []
                    clusters = self.diagnostics.diagnose(failing)
                    diagnostics_data = [c.to_dict() for c in clusters]
                except Exception:
                    pass

            context = build_evolution_context(
                report=report,
                regression=regression,
                pipeline_config_schema=self.pipeline_config_schema,
                hard_constraints=self.config.hard_constraints,
                optimization_target=self.config.optimization_target,
                sample_size=self.run_kwargs.get("sample_size"),
                diagnostics=diagnostics_data,
            )

            self._write_context(context)

            # Invoke agent
            try:
                agent_ok = self._invoke_agent()
            except Exception:
                iteration.rollback_reason = "agent_error"
                result.iterations.append(iteration)
                result.convergence_reason = "agent_error"
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                break

            if not agent_ok:
                iteration.rollback_reason = "agent_error"
                result.iterations.append(iteration)
                result.convergence_reason = "agent_error"
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                break

            # Check for changes
            has_changes = self._check_for_changes()
            iteration.had_changes = has_changes

            if not has_changes:
                no_improvement_count += 1
                iteration.rollback_reason = "no_changes"
                result.iterations.append(iteration)
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                if no_improvement_count >= self.config.convergence_patience:
                    result.convergence_reason = "converged"
                    break
                continue

            # Re-run benchmark
            new_report = self._run_bench()
            metrics_after = _extract_metrics(new_report)
            iteration.metrics_after = metrics_after

            # Check constraints
            violations = self._check_constraints(metrics_after)
            if violations:
                iteration.constraint_violated = True
                iteration.constraint_violations = violations
                iteration.rollback_reason = f"constraint_violated: {', '.join(violations)}"
                self._rollback()
                result.iterations.append(iteration)
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                # Don't break on constraint violation — agent might fix it next iteration
                no_improvement_count += 1
                if no_improvement_count >= self.config.convergence_patience:
                    result.convergence_reason = "converged"
                    break
                continue

            # Check improvement
            improved = self._is_improved(metrics_before, metrics_after)
            iteration.improved = improved

            if improved:
                iteration.accepted = True
                self._commit_changes(i, metrics_before, metrics_after)
                metrics_before = metrics_after
                report = new_report
                no_improvement_count = 0
            else:
                iteration.rollback_reason = "not_improved"
                self._rollback()
                no_improvement_count += 1

            result.iterations.append(iteration)
            if self.on_iteration_complete:
                self.on_iteration_complete(i, iteration)

            if no_improvement_count >= self.config.convergence_patience:
                result.convergence_reason = "converged"
                break
        else:
            result.convergence_reason = "max_iterations"

        result.total_iterations = len(result.iterations)
        result.accepted_iterations = sum(1 for it in result.iterations if it.accepted)
        result.final_metrics = dict(metrics_before)
        return result

    def _run_bench(self):
        """Run benchmark using configured strategy."""
        kwargs = dict(self.run_kwargs)
        seed = kwargs.pop("seed", None)
        seed_strategy = kwargs.pop("seed_strategy", None)
        sample_size = kwargs.pop("sample_size", None)
        tier = kwargs.pop("tier", None)

        if seed is not None:
            return self.runner.run(
                seed=seed, sample_size=sample_size, tier=tier,
            )
        elif seed_strategy is not None:
            return self.runner.run_multi(
                seed_strategy, sample_size=sample_size, tier=tier,
            )
        else:
            return self.runner.run(sample_size=sample_size, tier=tier)

    def _write_context(self, context) -> None:
        """Write evolution context to the context file."""
        path = Path(self.config.context_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context.to_prompt_context())

    def _invoke_agent(self) -> bool:
        """Invoke the agent subprocess. Returns True if successful."""
        cmd = self.config.agent_command.replace(
            "{context_file}", self.config.context_file
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0

    def _check_for_changes(self) -> bool:
        """Check if git working tree has changes."""
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())

    def _check_constraints(self, metrics: dict[str, float]) -> list[str]:
        """Evaluate hard constraints against metrics. Returns list of violations."""
        violations = []
        for constraint in self.config.hard_constraints:
            metric = constraint.get("metric", "")
            op = constraint.get("op", "<=")
            value = float(constraint.get("value", 0))
            actual = metrics.get(metric)
            if actual is None:
                continue
            if op == "<=" and actual > value:
                violations.append(f"{metric}={actual:.4f} > {value}")
            elif op == ">=" and actual < value:
                violations.append(f"{metric}={actual:.4f} < {value}")
            elif op == "<" and actual >= value:
                violations.append(f"{metric}={actual:.4f} >= {value}")
            elif op == ">" and actual <= value:
                violations.append(f"{metric}={actual:.4f} <= {value}")
            elif op == "==" and abs(actual - value) > 1e-9:
                violations.append(f"{metric}={actual:.4f} != {value}")
        return violations

    def _is_improved(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> bool:
        """Check if the optimization target improved."""
        target = self.config.optimization_target
        metric = target.get("metric", "sensitivity")
        direction = target.get("direction", "max")
        before_val = before.get(metric, 0)
        after_val = after.get(metric, 0)
        if direction == "max":
            return after_val > before_val
        return after_val < before_val

    def _rollback(self) -> None:
        """Discard uncommitted changes."""
        subprocess.run(
            ["git", "checkout", "--", "."],
            capture_output=True, timeout=10,
        )

    def _commit_changes(
        self,
        iteration: int,
        before: dict[str, float],
        after: dict[str, float],
    ) -> None:
        """Stage and commit changes."""
        target = self.config.optimization_target.get("metric", "sensitivity")
        before_val = before.get(target, 0)
        after_val = after.get(target, 0)
        msg = (
            f"forge evolve: iteration {iteration} "
            f"({target}: {before_val:.4f} -> {after_val:.4f})"
        )
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, timeout=30,
        )
