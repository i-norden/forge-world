"""Autonomous bench-modify-bench evolution loop.

Runs an iterative cycle: benchmark -> build context -> invoke agent ->
check constraints -> accept/reject -> repeat.

Includes iteration memory, parameter proposals, change decomposition,
smart convergence, and exploration budget.
"""

from __future__ import annotations

import logging
import math
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_world.core.agent_interface import build_evolution_context
from forge_world.core.memory import EvolutionMemory, MemoryEntry, compute_parameter_diff
from forge_world.core.proposals import (
    ProposalFile,
    apply_proposals,
    rollback_proposals,
    validate_proposals,
)
from forge_world.core.runner import BenchmarkRunner, MultiBenchmarkReport
from forge_world.core.sensitivity import (
    SensitivityReport,
    compute_sensitivity,
)
from forge_world.core.snapshots import SnapshotManager


@dataclass
class EvolutionConfig:
    agent_command: str  # e.g. "claude -p '{context_file}'"
    max_iterations: int = 10
    hard_constraints: list[dict[str, Any]] = field(
        default_factory=lambda: [{"metric": "fpr", "op": "<=", "value": 0}]
    )
    optimization_targets: list[dict[str, Any]] = field(
        default_factory=lambda: [{"metric": "sensitivity", "direction": "max"}]
    )

    @property
    def optimization_target(self) -> dict[str, Any]:
        """Primary optimization target (first in list). Backward compat."""
        return (
            self.optimization_targets[0]
            if self.optimization_targets
            else {"metric": "sensitivity", "direction": "max"}
        )

    convergence_patience: int = 3
    context_file: str = ".forge-world/evolution-context.md"

    # Memory
    memory_file: str = ".forge-world/evolution-memory.json"

    # Sensitivity
    run_sensitivity: bool = True
    sensitivity_cache_file: str = ".forge-world/sensitivity-cache.json"

    # Proposals
    proposal_file: str = ".forge-world/parameter-proposal.json"

    # Decomposition
    decompose_changes: bool = True

    # Exploration
    exploration_budget: int = 0  # lateral/worse moves allowed (0 = greedy)
    exploration_temperature: float = 1.0  # initial acceptance temperature
    temperature_decay: float = 0.8  # temperature *= decay each round

    # Convergence
    convergence_window: int = 5  # rolling window for oscillation detection
    min_progress: float = 0.001  # minimum metric change to count as progress

    # Auto-tuning (Optuna)
    auto_tune: bool = False
    auto_tune_trials: int = 20
    auto_tune_journal: str = ".forge-world/optuna-journal.log"


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
        metrics: dict[str, float] = {
            "pass_rate": am.mean_pass_rate,
            "sensitivity": am.mean_sensitivity,
            "fpr": am.worst_case_fpr,
            "f1": am.mean_f1,
        }
    else:
        metrics = {
            "pass_rate": report.pass_rate,
            "sensitivity": report.sensitivity,
            "fpr": report.fpr,
            "f1": report.f1,
        }

    # Include performance metrics if available
    perf = getattr(report, "performance", None)
    if perf is not None:
        metrics["latency_mean_ms"] = perf.latency_mean_ms
        metrics["latency_p50_ms"] = perf.latency_p50_ms
        metrics["latency_p95_ms"] = perf.latency_p95_ms
        metrics["latency_p99_ms"] = perf.latency_p99_ms
        metrics["throughput"] = perf.throughput_items_per_sec

    return metrics


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
        self._pending_validation_errors = ""
        self._last_tuner_result = None

    def run(self) -> EvolutionResult:
        """Execute the evolution loop."""
        result = EvolutionResult()
        no_improvement_count = 0

        # 1. Load memory (persists across restarts)
        memory = EvolutionMemory.load(Path(self.config.memory_file))
        memory.target_metric = self.config.optimization_target.get("metric", "sensitivity")
        memory.target_direction = self.config.optimization_target.get("direction", "max")

        # 2. Run sensitivity analysis (if enabled, cached)
        sensitivity = self._maybe_run_sensitivity()

        # 3. Initial benchmark
        report = self._run_bench()
        metrics_before = _extract_metrics(report)
        result.starting_metrics = dict(metrics_before)

        # Exploration state
        temperature = self.config.exploration_temperature
        exploration_remaining = self.config.exploration_budget
        metric_history: list[float] = []

        for i in range(1, self.config.max_iterations + 1):
            if self.on_iteration_start:
                self.on_iteration_start(i)

            iteration = IterationResult(
                iteration=i,
                had_changes=False,
                constraint_violated=False,
                metrics_before=dict(metrics_before),
            )

            # 4. Capture config before agent runs
            config_before = self._capture_config()

            # 5. Build context (with memory, sensitivity, enhanced clusters)
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
                            r for r in report.stable_reports[0].report.item_results if not r.passed
                        ]
                    elif hasattr(report, "item_results"):
                        failing = [r for r in report.item_results if not r.passed]
                    else:
                        failing = []
                    clusters = self.diagnostics.diagnose(failing)
                    diagnostics_data = [c.to_dict() for c in clusters]
                except Exception:
                    pass

            exploration_state = {
                "temperature": round(temperature, 4),
                "budget_remaining": exploration_remaining,
                "budget_total": self.config.exploration_budget,
                "metric_history_len": len(metric_history),
                "convergence_window": self.config.convergence_window,
            }

            context = build_evolution_context(
                report=report,
                regression=regression,
                pipeline_config_schema=self.pipeline_config_schema,
                hard_constraints=self.config.hard_constraints,
                optimization_targets=self.config.optimization_targets,
                sample_size=self.run_kwargs.get("sample_size"),
                diagnostics=diagnostics_data,
                memory=memory,
                sensitivity=sensitivity,
                evolve_mode=True,
                exploration_state=exploration_state if self.config.exploration_budget > 0 else None,
                validation_errors=self._pending_validation_errors,
                tuner_result=self._last_tuner_result,
            )
            self._pending_validation_errors = ""

            self._write_context(context)

            # 6. Invoke agent
            try:
                agent_ok = self._invoke_agent()
            except Exception:
                self._record_memory_entry(
                    memory,
                    i,
                    config_before,
                    [],
                    metrics_before,
                    None,
                    False,
                    "agent_error",
                    [],
                    "",
                )
                iteration.rollback_reason = "agent_error"
                result.iterations.append(iteration)
                result.convergence_reason = "agent_error"
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                break

            if not agent_ok:
                self._record_memory_entry(
                    memory,
                    i,
                    config_before,
                    [],
                    metrics_before,
                    None,
                    False,
                    "agent_error",
                    [],
                    "",
                )
                iteration.rollback_reason = "agent_error"
                result.iterations.append(iteration)
                result.convergence_reason = "agent_error"
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                break

            # 7. Check for parameter proposals
            proposal = ProposalFile.load(Path(self.config.proposal_file))
            old_config = None
            agent_reasoning = ""
            validation_error_summary = ""

            if proposal is not None:
                raw_proposals = list(proposal.proposals)
                agent_reasoning = proposal.agent_notes
            else:
                raw_proposals = []

            if raw_proposals:
                proposals_to_apply = raw_proposals
                # Validate against schema if available
                if self.pipeline_config_schema and proposals_to_apply:
                    validation_result = validate_proposals(
                        proposals_to_apply,
                        self.pipeline_config_schema,
                        config_before,
                    )
                    proposals_to_apply = validation_result.valid_proposals
                    if validation_result.errors:
                        validation_error_summary = validation_result.error_summary()
                try:
                    if proposals_to_apply:
                        old_config = apply_proposals(self.runner.pipeline, proposals_to_apply)
                    if validation_error_summary:
                        agent_reasoning = f"{validation_error_summary}\n{agent_reasoning}"
                except (KeyError, Exception) as exc:
                    logging.getLogger("forge_world").warning("Proposal failed: %s", exc)
                    agent_reasoning = f"[PROPOSAL ERROR: {exc}] {agent_reasoning}"

            if proposal is not None:
                Path(self.config.proposal_file).unlink(missing_ok=True)

            # Store validation errors for next iteration's context
            if validation_error_summary:
                self._pending_validation_errors = validation_error_summary

            # 8. Check for file changes
            has_file_changes = self._check_for_changes()
            has_proposal_changes = proposal is not None and old_config is not None
            has_changes = has_file_changes or has_proposal_changes
            iteration.had_changes = has_changes

            if not has_changes:
                self._record_memory_entry(
                    memory,
                    i,
                    config_before,
                    [],
                    metrics_before,
                    None,
                    False,
                    "no_changes",
                    [],
                    agent_reasoning,
                )
                no_improvement_count += 1
                iteration.rollback_reason = "no_changes"
                result.iterations.append(iteration)
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)
                if no_improvement_count >= self.config.convergence_patience:
                    result.convergence_reason = "converged"
                    break
                continue

            # 9. Decompose if multiple files changed
            decomposed = False
            if self.config.decompose_changes and has_file_changes:
                changed_files = self._get_changed_files()
                if len(changed_files) > 1:
                    accepted_files, decomposed_metrics = self._decompose_and_test(metrics_before)
                    if decomposed_metrics is not None:
                        decomposed = True
                        metrics_after = decomposed_metrics
                        iteration.metrics_after = metrics_after

                        violations = self._check_constraints(metrics_after)
                        if violations:
                            iteration.constraint_violated = True
                            iteration.constraint_violations = violations
                            reason = f"constraint_violated: {', '.join(violations)}"
                            iteration.rollback_reason = reason
                            self._rollback()
                            if old_config:
                                rollback_proposals(self.runner.pipeline, old_config)
                            config_after = self._capture_config()
                            param_changes = compute_parameter_diff(config_before, config_after)
                            self._record_memory_entry(
                                memory,
                                i,
                                config_before,
                                accepted_files,
                                metrics_before,
                                metrics_after,
                                False,
                                reason,
                                violations,
                                agent_reasoning,
                                param_changes=param_changes,
                            )
                            result.iterations.append(iteration)
                            if self.on_iteration_complete:
                                self.on_iteration_complete(i, iteration)
                            no_improvement_count += 1
                            if no_improvement_count >= self.config.convergence_patience:
                                result.convergence_reason = "converged"
                                break
                            continue

                        accept, reason = self._should_accept(
                            metrics_before,
                            metrics_after,
                            i,
                            temperature,
                            exploration_remaining,
                        )
                        iteration.improved = self._is_improved(metrics_before, metrics_after)
                        iteration.accepted = accept

                        config_after = self._capture_config()
                        param_changes = compute_parameter_diff(config_before, config_after)

                        if accept:
                            if reason == "exploration":
                                exploration_remaining -= 1
                            self._commit_changes(i, metrics_before, metrics_after)
                            self._record_memory_entry(
                                memory,
                                i,
                                config_before,
                                accepted_files,
                                metrics_before,
                                metrics_after,
                                True,
                                reason,
                                [],
                                agent_reasoning,
                                param_changes=param_changes,
                            )
                            # Auto-tune after acceptance
                            metrics_after = self._maybe_auto_tune(i, metrics_after)
                            metrics_before = metrics_after
                            report = self._run_bench()
                            no_improvement_count = 0
                        else:
                            iteration.rollback_reason = reason
                            self._rollback()
                            if old_config:
                                rollback_proposals(self.runner.pipeline, old_config)
                            self._record_memory_entry(
                                memory,
                                i,
                                config_before,
                                accepted_files,
                                metrics_before,
                                metrics_after,
                                False,
                                reason,
                                [],
                                agent_reasoning,
                                param_changes=param_changes,
                            )
                            no_improvement_count += 1

                        result.iterations.append(iteration)
                        if self.on_iteration_complete:
                            self.on_iteration_complete(i, iteration)

                        metric_history.append(
                            metrics_after.get(
                                self.config.optimization_target.get("metric", "sensitivity"), 0
                            )
                        )
                        # Only decay temperature when exploration budget is consumed
                        if reason == "exploration":
                            temperature *= self.config.temperature_decay

                        if no_improvement_count >= self.config.convergence_patience:
                            result.convergence_reason = "converged"
                            break

                        converged, conv_reason = self._detect_convergence(metric_history)
                        if converged:
                            result.convergence_reason = conv_reason
                            break

                        continue

            if not decomposed:
                # 10. Benchmark (all-or-nothing path)
                new_report = self._run_bench()
                metrics_after = _extract_metrics(new_report)
                iteration.metrics_after = metrics_after

                # Get changed files for memory
                changed_files = self._get_changed_files() if has_file_changes else []

                # 11. Check constraints
                violations = self._check_constraints(metrics_after)
                if violations:
                    iteration.constraint_violated = True
                    iteration.constraint_violations = violations
                    reason = f"constraint_violated: {', '.join(violations)}"
                    iteration.rollback_reason = reason
                    self._rollback()
                    if old_config:
                        rollback_proposals(self.runner.pipeline, old_config)
                    config_after = self._capture_config()
                    param_changes = compute_parameter_diff(config_before, config_after)
                    self._record_memory_entry(
                        memory,
                        i,
                        config_before,
                        changed_files,
                        metrics_before,
                        metrics_after,
                        False,
                        reason,
                        violations,
                        agent_reasoning,
                        param_changes=param_changes,
                    )
                    result.iterations.append(iteration)
                    if self.on_iteration_complete:
                        self.on_iteration_complete(i, iteration)
                    no_improvement_count += 1
                    if no_improvement_count >= self.config.convergence_patience:
                        result.convergence_reason = "converged"
                        break
                    continue

                # 12. Smart acceptance
                accept, reason = self._should_accept(
                    metrics_before,
                    metrics_after,
                    i,
                    temperature,
                    exploration_remaining,
                )
                iteration.improved = self._is_improved(metrics_before, metrics_after)
                iteration.accepted = accept

                config_after = self._capture_config()
                param_changes = compute_parameter_diff(config_before, config_after)

                # 13. Accept or reject
                if accept:
                    if reason == "exploration":
                        exploration_remaining -= 1
                    self._commit_changes(i, metrics_before, metrics_after)
                    self._record_memory_entry(
                        memory,
                        i,
                        config_before,
                        changed_files,
                        metrics_before,
                        metrics_after,
                        True,
                        reason,
                        [],
                        agent_reasoning,
                        param_changes=param_changes,
                    )
                    # Auto-tune after acceptance
                    metrics_after = self._maybe_auto_tune(i, metrics_after)
                    metrics_before = metrics_after
                    report = new_report
                    no_improvement_count = 0
                else:
                    iteration.rollback_reason = reason
                    self._rollback()
                    if old_config:
                        rollback_proposals(self.runner.pipeline, old_config)
                    self._record_memory_entry(
                        memory,
                        i,
                        config_before,
                        changed_files,
                        metrics_before,
                        metrics_after,
                        False,
                        reason,
                        [],
                        agent_reasoning,
                        param_changes=param_changes,
                    )
                    no_improvement_count += 1

                result.iterations.append(iteration)
                if self.on_iteration_complete:
                    self.on_iteration_complete(i, iteration)

                # 14. Track metric history and temperature
                metric_history.append(
                    metrics_after.get(
                        self.config.optimization_target.get("metric", "sensitivity"), 0
                    )
                )
                # Only decay temperature when exploration budget is consumed
                if reason == "exploration":
                    temperature *= self.config.temperature_decay

                if no_improvement_count >= self.config.convergence_patience:
                    result.convergence_reason = "converged"
                    break

                # 15. Convergence check
                converged, conv_reason = self._detect_convergence(metric_history)
                if converged:
                    result.convergence_reason = conv_reason
                    break
        else:
            result.convergence_reason = "max_iterations"

        result.total_iterations = len(result.iterations)
        result.accepted_iterations = sum(1 for it in result.iterations if it.accepted)
        result.final_metrics = dict(metrics_before)
        return result

    def _maybe_run_sensitivity(self) -> SensitivityReport | None:
        """Run sensitivity analysis if enabled, using cache if available."""
        if not self.config.run_sensitivity:
            return None

        cache_path = Path(self.config.sensitivity_cache_file)
        cached = SensitivityReport.load(cache_path)

        # Check if cache is still valid (config hash matches)
        if cached is not None:
            try:
                import hashlib
                import json

                current_config = self.runner.pipeline.get_config()
                current_hash = hashlib.sha256(
                    json.dumps(current_config, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]
                if cached.config_hash == current_hash:
                    return cached
            except Exception:
                pass

        try:
            target = self.config.optimization_target
            report = compute_sensitivity(
                self.runner,
                target_metric=target.get("metric", "sensitivity"),
                target_direction=target.get("direction", "max"),
                hard_constraints=self.config.hard_constraints,
                run_kwargs=self.run_kwargs,
            )
            report.save(cache_path)
            return report
        except Exception:
            return None

    def _maybe_auto_tune(
        self,
        iteration: int,
        current_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Run Optuna auto-tuning if enabled. Returns updated metrics."""
        if not self.config.auto_tune or not self.pipeline_config_schema:
            return current_metrics

        tuner_result = self._run_auto_tune()
        if tuner_result is None or tuner_result.n_trials_completed == 0:
            return current_metrics

        self._last_tuner_result = tuner_result

        try:
            from forge_world.core.tuner import apply_best_params

            old = apply_best_params(self.runner.pipeline, tuner_result)
            tuned_report = self._run_bench()
            tuned_metrics = _extract_metrics(tuned_report)
            violations = self._check_constraints(tuned_metrics)
            if violations:
                self.runner.pipeline.set_config(old)  # rollback tuning
                return current_metrics
            return tuned_metrics
        except Exception as exc:
            logging.getLogger("forge_world").warning("Auto-tune apply failed: %s", exc)
            return current_metrics

    def _run_auto_tune(self):
        """Run Optuna auto-tuning. Returns TunerResult or None."""
        try:
            from forge_world.core.tuner import TunerConfig, run_tuning
            from forge_world.core.sensitivity import walk_numeric_parameters

            schema_params = walk_numeric_parameters(self.pipeline_config_schema)
            if not schema_params:
                return None
            tuner_config = TunerConfig(
                n_trials=self.config.auto_tune_trials,
                journal_path=self.config.auto_tune_journal,
                optimization_targets=self.config.optimization_targets,
                hard_constraints=self.config.hard_constraints,
            )
            return run_tuning(self.runner, tuner_config, schema_params, self.run_kwargs)
        except ImportError:
            return None
        except Exception as exc:
            logging.getLogger("forge_world").warning("Auto-tune failed: %s", exc)
            return None

    def _capture_config(self) -> dict[str, Any]:
        """Capture current pipeline config as a dict."""
        try:
            config = self.runner.pipeline.get_config()
            if isinstance(config, dict):
                return dict(config)
            if hasattr(config, "model_dump"):
                return config.model_dump()
            return dict(config)
        except Exception:
            return {}

    def _record_memory_entry(
        self,
        memory: EvolutionMemory,
        iteration: int,
        config_before: dict[str, Any],
        files_changed: list[str],
        metrics_before: dict[str, float],
        metrics_after: dict[str, float] | None,
        accepted: bool,
        reason: str,
        constraint_violations: list[str],
        agent_reasoning: str,
        param_changes: list | None = None,
    ) -> None:
        """Create and save a memory entry."""
        if param_changes is None:
            config_after = self._capture_config()
            param_changes = compute_parameter_diff(config_before, config_after)

        entry = MemoryEntry(
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            files_changed=files_changed,
            parameter_changes=param_changes,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            accepted=accepted,
            reason=reason,
            constraint_violations=constraint_violations,
            agent_reasoning=agent_reasoning,
        )
        memory.add_entry(entry)
        memory.save(Path(self.config.memory_file))

    def _run_bench(self):
        """Run benchmark using configured strategy."""
        kwargs = dict(self.run_kwargs)
        seed = kwargs.pop("seed", None)
        seed_strategy = kwargs.pop("seed_strategy", None)
        sample_size = kwargs.pop("sample_size", None)
        tier = kwargs.pop("tier", None)

        if seed is not None:
            return self.runner.run(
                seed=seed,
                sample_size=sample_size,
                tier=tier,
            )
        elif seed_strategy is not None:
            return self.runner.run_multi(
                seed_strategy,
                sample_size=sample_size,
                tier=tier,
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
        cmd = self.config.agent_command.replace("{context_file}", self.config.context_file)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0

    def _check_for_changes(self) -> bool:
        """Check if git working tree has changes."""
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())

    def _get_changed_files(self) -> list[str]:
        """git diff --name-only → list of changed file paths."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return [f for f in result.stdout.strip().split("\n") if f]
        except Exception:
            return []

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
        """Check if optimization targets improved.

        Single target (len==1): strict improvement on that metric.
        Multi target: Pareto improvement — at least one target improved
        AND none worsened (beyond min_progress tolerance).
        """
        targets = self.config.optimization_targets
        if len(targets) <= 1:
            target = self.config.optimization_target
            metric = target.get("metric", "sensitivity")
            direction = target.get("direction", "max")
            before_val = before.get(metric, 0)
            after_val = after.get(metric, 0)
            if direction == "max":
                return after_val > before_val
            return after_val < before_val

        # Multi-target: Pareto improvement
        improved_any = False
        min_progress = self.config.min_progress
        for target in targets:
            metric = target.get("metric", "sensitivity")
            direction = target.get("direction", "max")
            before_val = before.get(metric, 0)
            after_val = after.get(metric, 0)
            if direction == "max":
                diff = after_val - before_val
            else:
                diff = before_val - after_val  # positive = improvement for min
            if diff > min_progress:
                improved_any = True
            elif diff < -min_progress:
                return False  # Worsened on this target
        return improved_any

    def _should_accept(
        self,
        before: dict[str, float],
        after: dict[str, float],
        iteration: int,
        temperature: float,
        exploration_remaining: int,
    ) -> tuple[bool, str]:
        """Smarter acceptance criterion. Decision cascade:

        1. If improved on target(s) → ("improved")
        2. If exploration_budget > 0 and temperature allows → ("exploration")
        3. → ("not_improved")
        """
        if self._is_improved(before, after):
            return True, "improved"

        if exploration_remaining > 0 and temperature > 0:
            target = self.config.optimization_target
            metric = target.get("metric", "sensitivity")
            target_before = before.get(metric, 0)
            target_after = after.get(metric, 0)
            delta = (target_before - target_after) / max(target_before, 0.001)
            if delta <= 0:
                # Equal or better — accept
                return True, "exploration"
            accept_prob = math.exp(-delta / temperature)
            if random.random() < accept_prob:
                return True, "exploration"

        return False, "not_improved"

    def _detect_convergence(
        self,
        metric_history: list[float],
    ) -> tuple[bool, str]:
        """Check for convergence beyond simple patience counter.

        Uses rolling window over last N target metric values:
        1. "oscillating": std(window) > 0 but net change < min_progress
        2. "plateau": all values in window within min_progress of each other
        """
        window = self.config.convergence_window
        if len(metric_history) < window:
            return False, ""

        recent = metric_history[-window:]
        min_val = min(recent)
        max_val = max(recent)
        net_change = abs(recent[-1] - recent[0])

        # Plateau: all values nearly identical
        if (max_val - min_val) < self.config.min_progress:
            return True, "plateau"

        # Oscillating: variance but no net progress
        if net_change < self.config.min_progress and (max_val - min_val) > self.config.min_progress:
            return True, "oscillating"

        return False, ""

    def _decompose_and_test(
        self,
        metrics_before: dict[str, float],
    ) -> tuple[list[str], dict[str, float] | None]:
        """Test multi-file changes independently, accept the best subset.

        Returns: (accepted_files, metrics_after)
        If metrics_after is None, caller should fall back to all-or-nothing.
        """
        patch_path: str | None = None
        stash_pushed = False
        try:
            changed_files = self._get_changed_files()
            if len(changed_files) <= 1:
                return (changed_files, None)

            # Save full diff
            diff_result = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if diff_result.returncode != 0:
                return (changed_files, None)

            full_patch = diff_result.stdout
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(full_patch)
                patch_path = f.name

            # Stash changes (keeps them safe until we're done)
            subprocess.run(
                ["git", "stash"],
                capture_output=True,
                timeout=10,
            )
            stash_pushed = True

            # Test each file independently
            file_results: list[tuple[str, dict[str, float], bool]] = []
            for file_path in changed_files:
                try:
                    # Apply only this file's changes
                    apply_result = subprocess.run(
                        ["git", "apply", f"--include={file_path}", patch_path],
                        capture_output=True,
                        timeout=10,
                    )
                    if apply_result.returncode != 0:
                        continue

                    # Benchmark
                    file_report = self._run_bench()
                    file_metrics = _extract_metrics(file_report)

                    # Check constraints
                    violations = self._check_constraints(file_metrics)
                    file_results.append((file_path, file_metrics, len(violations) == 0))

                    # Reset
                    subprocess.run(
                        ["git", "checkout", "--", "."],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    subprocess.run(
                        ["git", "checkout", "--", "."],
                        capture_output=True,
                        timeout=10,
                    )
                    continue

            if not file_results:
                return (changed_files, None)

            # Select files that don't violate constraints and improve or are neutral
            target = self.config.optimization_target
            metric = target.get("metric", "sensitivity")
            direction = target.get("direction", "max")
            before_val = metrics_before.get(metric, 0)

            accepted_files = []
            for file_path, file_metrics, passes_constraints in file_results:
                if not passes_constraints:
                    continue
                after_val = file_metrics.get(metric, 0)
                if direction == "max":
                    if after_val >= before_val - self.config.min_progress:
                        accepted_files.append(file_path)
                else:
                    if after_val <= before_val + self.config.min_progress:
                        accepted_files.append(file_path)

            if not accepted_files:
                return ([], None)

            # Reset and apply accepted subset from the patch file
            subprocess.run(
                ["git", "checkout", "--", "."],
                capture_output=True,
                timeout=10,
            )

            for file_path in accepted_files:
                subprocess.run(
                    ["git", "apply", f"--include={file_path}", patch_path],
                    capture_output=True,
                    timeout=10,
                )

            # Benchmark the combined subset
            combined_report = self._run_bench()
            combined_metrics = _extract_metrics(combined_report)

            # Verify constraints on combined
            violations = self._check_constraints(combined_metrics)
            if violations:
                # Fall back to best single file
                subprocess.run(
                    ["git", "checkout", "--", "."],
                    capture_output=True,
                    timeout=10,
                )
                if file_results:
                    # Find best single file
                    best_file = None
                    best_val = before_val
                    for fp, fm, passes in file_results:
                        if not passes:
                            continue
                        val = fm.get(metric, 0)
                        if direction == "max" and val > best_val:
                            best_val = val
                            best_file = (fp, fm)
                        elif direction == "min" and val < best_val:
                            best_val = val
                            best_file = (fp, fm)

                    if best_file:
                        subprocess.run(
                            ["git", "apply", f"--include={best_file[0]}", patch_path],
                            capture_output=True,
                            timeout=10,
                        )
                        return ([best_file[0]], best_file[1])

                return ([], None)

            return (accepted_files, combined_metrics)

        except Exception:
            return ([], None)
        finally:
            # Always clean up: drop stash and remove patch file
            if stash_pushed:
                subprocess.run(
                    ["git", "stash", "drop"],
                    capture_output=True,
                    timeout=10,
                )
            if patch_path is not None:
                Path(patch_path).unlink(missing_ok=True)

    def _rollback(self) -> None:
        """Discard uncommitted changes."""
        subprocess.run(
            ["git", "checkout", "--", "."],
            capture_output=True,
            timeout=10,
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
        msg = f"forge evolve: iteration {iteration} ({target}: {before_val:.4f} -> {after_val:.4f})"
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True,
            timeout=30,
        )
