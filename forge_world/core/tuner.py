"""Optuna-based automatic parameter tuning.

Provides a numeric parameter optimization layer that uses Bayesian optimization
(TPE for single-objective, NSGA-II for multi-objective) to efficiently explore
the parameter space. Runs as an ask-tell loop without requiring agent interaction.

Optuna is an optional dependency — import errors are raised with install instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_world.core.sensitivity import (
    get_nested_value,
    set_nested_value,
    walk_numeric_parameters,
)

logger = logging.getLogger("forge_world.tuner")


def _ensure_optuna():
    """Lazy-import optuna, raising a clear error if missing."""
    try:
        import optuna  # noqa: F811

        return optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for auto-tuning. Install with:\n"
            "  pip install 'forge-world[tuning]'\n"
            "  # or: pip install optuna>=4.0"
        )


@dataclass
class TunerConfig:
    n_trials: int = 20
    journal_path: str = ".forge-world/optuna-journal.log"
    study_name: str = "forge-world-tuning"
    optimization_targets: list[dict[str, Any]] = field(
        default_factory=lambda: [{"metric": "sensitivity", "direction": "max"}]
    )
    hard_constraints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TunerResult:
    n_trials_completed: int
    best_params: dict[str, Any]
    best_values: dict[str, float]  # metric -> value
    pareto_front: list[dict[str, Any]]  # [{params: ..., values: ...}]
    all_trials_summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials_completed": self.n_trials_completed,
            "best_params": self.best_params,
            "best_values": self.best_values,
            "pareto_front": self.pareto_front,
            "all_trials_summary": self.all_trials_summary,
        }

    def to_prompt_context(self) -> str:
        """Markdown summary for agent context."""
        if self.n_trials_completed == 0:
            return ""

        lines = ["## Auto-Tuning Results", ""]
        lines.append(f"Completed {self.n_trials_completed} Optuna trials.")
        lines.append("")

        if self.best_params:
            lines.append("### Best Parameters")
            for path, value in sorted(self.best_params.items()):
                lines.append(f"- {path}: {value}")
            lines.append("")

        if self.best_values:
            lines.append("### Best Metrics")
            for metric, value in sorted(self.best_values.items()):
                lines.append(f"- {metric}: {value:.4f}")
            lines.append("")

        if len(self.pareto_front) > 1:
            lines.append(f"### Pareto Front ({len(self.pareto_front)} solutions)")
            for i, sol in enumerate(self.pareto_front[:5]):
                vals = ", ".join(f"{k}={v:.4f}" for k, v in sorted(sol.get("values", {}).items()))
                lines.append(f"- Solution {i + 1}: {vals}")
            if len(self.pareto_front) > 5:
                lines.append(f"  (+{len(self.pareto_front) - 5} more)")
            lines.append("")

        return "\n".join(lines)


def run_tuning(
    runner: Any,
    tuner_config: TunerConfig,
    schema_params: list[dict[str, Any]] | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> TunerResult:
    """Run Optuna parameter optimization using ask-tell loop.

    1. study.ask() -> trial with suggested param values
    2. Apply params to pipeline
    3. Benchmark
    4. Extract metrics, check hard constraints (penalty if violated)
    5. study.tell(trial, values)

    Single-objective: TPESampler
    Multi-objective: NSGAIISampler
    Persistence: JournalStorage at journal_path
    """
    optuna = _ensure_optuna()
    from forge_world.core.evolve import _extract_metrics

    pipeline = runner.pipeline

    # Get schema params if not provided
    if schema_params is None:
        schema = pipeline.get_config_schema()
        schema_params = walk_numeric_parameters(schema)

    if not schema_params:
        return TunerResult(
            n_trials_completed=0,
            best_params={},
            best_values={},
            pareto_front=[],
            all_trials_summary=[],
        )

    original_config = pipeline.get_config()
    if not isinstance(original_config, dict):
        original_config = (
            original_config.model_dump()
            if hasattr(original_config, "model_dump")
            else dict(original_config)
        )

    targets = tuner_config.optimization_targets
    is_multi = len(targets) > 1

    # Determine Optuna directions
    directions = []
    for t in targets:
        directions.append("maximize" if t.get("direction", "max") == "max" else "minimize")

    # Penalty values for constraint violations
    penalty_values = []
    for t in targets:
        if t.get("direction", "max") == "max":
            penalty_values.append(-1e6)
        else:
            penalty_values.append(1e6)

    # Set up storage
    journal_path = Path(tuner_config.journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))

    # Create or load study
    if is_multi:
        sampler = optuna.samplers.NSGAIISampler()
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=tuner_config.study_name,
        storage=storage,
        directions=directions,
        sampler=sampler,
        load_if_exists=True,
    )

    # Build search space from schema params
    search_space = {}
    for param in schema_params:
        path = param["path"]
        param_min = param.get("minimum")
        param_max = param.get("maximum")
        param_type = param.get("type", "number")

        # Get current value as default center
        try:
            current = float(get_nested_value(original_config, path))
        except (KeyError, AttributeError, TypeError, ValueError):
            continue

        if param_min is None:
            param_min = current * 0.1 if current > 0 else current - abs(current) * 0.9
        if param_max is None:
            param_max = current * 2.0 if current > 0 else current + abs(current) * 0.9

        search_space[path] = {
            "min": float(param_min),
            "max": float(param_max),
            "type": param_type,
        }

    if not search_space:
        pipeline.set_config(original_config)
        return TunerResult(
            n_trials_completed=0,
            best_params={},
            best_values={},
            pareto_front=[],
            all_trials_summary=[],
        )

    all_trials: list[dict[str, Any]] = []

    # Register current config as initial trial so Optuna knows the baseline
    try:
        current_params = {}
        for path in search_space:
            try:
                current_params[path] = float(get_nested_value(original_config, path))
            except (KeyError, AttributeError, TypeError, ValueError):
                pass
        if current_params:
            baseline_report = _run_bench(runner, run_kwargs)
            baseline_metrics = _extract_metrics(baseline_report)
            baseline_violated = _check_constraints(baseline_metrics, tuner_config.hard_constraints)
            if baseline_violated:
                baseline_values = penalty_values
            else:
                baseline_values = [
                    baseline_metrics.get(t.get("metric", "sensitivity"), 0) for t in targets
                ]
            initial_trial = study.ask()
            study.tell(initial_trial, baseline_values)
    except Exception:
        pass  # Non-critical — just informing Optuna of baseline

    try:
        for _ in range(tuner_config.n_trials):
            trial = study.ask()

            # Suggest parameter values
            params: dict[str, Any] = {}
            for path, spec in search_space.items():
                if spec["type"] == "integer":
                    val = trial.suggest_int(path, int(spec["min"]), int(spec["max"]))
                else:
                    val = trial.suggest_float(path, spec["min"], spec["max"])
                params[path] = val

            # Apply to pipeline
            config = dict(original_config)
            for path, val in params.items():
                try:
                    set_nested_value(config, path, val)
                except (KeyError, AttributeError):
                    pass
            pipeline.set_config(config)

            # Benchmark
            report = _run_bench(runner, run_kwargs)
            metrics = _extract_metrics(report)

            # Check hard constraints
            violated = _check_constraints(metrics, tuner_config.hard_constraints)

            # Extract objective values
            if violated:
                values = penalty_values
            else:
                values = []
                for t in targets:
                    metric_name = t.get("metric", "sensitivity")
                    values.append(metrics.get(metric_name, 0))

            study.tell(trial, values)

            all_trials.append(
                {
                    "trial": trial.number,
                    "params": params,
                    "values": {t.get("metric", "?"): v for t, v in zip(targets, values)},
                    "violated": violated,
                }
            )

    finally:
        # Always restore original config
        pipeline.set_config(original_config)

    # Extract results
    n_completed = len(all_trials)

    if is_multi:
        best_trials = study.best_trials
        pareto_front = []
        for bt in best_trials:
            pareto_front.append(
                {
                    "params": bt.params,
                    "values": {t.get("metric", "?"): v for t, v in zip(targets, bt.values)},
                }
            )
        # Use first Pareto solution as "best"
        if pareto_front:
            best_params = pareto_front[0]["params"]
            best_values = pareto_front[0]["values"]
        else:
            best_params = {}
            best_values = {}
    else:
        best_trial = study.best_trial
        best_params = best_trial.params
        best_values = {targets[0].get("metric", "sensitivity"): best_trial.value}
        pareto_front = [{"params": best_params, "values": best_values}]

    return TunerResult(
        n_trials_completed=n_completed,
        best_params=best_params,
        best_values=best_values,
        pareto_front=pareto_front,
        all_trials_summary=all_trials,
    )


def apply_best_params(pipeline: Any, tuner_result: TunerResult) -> dict[str, Any]:
    """Apply best params from tuning result. Returns old config for rollback."""
    old_config = pipeline.get_config()
    if not isinstance(old_config, dict):
        old_config = (
            old_config.model_dump() if hasattr(old_config, "model_dump") else dict(old_config)
        )
    old_config_copy = dict(old_config)

    new_config = dict(old_config)
    for path, val in tuner_result.best_params.items():
        try:
            set_nested_value(new_config, path, val)
        except (KeyError, AttributeError):
            pass
    pipeline.set_config(new_config)
    return old_config_copy


def _run_bench(runner: Any, run_kwargs: dict[str, Any] | None = None):
    """Run benchmark with given kwargs."""
    kwargs = dict(run_kwargs or {})
    seed = kwargs.pop("seed", None)
    seed_strategy = kwargs.pop("seed_strategy", None)
    sample_size = kwargs.pop("sample_size", None)
    tier = kwargs.pop("tier", None)

    if seed is not None:
        return runner.run(seed=seed, sample_size=sample_size, tier=tier)
    elif seed_strategy is not None:
        return runner.run_multi(seed_strategy, sample_size=sample_size, tier=tier)
    else:
        return runner.run(sample_size=sample_size, tier=tier)


def _check_constraints(
    metrics: dict[str, float],
    hard_constraints: list[dict[str, Any]],
) -> bool:
    """Check if any hard constraint is violated. Returns True if violated."""
    for constraint in hard_constraints:
        metric = constraint.get("metric", "")
        op = constraint.get("op", "<=")
        value = float(constraint.get("value", 0))
        actual = metrics.get(metric)
        if actual is None:
            continue
        if op == "<=" and actual > value:
            return True
        elif op == ">=" and actual < value:
            return True
        elif op == "<" and actual >= value:
            return True
        elif op == ">" and actual <= value:
            return True
    return False
