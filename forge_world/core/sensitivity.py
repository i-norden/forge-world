"""Parameter sensitivity analysis via finite-difference perturbation.

Ranks pipeline parameters by their impact on the target metric, enabling
the agent to focus on high-impact knobs and avoid wasting iterations on
parameters that don't matter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ParameterSensitivity:
    """Sensitivity of target metric to a single parameter."""

    path: str  # "weight_ela" or "ela.quality"
    current_value: float
    delta: float  # perturbation size
    metric_minus: float  # target metric at value - delta
    metric_plus: float  # target metric at value + delta
    impact: float  # abs(metric_plus - metric_minus)
    direction: str  # "increase" or "decrease" (which improves target)
    constrained: bool  # True if either perturbation violated a constraint
    constraint_detail: str = ""  # e.g. "increase causes fpr=0.05>0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "current_value": self.current_value,
            "delta": self.delta,
            "metric_minus": self.metric_minus,
            "metric_plus": self.metric_plus,
            "impact": self.impact,
            "direction": self.direction,
            "constrained": self.constrained,
            "constraint_detail": self.constraint_detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSensitivity:
        return cls(
            path=data["path"],
            current_value=data["current_value"],
            delta=data["delta"],
            metric_minus=data["metric_minus"],
            metric_plus=data["metric_plus"],
            impact=data["impact"],
            direction=data["direction"],
            constrained=data["constrained"],
            constraint_detail=data.get("constraint_detail", ""),
        )


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis results."""

    parameters: list[ParameterSensitivity]  # sorted by impact descending
    target_metric: str
    baseline_value: float
    config_hash: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "target_metric": self.target_metric,
            "baseline_value": self.baseline_value,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensitivityReport:
        return cls(
            parameters=[
                ParameterSensitivity.from_dict(p) for p in data["parameters"]
            ],
            target_metric=data["target_metric"],
            baseline_value=data["baseline_value"],
            config_hash=data["config_hash"],
            timestamp=data["timestamp"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> SensitivityReport | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def to_prompt_context(self) -> str:
        """Produce actionable markdown."""
        if not self.parameters:
            return ""

        lines: list[str] = []
        lines.append(f"## Parameter Sensitivity (ranked by impact on {self.target_metric})")
        lines.append("")
        lines.append("| Parameter | Current | Impact | Direction | Constrained? |")
        lines.append("|-----------|---------|--------|-----------|--------------|")

        total_impact = sum(p.impact for p in self.parameters)
        cumulative = 0.0

        for p in self.parameters:
            constrained_str = p.constraint_detail if p.constrained else ""
            lines.append(
                f"| {p.path} | {p.current_value:.4g} | {p.impact:.4f} "
                f"| {p.direction} | {constrained_str} |"
            )
            cumulative += p.impact

        lines.append("")

        # Summary
        if total_impact > 0:
            top5_impact = sum(p.impact for p in self.parameters[:5])
            pct = top5_impact / total_impact * 100
            lines.append(f"Top {min(5, len(self.parameters))} parameters account for {pct:.0f}% of total sensitivity.")

        negligible = [p for p in self.parameters if p.impact < 0.005]
        if negligible:
            lines.append(
                f"{len(negligible)} parameter(s) have negligible impact (<0.005)."
            )
        lines.append("")

        return "\n".join(lines)


def walk_numeric_parameters(
    schema: dict[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Walk a JSON Schema and extract all numeric parameters.

    Returns: [{"path": "weight_ela", "type": "number", "minimum": 0.0,
               "maximum": 1.0}, ...]

    Handles:
    - Pydantic-generated schemas with $defs and $ref
    - Nested properties (flattened to dotted paths)
    - minimum/maximum/exclusiveMinimum/exclusiveMaximum
    - Integer and number types
    """
    result: list[dict[str, Any]] = []
    defs = schema.get("$defs", schema.get("definitions", {}))

    def _resolve_ref(ref: str) -> dict[str, Any]:
        # Handle "#/$defs/SomeModel" or "#/definitions/SomeModel"
        parts = ref.lstrip("#/").split("/")
        obj = schema
        for part in parts:
            obj = obj.get(part, {})
        return obj

    def _walk(node: dict[str, Any], path_prefix: str) -> None:
        # Handle $ref
        if "$ref" in node:
            resolved = _resolve_ref(node["$ref"])
            _walk(resolved, path_prefix)
            return

        # Handle allOf (common in Pydantic schemas)
        if "allOf" in node:
            for sub in node["allOf"]:
                _walk(sub, path_prefix)
            return

        node_type = node.get("type", "")

        if node_type in ("number", "integer"):
            entry: dict[str, Any] = {"path": path_prefix, "type": node_type}
            if "minimum" in node:
                entry["minimum"] = node["minimum"]
            if "maximum" in node:
                entry["maximum"] = node["maximum"]
            if "exclusiveMinimum" in node:
                entry["minimum"] = node["exclusiveMinimum"]
            if "exclusiveMaximum" in node:
                entry["maximum"] = node["exclusiveMaximum"]
            if "default" in node:
                entry["default"] = node["default"]
            result.append(entry)

        elif node_type == "object" or "properties" in node:
            for prop_name, prop_schema in node.get("properties", {}).items():
                child_path = f"{path_prefix}.{prop_name}" if path_prefix else prop_name
                _walk(prop_schema, child_path)

    _walk(schema, prefix)
    return result


def get_nested_value(config: Any, path: str) -> Any:
    """Get a value from a nested config by dotted path.
    Handles both dicts and objects with attributes."""
    parts = path.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def set_nested_value(config: Any, path: str, value: Any) -> None:
    """Set a value in a nested config by dotted path.
    Handles both dicts and objects with attributes."""
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    if isinstance(current, dict):
        current[parts[-1]] = value
    else:
        setattr(current, parts[-1], value)


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a hash of the config for cache invalidation."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_sensitivity(
    runner: Any,
    target_metric: str = "sensitivity",
    target_direction: str = "max",
    hard_constraints: list[dict[str, Any]] | None = None,
    delta_fraction: float = 0.1,
    run_kwargs: dict[str, Any] | None = None,
) -> SensitivityReport:
    """Compute parameter sensitivity by finite-difference perturbation.

    For each numeric parameter in pipeline.get_config_schema():
    1. Read current value from pipeline.get_config()
    2. Set to current - delta, run bench, record metric
    3. Set to current + delta, run bench, record metric
    4. Restore original value
    5. Impact = abs(metric_plus - metric_minus)
    6. Check if either perturbation violates hard_constraints
    """
    from forge_world.core.evolve import _extract_metrics

    pipeline = runner.pipeline
    original_config = pipeline.get_config()
    schema = pipeline.get_config_schema()
    config_hash = _compute_config_hash(original_config)

    # Get baseline metric
    baseline_report = _run_bench(runner, run_kwargs)
    baseline_metrics = _extract_metrics(baseline_report)
    baseline_value = baseline_metrics.get(target_metric, 0)

    params = walk_numeric_parameters(schema)
    sensitivities: list[ParameterSensitivity] = []

    for param in params:
        path = param["path"]
        try:
            current_value = float(get_nested_value(original_config, path))
        except (KeyError, AttributeError, TypeError, ValueError):
            continue

        # Compute delta
        param_min = param.get("minimum")
        param_max = param.get("maximum")
        if param_min is not None and param_max is not None:
            delta = delta_fraction * (float(param_max) - float(param_min))
        else:
            delta = delta_fraction * max(abs(current_value), 0.1)

        if delta == 0:
            continue

        # Clamp values
        val_minus = current_value - delta
        val_plus = current_value + delta
        if param_min is not None:
            val_minus = max(val_minus, float(param_min))
        if param_max is not None:
            val_plus = min(val_plus, float(param_max))

        # Test minus perturbation
        _set_pipeline_value(pipeline, original_config, path, val_minus)
        minus_report = _run_bench(runner, run_kwargs)
        minus_metrics = _extract_metrics(minus_report)
        metric_minus = minus_metrics.get(target_metric, 0)

        # Test plus perturbation
        _set_pipeline_value(pipeline, original_config, path, val_plus)
        plus_report = _run_bench(runner, run_kwargs)
        plus_metrics = _extract_metrics(plus_report)
        metric_plus = plus_metrics.get(target_metric, 0)

        # Restore original
        pipeline.set_config(dict(original_config))

        impact = abs(metric_plus - metric_minus)

        # Determine direction
        if target_direction == "max":
            direction = "increase" if metric_plus > metric_minus else "decrease"
        else:
            direction = "decrease" if metric_plus < metric_minus else "increase"

        # Check constraints
        constrained = False
        constraint_detail = ""
        if hard_constraints:
            for constraint in hard_constraints:
                c_metric = constraint.get("metric", "")
                c_op = constraint.get("op", "<=")
                c_value = float(constraint.get("value", 0))

                for label, metrics in [("increase", plus_metrics), ("decrease", minus_metrics)]:
                    actual = metrics.get(c_metric)
                    if actual is None:
                        continue
                    violated = False
                    if c_op == "<=" and actual > c_value:
                        violated = True
                    elif c_op == ">=" and actual < c_value:
                        violated = True
                    elif c_op == "<" and actual >= c_value:
                        violated = True
                    elif c_op == ">" and actual <= c_value:
                        violated = True
                    if violated:
                        constrained = True
                        constraint_detail = f"{label} causes {c_metric}={actual:.4f}"

        sensitivities.append(ParameterSensitivity(
            path=path,
            current_value=current_value,
            delta=delta,
            metric_minus=metric_minus,
            metric_plus=metric_plus,
            impact=impact,
            direction=direction,
            constrained=constrained,
            constraint_detail=constraint_detail,
        ))

    # Sort by impact descending
    sensitivities.sort(key=lambda s: s.impact, reverse=True)

    return SensitivityReport(
        parameters=sensitivities,
        target_metric=target_metric,
        baseline_value=baseline_value,
        config_hash=config_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _set_pipeline_value(
    pipeline: Any, original_config: dict[str, Any], path: str, value: float
) -> None:
    """Set a value in the pipeline config, preserving other values."""
    config = dict(original_config)
    set_nested_value(config, path, value)
    pipeline.set_config(config)


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
