"""AI agent feedback interface.

Produces structured context optimized for LLM reasoning. The EvolutionContext
contains everything an AI agent needs to decide what to change next:
failure analysis, method effectiveness, near-misses, config schema, and
regression data.

Supports multi-seed reports with seed variance and item stability info.
Includes iteration memory, sensitivity analysis, exploration state, and
enhanced failure clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge_world.core.metrics import (
    find_actionable_failure_clusters,
    find_near_misses,
)
from forge_world.core.runner import BenchmarkReport, MultiBenchmarkReport
from forge_world.core.snapshots import RegressionReport


@dataclass
class EvolutionContext:
    """Structured context for AI agent consumption.

    Contains all the information an LLM needs to decide what pipeline
    changes to make next, formatted for efficient reasoning.
    """

    # Current state
    current_metrics: dict[str, float] = field(default_factory=dict)
    category_breakdown: list[dict[str, Any]] = field(default_factory=list)

    # Change impact (vs baseline)
    items_regressed: list[dict[str, Any]] = field(default_factory=list)
    items_improved: list[dict[str, Any]] = field(default_factory=list)

    # Failure analysis
    failing_items: list[dict[str, Any]] = field(default_factory=list)
    near_miss_items: list[dict[str, Any]] = field(default_factory=list)
    failure_clusters: list[dict[str, Any]] = field(default_factory=list)

    # Method analysis
    method_effectiveness: list[dict[str, Any]] = field(default_factory=list)
    methods_never_firing: list[str] = field(default_factory=list)
    methods_causing_fp: list[dict[str, Any]] = field(default_factory=list)

    # Configuration
    current_config: dict[str, Any] = field(default_factory=dict)
    config_schema: dict[str, Any] = field(default_factory=dict)

    # History
    recent_runs: list[dict[str, Any]] = field(default_factory=list)
    change_log: list[dict[str, Any]] = field(default_factory=list)

    # Constraints
    hard_constraints: list[dict[str, Any]] = field(default_factory=list)
    optimization_target: dict[str, Any] = field(default_factory=dict)

    # Multi-seed context
    seed_variance: dict[str, Any] = field(default_factory=dict)
    item_stability: list[dict[str, Any]] = field(default_factory=list)
    sample_size: int | None = None

    # Domain-specific diagnostics
    diagnostic_clusters: list[dict[str, Any]] = field(default_factory=list)

    # New: memory, sensitivity, exploration, evolve mode
    memory_summary: str = ""
    sensitivity_summary: str = ""
    exploration_state: dict[str, Any] = field(default_factory=dict)
    evolve_mode: bool = False
    validation_errors: str = ""
    performance_summary: str = ""
    tuner_summary: str = ""
    optimization_targets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "current_metrics": self.current_metrics,
            "category_breakdown": self.category_breakdown,
            "items_regressed": self.items_regressed,
            "items_improved": self.items_improved,
            "failing_items": self.failing_items,
            "near_miss_items": self.near_miss_items,
            "failure_clusters": self.failure_clusters,
            "method_effectiveness": self.method_effectiveness,
            "methods_never_firing": self.methods_never_firing,
            "methods_causing_fp": self.methods_causing_fp,
            "current_config": self.current_config,
            "config_schema": self.config_schema,
            "recent_runs": self.recent_runs,
            "change_log": self.change_log,
            "hard_constraints": self.hard_constraints,
            "optimization_target": self.optimization_target,
        }
        if self.seed_variance:
            d["seed_variance"] = self.seed_variance
        if self.item_stability:
            d["item_stability"] = self.item_stability
        if self.sample_size is not None:
            d["sample_size"] = self.sample_size
        if self.diagnostic_clusters:
            d["diagnostic_clusters"] = self.diagnostic_clusters
        if self.memory_summary:
            d["memory_summary"] = self.memory_summary
        if self.sensitivity_summary:
            d["sensitivity_summary"] = self.sensitivity_summary
        if self.exploration_state:
            d["exploration_state"] = self.exploration_state
        d["evolve_mode"] = self.evolve_mode
        if self.validation_errors:
            d["validation_errors"] = self.validation_errors
        if self.performance_summary:
            d["performance_summary"] = self.performance_summary
        if self.tuner_summary:
            d["tuner_summary"] = self.tuner_summary
        return d

    def to_prompt_context(self) -> str:
        """Serialize to markdown optimized for LLM consumption."""
        lines: list[str] = []

        # 1. Header
        lines.append("# Evolution Context")
        lines.append("")

        # 1b. Validation errors from last iteration
        if self.validation_errors:
            lines.append(self.validation_errors)

        # 2. Current metrics
        lines.append("## Current Performance")
        m = self.current_metrics
        lines.append(f"- Pass rate: {m.get('pass_rate', 'N/A')}")
        lines.append(f"- Sensitivity (TPR): {m.get('sensitivity', 'N/A')}")
        lines.append(f"- Specificity (TNR): {m.get('specificity', 'N/A')}")
        lines.append(f"- False positive rate: {m.get('fpr', 'N/A')}")
        lines.append(f"- F1 score: {m.get('f1', 'N/A')}")
        if self.sample_size is not None:
            lines.append(f"- Sample size (M): {self.sample_size}")
        lines.append("")

        # 3. Iteration History (from memory)
        if self.memory_summary:
            lines.append(self.memory_summary)

        # 4. Parameter Sensitivity (from sensitivity)
        if self.sensitivity_summary:
            lines.append(self.sensitivity_summary)

        # 4b. Auto-tuning results
        if self.tuner_summary:
            lines.append(self.tuner_summary)

        # 4c. Performance
        if self.performance_summary:
            lines.append(self.performance_summary)

        # 5. Exploration State
        if self.exploration_state:
            lines.append("## Exploration State")
            es = self.exploration_state
            lines.append(f"- Temperature: {es.get('temperature', 0):.2f} (decays each iteration)")
            lines.append(
                f"- Exploration budget: {es.get('budget_remaining', 0)}"
                f"/{es.get('budget_total', 0)} remaining"
            )
            lines.append(
                f"- Convergence: {es.get('metric_history_len', 0)} of "
                f"{es.get('convergence_window', 5)} window slots filled"
            )
            lines.append("")

        # 6. Seed variance section (multi-seed)
        if self.seed_variance:
            lines.append("## Seed Variance")
            sv = self.seed_variance
            if "per_seed" in sv:
                for entry in sv["per_seed"]:
                    kind = entry.get("kind", "?")
                    seed = entry.get("seed", "?")
                    pr = entry.get("pass_rate", 0)
                    lines.append(f"- Seed {seed} ({kind}): pass_rate={pr:.1%}")
            if "range" in sv:
                r = sv["range"]
                lines.append(
                    f"- Range: {r.get('min', 0):.1%} - {r.get('max', 0):.1%} "
                    f"(mean: {r.get('mean', 0):.1%})"
                )
            lines.append("")

        # 7. Item stability (multi-seed)
        if self.item_stability:
            lines.append("## Unstable Items (vary across seeds)")
            lines.append(
                "These items pass on some seeds but fail on others — "
                "they are near the decision boundary:"
            )
            for item in self.item_stability:
                lines.append(f"- `{item['item_id']}`: passes {item['stability']:.0%} of seeds")
            lines.append("")

        # 8. Category breakdown
        if self.category_breakdown:
            lines.append("## Category Breakdown")
            for cat in self.category_breakdown:
                lines.append(
                    f"- **{cat['category']}**: {cat['passed']}/{cat['total']} "
                    f"({cat.get('pass_rate', 0):.0%})"
                )
            lines.append("")

        # 9. Regression info
        if self.items_regressed:
            lines.append("## Regressions (CRITICAL)")
            lines.append(f"**{len(self.items_regressed)} items regressed** from baseline:")
            for item in self.items_regressed:
                seed_info = f" [seed={item['seed']}]" if item.get("seed") is not None else ""
                lines.append(
                    f"- `{item['item_id']}` ({item['category']}): "
                    f"{item['baseline_risk']} -> {item['current_risk']}{seed_info}"
                )
            lines.append("")

        if self.items_improved:
            lines.append("## Improvements")
            lines.append(f"**{len(self.items_improved)} items improved** from baseline:")
            for item in self.items_improved:
                lines.append(
                    f"- `{item['item_id']}` ({item['category']}): "
                    f"{item['baseline_risk']} -> {item['current_risk']}"
                )
            lines.append("")

        # 10. Failing items
        if self.failing_items:
            lines.append("## Failing Items")
            lines.append(f"{len(self.failing_items)} items failing:")
            for item in self.failing_items:
                methods = ", ".join(item.get("methods_flagged", []))
                lines.append(
                    f"- `{item['item_id']}` ({item['category']}, "
                    f"expected={item['expected_label']}): "
                    f"risk={item['risk_level']}, conf={item.get('confidence', 0):.2f}"
                )
                if methods:
                    lines.append(f"  Methods: {methods}")
            lines.append("")

        # Near misses
        if self.near_miss_items:
            lines.append("## Near Misses (likely to flip)")
            for nm in self.near_miss_items:
                status = "PASSING" if nm["passed"] else "FAILING"
                lines.append(
                    f"- `{nm['item_id']}` [{status}]: "
                    f"risk={nm['risk_level']}, "
                    f"distance_to_boundary={nm['distance_to_boundary']:.3f}"
                )
            lines.append("")

        # 11. Enhanced Failure Patterns
        if self.failure_clusters:
            lines.append("## Failure Patterns")
            for cluster in self.failure_clusters:
                cluster_type = cluster.get("cluster_type", "")
                achievable = cluster.get("achievable", True)

                if cluster_type in ("no_signal", "below_threshold", "single_method_capped"):
                    # Enhanced actionable cluster
                    tag = "[ACHIEVABLE]" if achievable else "[NOT ACHIEVABLE]"
                    lines.append(f"### {cluster['pattern']} ({cluster['count']} items) {tag}")
                    if cluster.get("counterfactual"):
                        lines.append(cluster["counterfactual"])
                    if cluster.get("suggestion"):
                        lines.append(f"Suggestion: {cluster['suggestion']}")
                    if cluster.get("item_ids"):
                        items_str = ", ".join(cluster["item_ids"][:10])
                        if len(cluster["item_ids"]) > 10:
                            items_str += f" (+{len(cluster['item_ids']) - 10} more)"
                        lines.append(f"Items: {items_str}")
                    lines.append("")
                else:
                    # Standard cluster
                    methods = ", ".join(cluster.get("common_methods", []))
                    lines.append(f"- **{cluster['pattern']}** ({cluster['count']} items)")
                    if methods:
                        lines.append(f"  Common methods: {methods}")
            lines.append("")

        # 12. Method effectiveness
        if self.method_effectiveness:
            lines.append("## Method Effectiveness")
            for me in self.method_effectiveness:
                lines.append(
                    f"- **{me['method']}**: fired {me['times_fired']}x, "
                    f"detection_rate={me['detection_rate']:.2f}, "
                    f"avg_conf={me['avg_confidence']:.2f}"
                    + (" [CAUSES FP]" if me.get("contributes_to_fp") else "")
                )
            lines.append("")

        if self.methods_never_firing:
            lines.append("### Methods Never Firing")
            for m_name in self.methods_never_firing:
                lines.append(f"- {m_name}")
            lines.append("")

        if self.methods_causing_fp:
            lines.append("### Methods Causing False Positives")
            for m_fp in self.methods_causing_fp:
                lines.append(f"- **{m_fp['method']}**: {m_fp['false_detections']} false detections")
            lines.append("")

        # 13. Domain-specific diagnostics
        if self.diagnostic_clusters:
            lines.append("## Domain-Specific Diagnosis")
            for cluster in self.diagnostic_clusters:
                achievable_tag = "ACHIEVABLE" if cluster.get("achievable") else "NOT ACHIEVABLE"
                lines.append(
                    f"- **{cluster['label']}** [{achievable_tag}] "
                    f"({len(cluster.get('item_ids', []))} items)"
                )
                lines.append(f"  {cluster['description']}")
                lines.append(f"  Suggested: {cluster['suggested_action']}")
            lines.append("")

        # 14. Configuration
        if self.current_config:
            lines.append("## Current Configuration")
            lines.append("```json")
            import json

            lines.append(json.dumps(self.current_config, indent=2, default=str))
            lines.append("```")
            lines.append("")

        # 15. Constraints
        if self.hard_constraints:
            lines.append("## Hard Constraints")
            for c in self.hard_constraints:
                lines.append(f"- {c['metric']} {c['op']} {c['value']}")
            lines.append("")

        if self.optimization_targets:
            lines.append("## Optimization Targets")
            for t in self.optimization_targets:
                verb = "Maximize" if t.get("direction", "max") == "max" else "Minimize"
                lines.append(f"- {verb} **{t.get('metric', '?')}**")
            lines.append("")
        elif self.optimization_target:
            lines.append("## Optimization Target")
            lines.append(
                f"Maximize **{self.optimization_target.get('metric', '?')}** "
                f"(direction: {self.optimization_target.get('direction', 'max')})"
            )
            lines.append("")

        # 16. How to Make Changes (only in evolve_mode)
        if self.evolve_mode:
            lines.append("## How to Make Changes")
            lines.append("")
            lines.append("### Option 1: Parameter Proposals (recommended for tuning)")
            lines.append("Create `.forge-world/parameter-proposal.json`:")
            lines.append("```json")
            lines.append("{")
            lines.append('  "proposals": [')
            lines.append(
                '    {"parameter_path": "weight_ela", "new_value": 0.75, '
                '"reasoning": "High sensitivity impact"},'
            )
            lines.append(
                '    {"parameter_path": "convergence_confidence_threshold", '
                '"new_value": 0.55, "reasoning": "Recover near-misses"}'
            )
            lines.append("  ],")
            lines.append('  "agent_notes": "Focusing on top-2 sensitivity parameters"')
            lines.append("}")
            lines.append("```")
            lines.append("")
            lines.append("### Option 2: File Editing (for structural/algorithmic changes)")
            lines.append("Edit source files directly. Both options can be combined.")
            lines.append("")
            lines.append("### Auto-Tuning")
            lines.append("Numeric parameters are automatically optimized via Optuna after your")
            lines.append("structural changes are accepted (when enabled). Focus on algorithmic")
            lines.append("and structural improvements.")
            lines.append("")

        return "\n".join(lines)


def build_evolution_context(
    report: BenchmarkReport | MultiBenchmarkReport,
    regression: RegressionReport | None = None,
    pipeline_config_schema: dict[str, Any] | None = None,
    hard_constraints: list[dict[str, Any]] | None = None,
    optimization_target: dict[str, Any] | None = None,
    optimization_targets: list[dict[str, Any]] | None = None,
    sample_size: int | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    memory: Any | None = None,
    sensitivity: Any | None = None,
    evolve_mode: bool = False,
    exploration_state: dict[str, Any] | None = None,
    validation_errors: str = "",
    tuner_result: Any | None = None,
) -> EvolutionContext:
    """Build an EvolutionContext from a benchmark report and optional regression data.

    Accepts either a single-seed ``BenchmarkReport`` or a ``MultiBenchmarkReport``.
    Supports both singular ``optimization_target`` (backward compat) and plural
    ``optimization_targets`` (multi-objective).
    """
    # Normalize: prefer optimization_targets if provided
    resolved_targets = optimization_targets
    if resolved_targets is None and optimization_target is not None:
        resolved_targets = [optimization_target]

    if isinstance(report, MultiBenchmarkReport):
        ctx = _build_multi_context(
            report,
            regression,
            pipeline_config_schema,
            hard_constraints,
            resolved_targets,
            sample_size,
        )
    else:
        ctx = _build_single_context(
            report,
            regression,
            pipeline_config_schema,
            hard_constraints,
            resolved_targets,
            sample_size,
        )
    if diagnostics:
        ctx.diagnostic_clusters = diagnostics

    # Set multi-target list if more than one target
    if resolved_targets and len(resolved_targets) > 1:
        ctx.optimization_targets = resolved_targets

    # Add memory summary
    if memory is not None:
        ctx.memory_summary = memory.to_prompt_context()

    # Add sensitivity summary
    if sensitivity is not None:
        ctx.sensitivity_summary = sensitivity.to_prompt_context()

    # Set evolve mode and exploration state
    ctx.evolve_mode = evolve_mode
    if exploration_state:
        ctx.exploration_state = exploration_state
    if validation_errors:
        ctx.validation_errors = validation_errors

    # Add performance summary
    perf = getattr(report, "performance", None)
    if perf is not None:
        lines = ["## Performance"]
        lines.append(f"- Mean latency: {perf.latency_mean_ms:.1f}ms")
        lines.append(f"- P95 latency: {perf.latency_p95_ms:.1f}ms")
        lines.append(f"- Throughput: {perf.throughput_items_per_sec:.1f} items/sec")
        lines.append(f"- Items analyzed: {perf.item_count}")
        lines.append("")
        ctx.performance_summary = "\n".join(lines)

    # Add tuner summary
    if tuner_result is not None and hasattr(tuner_result, "to_prompt_context"):
        ctx.tuner_summary = tuner_result.to_prompt_context()

    return ctx


def _build_single_context(
    report: BenchmarkReport,
    regression: RegressionReport | None,
    pipeline_config_schema: dict[str, Any] | None,
    hard_constraints: list[dict[str, Any]] | None,
    optimization_targets: list[dict[str, Any]] | None,
    sample_size: int | None,
) -> EvolutionContext:
    """Build context from a single-seed report."""
    current_metrics: dict[str, Any] = {
        "pass_rate": f"{report.pass_count}/{report.total_count}",
        "pass_rate_pct": round(report.pass_rate, 4),
        "sensitivity": round(report.sensitivity, 4),
        "specificity": round(report.specificity, 4),
        "fpr": round(report.fpr, 4),
        "f1": round(report.f1, 4),
    }

    category_breakdown = [c.to_dict() for c in report.category_metrics]

    result_dicts = [r.to_dict() for r in report.item_results]
    failing_items = [r for r in result_dicts if not r["passed"]]

    near_misses = find_near_misses(result_dicts)
    near_miss_dicts = [nm.to_dict() for nm in near_misses]

    # Use actionable clusters (superset of existing)
    actionable_clusters = find_actionable_failure_clusters(result_dicts, report.config_snapshot)
    cluster_dicts = [c.to_dict() for c in actionable_clusters]

    method_metrics = [me.to_dict() for me in report.method_metrics.values()]
    method_metrics.sort(key=lambda m: m["times_fired"], reverse=True)

    methods_never_firing: list[str] = []
    methods_causing_fp = [me for me in method_metrics if me.get("contributes_to_fp")]

    items_regressed: list[dict[str, Any]] = []
    items_improved: list[dict[str, Any]] = []
    if regression:
        items_regressed = [r.to_dict() for r in regression.regressions]
        items_improved = [i.to_dict() for i in regression.improvements]

    return EvolutionContext(
        current_metrics=current_metrics,
        category_breakdown=category_breakdown,
        items_regressed=items_regressed,
        items_improved=items_improved,
        failing_items=failing_items,
        near_miss_items=near_miss_dicts,
        failure_clusters=cluster_dicts,
        method_effectiveness=method_metrics,
        methods_never_firing=methods_never_firing,
        methods_causing_fp=methods_causing_fp,
        current_config=report.config_snapshot,
        config_schema=pipeline_config_schema or {},
        hard_constraints=hard_constraints or [],
        optimization_target=(
            optimization_targets or [{"metric": "sensitivity", "direction": "max"}]
        )[0],
        sample_size=sample_size,
    )


def _build_multi_context(
    report: MultiBenchmarkReport,
    regression: RegressionReport | None,
    pipeline_config_schema: dict[str, Any] | None,
    hard_constraints: list[dict[str, Any]] | None,
    optimization_targets: list[dict[str, Any]] | None,
    sample_size: int | None,
) -> EvolutionContext:
    """Build context from a multi-seed report."""
    am = report.aggregate_metrics

    current_metrics: dict[str, Any] = {
        "pass_rate": f"{am.mean_pass_rate:.1%}",
        "pass_rate_pct": round(am.mean_pass_rate, 4),
        "min_pass_rate": round(am.min_pass_rate, 4),
        "max_pass_rate": round(am.max_pass_rate, 4),
        "sensitivity": round(am.mean_sensitivity, 4),
        "min_sensitivity": round(am.min_sensitivity, 4),
        "fpr": round(am.worst_case_fpr, 4),
        "f1": round(am.mean_f1, 4),
        "seeds_evaluated": len(report.seed_reports),
    }

    # Use first stable seed for detailed analysis (category breakdown, failures, etc.)
    stable = report.stable_reports
    primary = stable[0].report if stable else report.seed_reports[0].report

    category_breakdown = [c.to_dict() for c in primary.category_metrics]

    result_dicts = [r.to_dict() for r in primary.item_results]
    failing_items = [r for r in result_dicts if not r["passed"]]

    near_misses = find_near_misses(result_dicts)
    near_miss_dicts = [nm.to_dict() for nm in near_misses]

    # Use actionable clusters (superset of existing)
    actionable_clusters = find_actionable_failure_clusters(result_dicts, report.config_snapshot)
    cluster_dicts = [c.to_dict() for c in actionable_clusters]

    method_metrics = [me.to_dict() for me in primary.method_metrics.values()]
    method_metrics.sort(key=lambda m: m["times_fired"], reverse=True)

    methods_never_firing: list[str] = []
    methods_causing_fp = [me for me in method_metrics if me.get("contributes_to_fp")]

    items_regressed: list[dict[str, Any]] = []
    items_improved: list[dict[str, Any]] = []
    if regression:
        items_regressed = [r.to_dict() for r in regression.regressions]
        items_improved = [i.to_dict() for i in regression.improvements]

    # Seed variance: per-seed pass rates
    seed_variance: dict[str, Any] = {
        "per_seed": [
            {
                "seed": sr.seed,
                "kind": sr.seed_kind,
                "pass_rate": round(sr.report.pass_rate, 4),
                "sensitivity": round(sr.report.sensitivity, 4),
                "fpr": round(sr.report.fpr, 4),
            }
            for sr in report.seed_reports
        ],
        "range": {
            "min": round(am.min_pass_rate, 4),
            "max": round(am.max_pass_rate, 4),
            "mean": round(am.mean_pass_rate, 4),
        },
    }

    # Item stability: items that are unstable across seeds
    item_stability_list = [
        {"item_id": iid, "stability": round(stab, 4)}
        for iid, stab in sorted(am.item_stability.items())
        if 0 < stab < 1
    ]

    return EvolutionContext(
        current_metrics=current_metrics,
        category_breakdown=category_breakdown,
        items_regressed=items_regressed,
        items_improved=items_improved,
        failing_items=failing_items,
        near_miss_items=near_miss_dicts,
        failure_clusters=cluster_dicts,
        method_effectiveness=method_metrics,
        methods_never_firing=methods_never_firing,
        methods_causing_fp=methods_causing_fp,
        current_config=report.config_snapshot,
        config_schema=pipeline_config_schema or {},
        hard_constraints=hard_constraints or [],
        optimization_target=(
            optimization_targets or [{"metric": "sensitivity", "direction": "max"}]
        )[0],
        seed_variance=seed_variance,
        item_stability=item_stability_list,
        sample_size=sample_size,
    )
