"""Benchmark runner: orchestrates pipeline + aggregator + dataset + rules.

Generalizes snoopy's run_demo() loop into a reusable BenchmarkRunner that
produces a structured BenchmarkReport with full metrics.

Supports multi-seed evaluation via ``run_multi()`` to prevent overfitting
to a single dataset sample.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from forge_world.core.metrics import (
    ConfusionMatrix,
    CategoryMetrics,
    MethodEffectiveness,
    compute_category_metrics,
    compute_confusion_matrix,
    compute_method_effectiveness,
)
from forge_world.core.protocols import (
    AggregatedResult,
    Aggregator,
    Finding,
    LabeledDataset,
    PassFailRule,
    PassFailRuleSet,
    Pipeline,
)


@dataclass
class ItemResult:
    """Result of benchmarking a single item."""

    item_id: str
    category: str
    expected_label: str
    passed: bool
    risk_level: str
    confidence: float
    converging_evidence: bool
    findings: list[dict[str, Any]]
    methods_flagged: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "expected_label": self.expected_label,
            "passed": self.passed,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "converging_evidence": self.converging_evidence,
            "findings": self.findings,
            "methods_flagged": self.methods_flagged,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark run report with all metrics."""

    run_id: str
    timestamp: str
    git_sha: str | None
    config_hash: str
    item_results: list[ItemResult]
    confusion_matrix: ConfusionMatrix
    category_metrics: list[CategoryMetrics]
    method_metrics: dict[str, MethodEffectiveness]
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.item_results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.item_results)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def sensitivity(self) -> float:
        return self.confusion_matrix.sensitivity

    @property
    def specificity(self) -> float:
        return self.confusion_matrix.specificity

    @property
    def fpr(self) -> float:
        return self.confusion_matrix.fpr

    @property
    def f1(self) -> float:
        return self.confusion_matrix.f1

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "pass_rate": f"{self.pass_count}/{self.total_count}",
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "fpr": round(self.fpr, 4),
            "f1": round(self.f1, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "config_snapshot": self.config_snapshot,
            "item_results": [r.to_dict() for r in self.item_results],
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "category_metrics": [c.to_dict() for c in self.category_metrics],
            "method_metrics": {k: v.to_dict() for k, v in self.method_metrics.items()},
            "summary": self.summary(),
        }


# --- Multi-seed types ---


@dataclass
class SeedStrategy:
    """Controls which seeds to evaluate for multi-seed benchmarking.

    ``stable_seeds`` are fixed across runs — used for deterministic regression
    detection.  ``n_exploration_seeds`` are derived from the run_id each time
    — used for generalization testing.
    """

    stable_seeds: list[int] = field(default_factory=lambda: [42])
    n_exploration_seeds: int = 1


@dataclass
class SeedReport:
    """Result of a single-seed benchmark run within a multi-seed evaluation."""

    seed: int
    seed_kind: str  # "stable" | "exploration"
    report: BenchmarkReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "seed_kind": self.seed_kind,
            "report": self.report.to_dict(),
        }


@dataclass
class AggregateMetrics:
    """Aggregate statistics across all seed runs."""

    mean_pass_rate: float
    min_pass_rate: float
    max_pass_rate: float
    mean_sensitivity: float
    min_sensitivity: float
    worst_case_fpr: float  # max FPR across ALL seeds — must be 0
    mean_f1: float
    item_stability: dict[str, float]  # item_id -> fraction of seeds where it passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_pass_rate": round(self.mean_pass_rate, 4),
            "min_pass_rate": round(self.min_pass_rate, 4),
            "max_pass_rate": round(self.max_pass_rate, 4),
            "mean_sensitivity": round(self.mean_sensitivity, 4),
            "min_sensitivity": round(self.min_sensitivity, 4),
            "worst_case_fpr": round(self.worst_case_fpr, 4),
            "mean_f1": round(self.mean_f1, 4),
            "item_stability": {
                k: round(v, 4) for k, v in sorted(self.item_stability.items())
            },
        }


@dataclass
class MultiBenchmarkReport:
    """Aggregated report across multiple seed runs."""

    run_id: str
    timestamp: str
    git_sha: str | None
    config_hash: str
    seed_reports: list[SeedReport]
    aggregate_metrics: AggregateMetrics
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_reports(self) -> list[SeedReport]:
        return [sr for sr in self.seed_reports if sr.seed_kind == "stable"]

    @property
    def exploration_reports(self) -> list[SeedReport]:
        return [sr for sr in self.seed_reports if sr.seed_kind == "exploration"]

    @property
    def pass_rate(self) -> float:
        return self.aggregate_metrics.mean_pass_rate

    @property
    def sensitivity(self) -> float:
        return self.aggregate_metrics.mean_sensitivity

    @property
    def fpr(self) -> float:
        return self.aggregate_metrics.worst_case_fpr

    @property
    def f1(self) -> float:
        return self.aggregate_metrics.mean_f1

    def summary(self) -> dict[str, Any]:
        am = self.aggregate_metrics
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "seeds_evaluated": len(self.seed_reports),
            "stable_seeds": [sr.seed for sr in self.stable_reports],
            "exploration_seeds": [sr.seed for sr in self.exploration_reports],
            "mean_pass_rate": round(am.mean_pass_rate, 4),
            "min_pass_rate": round(am.min_pass_rate, 4),
            "max_pass_rate": round(am.max_pass_rate, 4),
            "mean_sensitivity": round(am.mean_sensitivity, 4),
            "worst_case_fpr": round(am.worst_case_fpr, 4),
            "mean_f1": round(am.mean_f1, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "config_snapshot": self.config_snapshot,
            "seed_reports": [sr.to_dict() for sr in self.seed_reports],
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "summary": self.summary(),
        }


# --- Helpers ---


def _get_git_sha() -> str | None:
    """Get current git SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _hash_config(config: Any) -> str:
    """Produce a stable hash of the pipeline config."""
    try:
        if hasattr(config, "model_dump"):
            data = config.model_dump()
        elif hasattr(config, "__dict__"):
            data = config.__dict__
        else:
            data = config
        serialized = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(config)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _apply_pass_fail(
    aggregated: AggregatedResult,
    rule: PassFailRule,
) -> bool:
    """Determine pass/fail based on aggregated result and the rule."""
    if rule.min_risk_for_pass is not None:
        # Item expected to have findings: pass if risk >= min threshold
        return aggregated.risk_level >= rule.min_risk_for_pass
    if rule.max_risk_for_pass is not None:
        # Item expected to be clean: pass if risk <= max threshold
        return aggregated.risk_level <= rule.max_risk_for_pass
    # Default: pass
    return True


def _derive_exploration_seeds(run_id: str, count: int) -> list[int]:
    """Derive deterministic exploration seeds from a run_id."""
    seeds = []
    for i in range(count):
        h = hashlib.sha256(f"{run_id}:exploration:{i}".encode()).hexdigest()
        seeds.append(int(h[:8], 16) % (2**31))
    return seeds


def _compute_aggregate_metrics(seed_reports: list[SeedReport]) -> AggregateMetrics:
    """Compute aggregate statistics across seed runs."""
    if not seed_reports:
        return AggregateMetrics(
            mean_pass_rate=0.0,
            min_pass_rate=0.0,
            max_pass_rate=0.0,
            mean_sensitivity=0.0,
            min_sensitivity=0.0,
            worst_case_fpr=0.0,
            mean_f1=0.0,
            item_stability={},
        )

    pass_rates = [sr.report.pass_rate for sr in seed_reports]
    sensitivities = [sr.report.sensitivity for sr in seed_reports]
    fprs = [sr.report.fpr for sr in seed_reports]
    f1s = [sr.report.f1 for sr in seed_reports]

    # Item stability: for each item, fraction of seeds where it passed
    item_pass_counts: Counter[str] = Counter()
    item_total_counts: Counter[str] = Counter()
    for sr in seed_reports:
        for result in sr.report.item_results:
            item_total_counts[result.item_id] += 1
            if result.passed:
                item_pass_counts[result.item_id] += 1

    item_stability = {
        item_id: item_pass_counts[item_id] / item_total_counts[item_id]
        for item_id in item_total_counts
    }

    return AggregateMetrics(
        mean_pass_rate=sum(pass_rates) / len(pass_rates),
        min_pass_rate=min(pass_rates),
        max_pass_rate=max(pass_rates),
        mean_sensitivity=sum(sensitivities) / len(sensitivities),
        min_sensitivity=min(sensitivities),
        worst_case_fpr=max(fprs),
        mean_f1=sum(f1s) / len(f1s),
        item_stability=item_stability,
    )


class BenchmarkRunner:
    """Runs a full benchmark cycle: pipeline + aggregator + dataset + rules.

    Generalizes snoopy's run_demo() loop.  Supports single-seed runs via
    ``run()`` and multi-seed evaluation via ``run_multi()``.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        aggregator: Aggregator,
        dataset: LabeledDataset,
        rules: PassFailRuleSet,
        *,
        on_item_start: Any | None = None,
        on_item_complete: Any | None = None,
        on_seed_start: Any | None = None,
        on_seed_complete: Any | None = None,
        analysis_cache: Any | None = None,
    ):
        self.pipeline = pipeline
        self.aggregator = aggregator
        self.dataset = dataset
        self.rules = rules
        self.on_item_start = on_item_start
        self.on_item_complete = on_item_complete
        self.on_seed_start = on_seed_start
        self.on_seed_complete = on_seed_complete
        self.analysis_cache = analysis_cache  # AnalysisCache for disk persistence

    def run(
        self,
        *,
        seed: int | None = None,
        sample_size: int | None = None,
        tier: str | None = None,
        item_filter: set[str] | None = None,
        _analysis_cache: dict[str, list[Finding]] | None = None,
    ) -> BenchmarkReport:
        """Execute the benchmark on all dataset items.

        When *seed* is provided, the dataset returns seed-sampled items in
        addition to fixed items.  *sample_size* controls how many random
        items per seed.

        *tier* filters items to only those in the specified tier (requires
        the dataset to implement ``TieredDataset``).

        *item_filter* restricts processing to a specific set of item IDs.

        *_analysis_cache* is an internal cache used by ``run_multi()`` to
        avoid redundant ``pipeline.analyze()`` calls for fixed items that
        appear in every seed run.
        """
        items = self.dataset.items(seed=seed, sample_size=sample_size)

        # Tier filtering
        if tier is not None:
            if not hasattr(self.dataset, "tiers"):
                raise ValueError("Dataset does not support tiers.")
            tier_map = self.dataset.tiers()
            if tier not in tier_map:
                raise ValueError(
                    f"Unknown tier '{tier}'. Available: {sorted(tier_map.keys())}"
                )
            allowed = set(tier_map[tier])
            items = [item for item in items if item.category in allowed]

        # Item filter (e.g. --only-failures)
        if item_filter is not None:
            items = [item for item in items if item.id in item_filter]

        config = self.pipeline.get_config()
        config_hash = _hash_config(config)
        git_sha = _get_git_sha()
        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        item_results: list[ItemResult] = []
        total_items = len(items)

        for idx, item in enumerate(items):
            cached = False
            findings = None

            # Layer 1: in-memory cache (from run_multi)
            if _analysis_cache is not None and item.id in _analysis_cache:
                findings = _analysis_cache[item.id]
                cached = True

            # Layer 2: disk cache
            if findings is None and self.analysis_cache is not None:
                findings = self.analysis_cache.get(config_hash, item.id)
                if findings is not None:
                    cached = True
                    if _analysis_cache is not None:
                        _analysis_cache[item.id] = findings  # promote to in-memory

            if self.on_item_start:
                self.on_item_start(item, idx, total_items, cached)

            # Layer 3: analyze
            if findings is None:
                findings = self.pipeline.analyze(item.data)
                if _analysis_cache is not None:
                    _analysis_cache[item.id] = findings
                if self.analysis_cache is not None:
                    self.analysis_cache.put(config_hash, item.id, findings)

            aggregated = self.aggregator.aggregate(findings)
            rule = self.rules.get_rule(item.expected_label)
            passed = _apply_pass_fail(aggregated, rule)

            result = ItemResult(
                item_id=item.id,
                category=item.category,
                expected_label=item.expected_label,
                passed=passed,
                risk_level=aggregated.risk_level.value,
                confidence=aggregated.overall_confidence,
                converging_evidence=aggregated.converging_evidence,
                findings=[f.to_dict() for f in findings],
                methods_flagged=sorted(aggregated.methods_flagged),
                metadata=item.metadata,
            )
            item_results.append(result)

            if self.on_item_complete:
                self.on_item_complete(result)

        # Compute metrics from flat dicts
        result_dicts = [r.to_dict() for r in item_results]
        confusion = compute_confusion_matrix(result_dicts)
        cat_metrics = compute_category_metrics(result_dicts)
        method_metrics = compute_method_effectiveness(result_dicts)

        # Serialize config snapshot
        try:
            if hasattr(config, "model_dump"):
                config_snapshot = config.model_dump()
            elif hasattr(config, "__dict__"):
                config_snapshot = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
            else:
                config_snapshot = {"raw": str(config)}
        except Exception:
            config_snapshot = {}

        return BenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            git_sha=git_sha,
            config_hash=config_hash,
            item_results=item_results,
            confusion_matrix=confusion,
            category_metrics=cat_metrics,
            method_metrics=method_metrics,
            config_snapshot=config_snapshot,
        )

    def run_multi(
        self,
        seed_strategy: SeedStrategy | None = None,
        *,
        sample_size: int | None = None,
        tier: str | None = None,
        item_filter: set[str] | None = None,
    ) -> MultiBenchmarkReport:
        """Run benchmark across multiple seeds and aggregate results.

        Fixed items (those returned by ``dataset.items(seed=None)``) are
        analyzed once and cached — only seed-sampled items require fresh
        analysis per seed.
        """
        if seed_strategy is None:
            seed_strategy = SeedStrategy()

        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        git_sha = _get_git_sha()
        config = self.pipeline.get_config()
        config_hash = _hash_config(config)

        exploration_seeds = _derive_exploration_seeds(
            run_id, seed_strategy.n_exploration_seeds
        )

        all_seeds: list[tuple[int, str]] = [
            (s, "stable") for s in seed_strategy.stable_seeds
        ] + [
            (s, "exploration") for s in exploration_seeds
        ]

        # Cache pipeline.analyze() results by item_id to skip redundant work.
        # Fixed items appear in every seed run and produce identical findings.
        analysis_cache: dict[str, list[Finding]] = {}

        seed_reports: list[SeedReport] = []
        for seed_idx, (seed, kind) in enumerate(all_seeds):
            if self.on_seed_start:
                self.on_seed_start(seed, kind, seed_idx, len(all_seeds))
            report = self.run(
                seed=seed,
                sample_size=sample_size,
                tier=tier,
                item_filter=item_filter,
                _analysis_cache=analysis_cache,
            )
            seed_reports.append(SeedReport(seed=seed, seed_kind=kind, report=report))
            if self.on_seed_complete:
                self.on_seed_complete(seed, kind, seed_idx, len(all_seeds), report)

        aggregate = _compute_aggregate_metrics(seed_reports)

        # Serialize config snapshot
        try:
            if hasattr(config, "model_dump"):
                config_snapshot = config.model_dump()
            elif hasattr(config, "__dict__"):
                config_snapshot = {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                }
            else:
                config_snapshot = {"raw": str(config)}
        except Exception:
            config_snapshot = {}

        return MultiBenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            git_sha=git_sha,
            config_hash=config_hash,
            seed_reports=seed_reports,
            aggregate_metrics=aggregate,
            config_snapshot=config_snapshot,
        )
