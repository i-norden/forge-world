"""Metrics computation: confusion matrix, per-category, per-method effectiveness.

Includes near-miss detection and failure clustering for actionable AI feedback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any



@dataclass
class ConfusionMatrix:
    """Binary confusion matrix for pass/fail classification."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positives + self.true_negatives
            + self.false_positives + self.false_negatives
        )

    @property
    def sensitivity(self) -> float:
        """True positive rate (recall). TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        """True negative rate. TN / (TN + FP)."""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False positive rate. FP / (FP + TN)."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.sensitivity
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def pass_rate(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "fpr": round(self.fpr, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class CategoryMetrics:
    """Metrics for a single category of test items."""

    category: str
    total: int
    passed: int
    failed: int
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "confusion": self.confusion.to_dict(),
        }


@dataclass
class MethodEffectiveness:
    """Effectiveness metrics for a single analysis method."""

    method: str
    times_fired: int = 0
    true_detections: int = 0
    false_detections: int = 0
    items_with_findings: int = 0
    avg_confidence: float = 0.0
    contributes_to_fp: bool = False

    @property
    def detection_rate(self) -> float:
        total = self.true_detections + self.false_detections
        return self.true_detections / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "times_fired": self.times_fired,
            "true_detections": self.true_detections,
            "false_detections": self.false_detections,
            "avg_confidence": round(self.avg_confidence, 4),
            "detection_rate": round(self.detection_rate, 4),
            "contributes_to_fp": self.contributes_to_fp,
        }


@dataclass
class NearMiss:
    """An item close to the pass/fail boundary -- likely to flip with small changes."""

    item_id: str
    category: str
    expected_label: str
    passed: bool
    risk_level: str
    confidence: float
    distance_to_boundary: float
    methods_flagged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "expected_label": self.expected_label,
            "passed": self.passed,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "distance_to_boundary": round(self.distance_to_boundary, 4),
            "methods_flagged": self.methods_flagged,
        }


@dataclass
class FailureCluster:
    """A group of failures sharing common characteristics."""

    pattern: str
    count: int
    item_ids: list[str] = field(default_factory=list)
    common_methods: list[str] = field(default_factory=list)
    common_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "item_ids": self.item_ids,
            "common_methods": self.common_methods,
            "common_category": self.common_category,
        }


def compute_confusion_matrix(
    results: list[dict[str, Any]],
) -> ConfusionMatrix:
    """Compute confusion matrix from item results.

    Each result dict must have:
        - expected_label: str ('findings', 'clean', etc.)
        - passed: bool
    """
    cm = ConfusionMatrix()
    for r in results:
        expected = r["expected_label"]
        passed = r["passed"]
        expects_findings = expected not in ("clean", "informational")
        if expects_findings:
            if passed:
                cm.true_positives += 1
            else:
                cm.false_negatives += 1
        else:
            if passed:
                cm.true_negatives += 1
            else:
                cm.false_positives += 1
    return cm


def compute_category_metrics(
    results: list[dict[str, Any]],
) -> list[CategoryMetrics]:
    """Compute per-category metrics from item results."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r)

    metrics = []
    for cat, items in sorted(by_category.items()):
        passed = sum(1 for r in items if r["passed"])
        cm = compute_confusion_matrix(items)
        metrics.append(
            CategoryMetrics(
                category=cat,
                total=len(items),
                passed=passed,
                failed=len(items) - passed,
                confusion=cm,
            )
        )
    return metrics


def compute_method_effectiveness(
    results: list[dict[str, Any]],
) -> dict[str, MethodEffectiveness]:
    """Compute per-method effectiveness from item results.

    Each result dict must have:
        - expected_label: str
        - passed: bool
        - findings: list[dict] with 'method' and 'confidence' keys
    """
    method_stats: dict[str, MethodEffectiveness] = {}

    for r in results:
        expected = r["expected_label"]
        expects_findings = expected not in ("clean", "informational")
        is_false_positive = not expects_findings and not r["passed"]

        methods_seen: set[str] = set()
        for f in r.get("findings", []):
            method = f.get("method", "")
            if not method:
                continue

            if method not in method_stats:
                method_stats[method] = MethodEffectiveness(method=method)

            me = method_stats[method]
            me.times_fired += 1

            if method not in methods_seen:
                methods_seen.add(method)
                me.items_with_findings += 1

            if expects_findings:
                me.true_detections += 1
            else:
                me.false_detections += 1
                if is_false_positive:
                    me.contributes_to_fp = True

        # Update avg confidence per method
        for method in methods_seen:
            findings_for_method = [
                float(f.get("confidence", 0.0))
                for f in r.get("findings", [])
                if f.get("method") == method
            ]
            if findings_for_method:
                me = method_stats[method]
                # Running average
                old_total = me.avg_confidence * (me.items_with_findings - 1)
                new_avg = sum(findings_for_method) / len(findings_for_method)
                me.avg_confidence = (old_total + new_avg) / me.items_with_findings

    return method_stats


def find_near_misses(
    results: list[dict[str, Any]],
    confidence_threshold: float = 0.15,
) -> list[NearMiss]:
    """Find items close to the pass/fail boundary.

    Near misses are items whose aggregated confidence is within
    `confidence_threshold` of the decision boundary. These are the items
    most likely to flip with small parameter changes.
    """
    near_misses = []
    for r in results:
        confidence = float(r.get("confidence", 0.0))
        risk_level = r.get("risk_level", "clean")
        expected = r["expected_label"]
        passed = r["passed"]

        # For items expected to have findings: boundary is at medium risk
        # For clean items: boundary is at high risk
        expects_findings = expected not in ("clean", "informational")

        if expects_findings:
            # Distance: how far from becoming medium risk
            severity_scores = {"clean": 0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1}
            score = severity_scores.get(risk_level, 0)
            boundary = 0.5  # medium
            distance = abs(score - boundary)
        else:
            # Distance: how far from becoming a false positive
            severity_scores = {"clean": 0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1}
            score = severity_scores.get(risk_level, 0)
            boundary = 0.75  # high
            distance = abs(score - boundary)

        if distance <= confidence_threshold * 2 or abs(confidence - 0.5) <= confidence_threshold:
            near_misses.append(
                NearMiss(
                    item_id=r["item_id"],
                    category=r["category"],
                    expected_label=expected,
                    passed=passed,
                    risk_level=risk_level,
                    confidence=confidence,
                    distance_to_boundary=distance,
                    methods_flagged=r.get("methods_flagged", []),
                )
            )

    near_misses.sort(key=lambda nm: nm.distance_to_boundary)
    return near_misses


def find_failure_clusters(
    results: list[dict[str, Any]],
) -> list[FailureCluster]:
    """Group failures by common patterns (category, methods, risk level).

    Identifies clusters of failing items that share characteristics,
    helping the agent understand systemic issues vs one-off failures.
    """
    failures = [r for r in results if not r["passed"]]
    if not failures:
        return []

    clusters: list[FailureCluster] = []

    # Cluster by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for f in failures:
        cat = f["category"]
        by_category.setdefault(cat, []).append(f)

    for cat, items in by_category.items():
        if len(items) >= 2:
            # Find common methods across failures in this category
            method_counts: Counter[str] = Counter()
            for item in items:
                for finding in item.get("findings", []):
                    m = finding.get("method", "")
                    if m:
                        method_counts[m] += 1
            common = [m for m, c in method_counts.most_common(3) if c >= len(items) // 2]
            clusters.append(
                FailureCluster(
                    pattern=f"category:{cat}",
                    count=len(items),
                    item_ids=[i["item_id"] for i in items],
                    common_methods=common,
                    common_category=cat,
                )
            )

    # Cluster by method combination (items failing with the same set of methods)
    method_combos: dict[tuple[str, ...], list[str]] = {}
    for f in failures:
        methods = sorted({
            finding.get("method", "")
            for finding in f.get("findings", [])
            if finding.get("method")
        })
        if methods:
            key = tuple(methods)
            method_combos.setdefault(key, []).append(f["item_id"])

    for methods, item_ids in method_combos.items():
        if len(item_ids) >= 2:
            clusters.append(
                FailureCluster(
                    pattern=f"methods:{'+'.join(methods)}",
                    count=len(item_ids),
                    item_ids=item_ids,
                    common_methods=list(methods),
                )
            )

    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters
