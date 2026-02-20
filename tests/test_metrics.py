"""Tests for metrics computation."""

from __future__ import annotations

from forge_world.core.metrics import (
    ConfusionMatrix,
    compute_category_metrics,
    compute_confusion_matrix,
    compute_method_effectiveness,
    find_failure_clusters,
    find_near_misses,
)


class TestConfusionMatrix:
    def test_perfect_classification(self):
        cm = ConfusionMatrix(true_positives=10, true_negatives=5, false_positives=0, false_negatives=0)
        assert cm.sensitivity == 1.0
        assert cm.specificity == 1.0
        assert cm.fpr == 0.0
        assert cm.f1 == 1.0
        assert cm.pass_rate == 1.0

    def test_all_false_negatives(self):
        cm = ConfusionMatrix(true_positives=0, true_negatives=5, false_positives=0, false_negatives=10)
        assert cm.sensitivity == 0.0
        assert cm.specificity == 1.0
        assert cm.fpr == 0.0

    def test_with_false_positives(self):
        cm = ConfusionMatrix(true_positives=8, true_negatives=3, false_positives=2, false_negatives=2)
        assert cm.fpr > 0
        assert cm.sensitivity == 8 / 10
        assert cm.specificity == 3 / 5

    def test_empty(self):
        cm = ConfusionMatrix()
        assert cm.total == 0
        assert cm.sensitivity == 0.0
        assert cm.pass_rate == 0.0

    def test_to_dict(self):
        cm = ConfusionMatrix(true_positives=5, true_negatives=3, false_positives=1, false_negatives=1)
        d = cm.to_dict()
        assert d["tp"] == 5
        assert d["fp"] == 1
        assert "sensitivity" in d
        assert "f1" in d


class TestComputeConfusionMatrix:
    def test_basic(self):
        results = [
            {"expected_label": "findings", "passed": True},
            {"expected_label": "findings", "passed": True},
            {"expected_label": "findings", "passed": False},
            {"expected_label": "clean", "passed": True},
            {"expected_label": "clean", "passed": False},
        ]
        cm = compute_confusion_matrix(results)
        assert cm.true_positives == 2
        assert cm.false_negatives == 1
        assert cm.true_negatives == 1
        assert cm.false_positives == 1

    def test_informational_is_clean(self):
        results = [
            {"expected_label": "informational", "passed": True},
        ]
        cm = compute_confusion_matrix(results)
        assert cm.true_negatives == 1


class TestComputeCategoryMetrics:
    def test_basic(self):
        results = [
            {"category": "synthetic", "expected_label": "findings", "passed": True},
            {"category": "synthetic", "expected_label": "findings", "passed": False},
            {"category": "clean", "expected_label": "clean", "passed": True},
        ]
        metrics = compute_category_metrics(results)
        assert len(metrics) == 2
        synth = next(m for m in metrics if m.category == "synthetic")
        assert synth.passed == 1
        assert synth.failed == 1


class TestComputeMethodEffectiveness:
    def test_basic(self):
        results = [
            {
                "expected_label": "findings",
                "passed": True,
                "findings": [
                    {"method": "ela", "confidence": 0.8},
                    {"method": "clone_detection", "confidence": 0.9},
                ],
            },
            {
                "expected_label": "clean",
                "passed": False,
                "findings": [
                    {"method": "ela", "confidence": 0.6},
                ],
            },
        ]
        metrics = compute_method_effectiveness(results)
        assert "ela" in metrics
        assert "clone_detection" in metrics
        assert metrics["ela"].times_fired == 2
        assert metrics["ela"].contributes_to_fp is True
        assert metrics["clone_detection"].contributes_to_fp is False


class TestFindNearMisses:
    def test_finds_near_boundary(self):
        results = [
            {
                "item_id": "img1",
                "category": "synthetic",
                "expected_label": "findings",
                "passed": True,
                "risk_level": "medium",
                "confidence": 0.55,
                "methods_flagged": ["ela"],
            },
            {
                "item_id": "img2",
                "category": "synthetic",
                "expected_label": "findings",
                "passed": False,
                "risk_level": "low",
                "confidence": 0.45,
                "methods_flagged": ["noise_analysis"],
            },
        ]
        nms = find_near_misses(results, confidence_threshold=0.15)
        assert len(nms) >= 1
        ids = {nm.item_id for nm in nms}
        # Both should be near misses since they're close to boundaries
        assert "img1" in ids or "img2" in ids


class TestFindFailureClusters:
    def test_clusters_by_category(self):
        results = [
            {
                "item_id": "img1",
                "category": "rsiil",
                "passed": False,
                "findings": [{"method": "ela"}],
            },
            {
                "item_id": "img2",
                "category": "rsiil",
                "passed": False,
                "findings": [{"method": "ela"}, {"method": "noise_analysis"}],
            },
            {
                "item_id": "img3",
                "category": "clean",
                "passed": True,
                "findings": [],
            },
        ]
        clusters = find_failure_clusters(results)
        assert len(clusters) >= 1
        cat_cluster = next(
            (c for c in clusters if c.pattern == "category:rsiil"), None
        )
        assert cat_cluster is not None
        assert cat_cluster.count == 2

    def test_no_failures(self):
        results = [
            {"item_id": "img1", "category": "clean", "passed": True, "findings": []},
        ]
        assert find_failure_clusters(results) == []
