"""Tests for core protocol types and data classes."""

from __future__ import annotations

from forge_world.core.protocols import (
    AggregatedResult,
    DiagnosticCluster,
    Diagnostics,
    Finding,
    FitnessScore,
    LabeledDataset,
    LabeledItem,
    PassFailRule,
    Severity,
    TieredDataset,
)


class TestSeverity:
    def test_ordering(self):
        assert Severity.CLEAN < Severity.LOW
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL

    def test_comparison_operators(self):
        assert Severity.MEDIUM >= Severity.MEDIUM
        assert Severity.HIGH >= Severity.MEDIUM
        assert Severity.LOW <= Severity.MEDIUM
        assert not Severity.HIGH < Severity.LOW

    def test_equality(self):
        assert Severity.MEDIUM == Severity.MEDIUM
        assert Severity.HIGH != Severity.LOW


class TestFinding:
    def test_create(self):
        f = Finding(
            title="Test finding",
            method="ela",
            severity=Severity.HIGH,
            confidence=0.85,
            description="ELA anomaly detected",
            item_id="figure_1.png",
            evidence={"max_diff": 65.0},
        )
        assert f.title == "Test finding"
        assert f.severity == Severity.HIGH
        assert f.confidence == 0.85

    def test_to_dict(self):
        f = Finding(
            title="Test",
            method="ela",
            severity=Severity.MEDIUM,
            confidence=0.5,
        )
        d = f.to_dict()
        assert d["severity"] == "medium"
        assert d["method"] == "ela"
        assert d["confidence"] == 0.5

    def test_from_dict(self):
        d = {
            "title": "Clone detected",
            "method": "clone_detection",
            "severity": "high",
            "confidence": 0.9,
            "description": "Copy-move region found",
            "item_id": "fig1.png",
            "evidence": {"inliers": 80},
        }
        f = Finding.from_dict(d)
        assert f.severity == Severity.HIGH
        assert f.method == "clone_detection"
        assert f.evidence["inliers"] == 80

    def test_from_dict_defaults(self):
        f = Finding.from_dict({})
        assert f.title == ""
        assert f.severity == Severity.LOW
        assert f.confidence == 0.0

    def test_frozen(self):
        f = Finding(title="t", method="m", severity=Severity.LOW, confidence=0.5)
        try:
            f.title = "changed"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_round_trip(self):
        original = Finding(
            title="Test",
            method="ela",
            severity=Severity.CRITICAL,
            confidence=0.95,
            description="desc",
            item_id="id1",
            evidence={"key": "val"},
        )
        restored = Finding.from_dict(original.to_dict())
        assert restored == original


class TestAggregatedResult:
    def test_to_dict(self):
        ar = AggregatedResult(
            risk_level=Severity.HIGH,
            overall_confidence=0.8,
            converging_evidence=True,
            total_findings=5,
            methods_flagged={"ela", "clone_detection"},
        )
        d = ar.to_dict()
        assert d["risk_level"] == "high"
        assert d["converging_evidence"] is True
        assert "clone_detection" in d["methods_flagged"]
        assert "ela" in d["methods_flagged"]


class TestLabeledItem:
    def test_create(self):
        item = LabeledItem(
            id="paper1.pdf",
            category="retracted",
            expected_label="findings",
            data="/path/to/paper1.pdf",
            metadata={"source": "PMC"},
        )
        assert item.category == "retracted"
        assert item.expected_label == "findings"


class TestPassFailRule:
    def test_findings_rule(self):
        rule = PassFailRule(
            expected_label="findings",
            min_risk_for_pass=Severity.MEDIUM,
        )
        assert rule.min_risk_for_pass == Severity.MEDIUM
        assert rule.max_risk_for_pass is None

    def test_clean_rule(self):
        rule = PassFailRule(
            expected_label="clean",
            max_risk_for_pass=Severity.MEDIUM,
        )
        assert rule.max_risk_for_pass == Severity.MEDIUM
        assert rule.min_risk_for_pass is None


class TestFitnessScore:
    def test_create(self):
        fs = FitnessScore(
            score=0.85,
            meets_constraints=True,
            metrics={"sensitivity": 0.9, "fpr": 0.0},
        )
        assert fs.meets_constraints is True
        assert fs.metrics["sensitivity"] == 0.9


class TestLabeledDatasetProtocol:
    """Verify LabeledDataset protocol accepts seed and sample_size params."""

    def test_protocol_accepts_seed(self):
        """A class implementing items(seed, sample_size) satisfies LabeledDataset."""

        class SeedAwareDataset:
            def items(self, seed: int | None = None, sample_size: int | None = None) -> list[LabeledItem]:
                return []

            def categories(self) -> list[str]:
                return []

        ds = SeedAwareDataset()
        assert isinstance(ds, LabeledDataset)

    def test_protocol_no_params_still_works(self):
        """A class implementing items() with no required params still satisfies protocol."""

        class SimpleDataset:
            def items(self, seed=None, sample_size=None):
                return []

            def categories(self):
                return []

        ds = SimpleDataset()
        assert isinstance(ds, LabeledDataset)


class TestTieredDatasetProtocol:
    def test_protocol_check(self):
        class MyTieredDataset:
            def items(self, seed=None, sample_size=None):
                return []
            def categories(self):
                return []
            def tiers(self) -> dict[str, list[str]]:
                return {"smoke": ["cat_a"], "full": ["cat_a", "cat_b"]}

        ds = MyTieredDataset()
        assert isinstance(ds, TieredDataset)
        assert isinstance(ds, LabeledDataset)

    def test_non_tiered_not_instance(self):
        class PlainDataset:
            def items(self, seed=None, sample_size=None):
                return []
            def categories(self):
                return []

        ds = PlainDataset()
        assert not isinstance(ds, TieredDataset)


class TestDiagnosticCluster:
    def test_to_dict(self):
        cluster = DiagnosticCluster(
            label="Undetectable retracted papers",
            description="Retracted for data fabrication, no image evidence",
            item_ids=["paper1.pdf", "paper2.pdf"],
            suggested_action="Skip these — undetectable by image forensics",
            achievable=False,
        )
        d = cluster.to_dict()
        assert d["label"] == "Undetectable retracted papers"
        assert d["achievable"] is False
        assert len(d["item_ids"]) == 2
        assert "suggested_action" in d


class TestDiagnosticsProtocol:
    def test_protocol_check(self):
        class FakeDiagnostics:
            def diagnose(self, failures):
                return [
                    DiagnosticCluster(
                        label="test", description="test desc",
                        item_ids=["a"], suggested_action="fix it",
                        achievable=True,
                    )
                ]

        diag = FakeDiagnostics()
        assert isinstance(diag, Diagnostics)

    def test_non_diagnostics_not_instance(self):
        class NotDiag:
            pass
        assert not isinstance(NotDiag(), Diagnostics)
