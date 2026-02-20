"""Core protocol types and data classes for forge-world.

Defines the six protocol types that any domain must implement to use the
benchmark-modify-benchmark cycle:

1. Pipeline - Analyzes items and produces findings
2. Aggregator - Combines findings into an aggregated result
3. LabeledDataset - Provides labeled test items
4. PassFailRuleSet - Determines pass/fail for each item
5. FitnessFunction - Evaluates overall benchmark quality
6. Finding/AggregatedResult - Shared data types
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")
ConfigT = TypeVar("ConfigT")


class Severity(enum.Enum):
    """Severity levels for findings, ordered from lowest to highest."""

    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) < order.index(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) > order.index(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self == other or self > other


@dataclass(frozen=True)
class Finding:
    """A single finding produced by a pipeline analysis method.

    Maps to snoopy's FindingDict but as a proper dataclass with typed fields.
    """

    title: str
    method: str
    severity: Severity
    confidence: float
    description: str = ""
    item_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "method": self.method,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "item_id": self.item_id,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        sev = data.get("severity", "low")
        if isinstance(sev, Severity):
            severity = sev
        else:
            severity = Severity(sev)
        return cls(
            title=data.get("title", ""),
            method=data.get("method", ""),
            severity=severity,
            confidence=float(data.get("confidence", 0.0)),
            description=data.get("description", ""),
            item_id=data.get("item_id", ""),
            evidence=data.get("evidence", {}),
        )


@dataclass
class AggregatedResult:
    """Result of aggregating findings for a single item.

    Maps to snoopy's AggregatedEvidence but generalized.
    """

    risk_level: Severity
    overall_confidence: float
    converging_evidence: bool
    total_findings: int
    methods_flagged: set[str] = field(default_factory=set)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "overall_confidence": self.overall_confidence,
            "converging_evidence": self.converging_evidence,
            "total_findings": self.total_findings,
            "methods_flagged": sorted(self.methods_flagged),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class LabeledItem:
    """A single item in a labeled dataset with its expected outcome."""

    id: str
    category: str
    expected_label: str
    data: Any  # The actual input data (path, object, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassFailRule:
    """Rule for determining pass/fail for items with a given expected label.

    For 'findings' items: pass if risk >= min_risk_for_pass
    For 'clean' items: pass if risk <= max_risk_for_pass
    """

    expected_label: str
    min_risk_for_pass: Severity | None = None
    max_risk_for_pass: Severity | None = None


@dataclass
class FitnessScore:
    """Result of evaluating a benchmark report's fitness."""

    score: float
    meets_constraints: bool
    constraint_violations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


# --- Protocols ---


@runtime_checkable
class Pipeline(Protocol[InputT, ConfigT]):
    """Analyzes items and produces findings.

    Maps to snoopy's run_image_forensics + _analyze_pdf.
    """

    def analyze(self, item: InputT) -> list[Finding]: ...
    def get_config(self) -> ConfigT: ...
    def set_config(self, config: ConfigT) -> None: ...
    def get_config_schema(self) -> dict[str, Any]: ...


@runtime_checkable
class Aggregator(Protocol):
    """Combines findings into an aggregated result.

    Maps to snoopy's aggregate_findings.
    """

    def aggregate(self, findings: list[Finding]) -> AggregatedResult: ...


@runtime_checkable
class LabeledDataset(Protocol):
    """Provides labeled test items for benchmarking.

    When ``seed`` is None, return only fixed/deterministic items.
    When ``seed`` is an int, return fixed items plus seed-sampled items.
    ``sample_size`` controls how many seeded-random items are drawn per seed.
    """

    def items(
        self, seed: int | None = None, sample_size: int | None = None
    ) -> list[LabeledItem]: ...
    def categories(self) -> list[str]: ...


@runtime_checkable
class PassFailRuleSet(Protocol):
    """Determines pass/fail rules per expected label."""

    def get_rule(self, expected_label: str) -> PassFailRule: ...


@runtime_checkable
class FitnessFunction(Protocol):
    """Evaluates overall benchmark quality."""

    def evaluate(self, report: Any) -> FitnessScore: ...


@runtime_checkable
class TieredDataset(Protocol):
    """Optional protocol for datasets that support tiered evaluation.

    Returns a mapping of tier name to list of category names.
    """

    def tiers(self) -> dict[str, list[str]]: ...


@dataclass
class DiagnosticCluster:
    """A domain-specific group of failures with suggested action."""

    label: str
    description: str
    item_ids: list[str]
    suggested_action: str
    achievable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "item_ids": self.item_ids,
            "suggested_action": self.suggested_action,
            "achievable": self.achievable,
        }


@runtime_checkable
class Diagnostics(Protocol):
    """Optional protocol for domain-specific failure diagnosis."""

    def diagnose(self, failures: list[Any]) -> list[DiagnosticCluster]: ...
