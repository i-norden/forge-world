"""Tests for persistent disk cache."""

from __future__ import annotations

import json

import pytest

from forge_world.core.cache import AnalysisCache
from forge_world.core.protocols import Finding, Severity


def _make_findings() -> list[Finding]:
    return [
        Finding(
            title="ELA anomaly",
            method="ela",
            severity=Severity.HIGH,
            confidence=0.85,
            description="High ELA response",
            item_id="img1.png",
            evidence={"max_diff": 65.0},
        ),
        Finding(
            title="Clone detected",
            method="clone_detection",
            severity=Severity.MEDIUM,
            confidence=0.6,
        ),
    ]


class TestAnalysisCache:
    def test_put_and_get(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg123", "item1", findings)

        result = cache.get("cfg123", "item1")
        assert result is not None
        assert len(result) == 2
        assert result[0].title == "ELA anomaly"
        assert result[0].severity == Severity.HIGH
        assert result[1].method == "clone_detection"

    def test_miss_returns_none(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        assert cache.get("cfg123", "nonexistent") is None

    def test_different_config_hash_separate(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg_a", "item1", findings)

        # Same item, different config = miss
        assert cache.get("cfg_b", "item1") is None

        # Original config still works
        assert cache.get("cfg_a", "item1") is not None

    def test_clear_all(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg_a", "item1", findings)
        cache.put("cfg_b", "item2", findings)

        count = cache.clear()
        assert count == 2
        assert cache.get("cfg_a", "item1") is None
        assert cache.get("cfg_b", "item2") is None

    def test_clear_specific_config(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg_a", "item1", findings)
        cache.put("cfg_b", "item2", findings)

        count = cache.clear("cfg_a")
        assert count == 1
        assert cache.get("cfg_a", "item1") is None
        # cfg_b untouched
        assert cache.get("cfg_b", "item2") is not None

    def test_corrupt_json_returns_none(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg123", "item1", findings)

        # Corrupt the file
        path = cache._entry_path("cfg123", "item1")
        path.write_text("not valid json{{{")

        assert cache.get("cfg123", "item1") is None

    def test_finding_round_trip(self, tmp_path):
        """Findings survive JSON serialization through the cache."""
        cache = AnalysisCache(tmp_path / "cache")
        original = _make_findings()
        cache.put("cfg123", "item1", original)

        restored = cache.get("cfg123", "item1")
        assert restored is not None
        for orig, rest in zip(original, restored):
            assert orig.title == rest.title
            assert orig.method == rest.method
            assert orig.severity == rest.severity
            assert orig.confidence == rest.confidence
            assert orig.description == rest.description

    def test_stats_tracking(self, tmp_path):
        cache = AnalysisCache(tmp_path / "cache")
        findings = _make_findings()
        cache.put("cfg123", "item1", findings)

        cache.get("cfg123", "item1")  # hit
        cache.get("cfg123", "item1")  # hit
        cache.get("cfg123", "missing")  # miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_empty_findings(self, tmp_path):
        """Empty findings list round-trips correctly."""
        cache = AnalysisCache(tmp_path / "cache")
        cache.put("cfg123", "clean_item", [])

        result = cache.get("cfg123", "clean_item")
        assert result is not None
        assert result == []

    def test_clear_empty_cache(self, tmp_path):
        """Clearing a nonexistent cache returns 0."""
        cache = AnalysisCache(tmp_path / "cache")
        assert cache.clear() == 0
