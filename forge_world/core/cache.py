"""Persistent disk cache for pipeline analysis results.

Stores ``pipeline.analyze()`` results as JSON files to avoid re-analyzing
unchanged items across runs.

Layout: ``{cache_dir}/{config_hash}/{item_id_hash}.json``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forge_world.core.protocols import Finding


def _hash_item_id(item_id: str) -> str:
    """SHA256[:16] hash of item_id for filesystem safety."""
    return hashlib.sha256(item_id.encode()).hexdigest()[:16]


class AnalysisCache:
    """Disk-backed cache for pipeline analysis results.

    Each cached entry is a JSON file containing the item_id (for collision
    detection) and the serialized findings list.
    """

    def __init__(self, cache_dir: str | Path = ".forge-world/cache"):
        self.cache_dir = Path(cache_dir)
        self._hits = 0
        self._misses = 0

    def get(self, config_hash: str, item_id: str) -> list[Finding] | None:
        """Retrieve cached findings, or None on miss/corruption/collision."""
        path = self._entry_path(config_hash, item_id)
        if not path.exists():
            self._misses += 1
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            # Collision guard: stored item_id must match
            if data.get("item_id") != item_id:
                self._misses += 1
                return None
            findings = [Finding.from_dict(fd) for fd in data["findings"]]
            self._hits += 1
            return findings
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._misses += 1
            return None

    def put(self, config_hash: str, item_id: str, findings: list[Finding]) -> None:
        """Store findings for an item."""
        path = self._entry_path(config_hash, item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "item_id": item_id,
            "findings": [f.to_dict() for f in findings],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def clear(self, config_hash: str | None = None) -> int:
        """Remove cached entries. Returns count of files removed.

        If *config_hash* is given, only that config's entries are removed.
        Otherwise all cached entries are removed.
        """
        if not self.cache_dir.exists():
            return 0
        count = 0
        if config_hash is not None:
            target = self.cache_dir / config_hash
            if target.exists():
                for f in target.glob("*.json"):
                    f.unlink()
                    count += 1
                # Remove the directory too if empty
                try:
                    target.rmdir()
                except OSError:
                    pass
        else:
            for f in self.cache_dir.rglob("*.json"):
                f.unlink()
                count += 1
            # Clean up empty subdirectories
            for d in sorted(self.cache_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
        self._hits = 0
        self._misses = 0
        return count

    @property
    def stats(self) -> dict[str, int]:
        """Return hit/miss counters."""
        return {"hits": self._hits, "misses": self._misses}

    def _entry_path(self, config_hash: str, item_id: str) -> Path:
        return self.cache_dir / config_hash / f"{_hash_item_id(item_id)}.json"
