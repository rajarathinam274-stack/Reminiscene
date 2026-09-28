"""Deterministic evidence pack for LLM-free memory answering.

The evidence pack is the canonical hand-off between retrieval and answer
generation. It contains only source-anchored evidence, deterministic ranking
metadata, temporal context, and explicit conflicts. No generative model is
required to construct or consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .resolver import Evidence


@dataclass(frozen=True)
class EvidenceItem:
    """A source-anchored item safe for deterministic answer generation."""

    evidence: Evidence
    relevance: float = 0.0
    confidence: float = 0.0

    @property
    def memory_id(self) -> str:
        return self.evidence.memory_id

    def to_dict(self) -> dict:
        data = self.evidence.to_dict()
        data.update(
            {
                "relevance": round(max(0.0, min(1.0, self.relevance)), 4),
                "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
            }
        )
        return data


@dataclass(frozen=True)
class EvidenceConflict:
    """Explicit contradiction between source-anchored evidence items."""

    memory_ids: tuple[str, ...]
    kind: str
    description: str

    def to_dict(self) -> dict:
        return {
            "memory_ids": list(self.memory_ids),
            "kind": self.kind,
            "description": self.description,
        }


@dataclass
class EvidencePack:
    """Validated, deduplicated evidence for a single user query.

    The pack intentionally contains no generated claims. Consumers can use it
    for deterministic facts, timelines, source lookup, and extractive
    summaries, while an optional LLM can consume the same contract later.
    """

    query: str
    items: list[EvidenceItem] = field(default_factory=list)
    temporal_context: Optional[dict] = None
    conflicts: list[EvidenceConflict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.items = self._deduplicate(self.items)
        self.items.sort(
            key=lambda item: (item.relevance, item.confidence), reverse=True
        )

    @staticmethod
    def _deduplicate(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
        seen: set[str] = set()
        result: list[EvidenceItem] = []
        for item in items:
            if not item.memory_id or item.memory_id in seen:
                continue
            if not item.evidence.source or not item.evidence.source_path:
                continue
            seen.add(item.memory_id)
            result.append(item)
        return result

    @property
    def grounded(self) -> bool:
        return bool(self.items)

    @property
    def confidence(self) -> float:
        if not self.items:
            return 0.0
        return max(
            0.0,
            min(1.0, sum(i.confidence for i in self.items) / len(self.items)),
        )

    def top(self, limit: int = 5) -> list[EvidenceItem]:
        if limit < 1:
            return []
        return self.items[:limit]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "grounded": self.grounded,
            "confidence": round(self.confidence, 4),
            "items": [item.to_dict() for item in self.items],
            "temporal_context": self.temporal_context,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    @classmethod
    def from_evidence(
        cls,
        query: str,
        evidences: Iterable[Evidence],
        *,
        relevance: Optional[dict[str, float]] = None,
        confidence: Optional[dict[str, float]] = None,
        temporal_context: Optional[dict] = None,
        conflicts: Optional[Iterable[EvidenceConflict]] = None,
    ) -> "EvidencePack":
        relevance = relevance or {}
        confidence = confidence or {}
        items = [
            EvidenceItem(
                evidence=evidence,
                relevance=relevance.get(evidence.memory_id, 1.0),
                confidence=confidence.get(evidence.memory_id, 1.0),
            )
            for evidence in evidences
        ]
        return cls(
            query=query,
            items=items,
            temporal_context=temporal_context,
            conflicts=list(conflicts or []),
        )
