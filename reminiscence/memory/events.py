"""Unified Memory Event representation.

Every modality (PDF, DOCX, PPTX, TXT/MD, audio, video, image/screenshot)
maps into a single ``MemoryEvent`` structure so that retrieval, storage and
evidence handling can be modality-agnostic.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    PDF = "pdf"
    DOCUMENT = "document"  # docx / pptx / txt / markdown
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"  # photos, screenshots
    NOTE = "note"


@dataclass
class Region:
    """A spatial region inside an image/page (bounding box)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict:
        return {"bbox": [self.x0, self.y0, self.x1, self.y1]}

    @staticmethod
    def from_dict(d: dict) -> Region:
        b = d["bbox"]
        return Region(b[0], b[1], b[2], b[3])

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection_over_union(self, other: Region) -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass
class MemoryEvent:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str = ""
    modality: Modality = Modality.DOCUMENT
    content: str = ""
    embedding: list[float] | None = None

    # Temporal evidence (audio/video)
    timestamp_start: float | None = None
    timestamp_end: float | None = None

    # Structural evidence (documents)
    page: int | None = None
    section: str | None = None

    # Spatial evidence (images/screenshots/PDF regions)
    location: Region | None = None

    concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_event: str | None = None

    created_at: str | None = None  # ISO string; set by persistence layer

    # -- Phase 14 (P2): explicit temporal semantics -----------------------
    # created_at is the INDEXING time. It must never be used as a stand-in
    # for when the memory actually happened or was captured:
    #   photo taken 2022-12-14, imported 2026-09-24  ->  event/capture time
    #   is 2022-12-14; "memories from December 2022" queries must use it.
    event_time_start: str | None = None  # ISO; when the remembered thing occurred
    event_time_end: str | None = None  # ISO; end of the occurrence if bounded
    captured_at: str | None = None  # ISO; when media/file was captured (EXIF/mtime)
    modified_at: str | None = None  # ISO; last modification of the underlying content
    time_source: str | None = None  # exif | filename | transcript | import | manual | inferred
    time_confidence: float = 1.0  # 0..1 — how sure we are about event/capture time

    # -- Phase 14 (P2): reproducibility / incremental re-indexing ----------
    # Every processed memory must be traceable to the exact pipeline and
    # model versions that produced it (debugging + benchmark reproducibility).
    extraction_version: str | None = None
    embedding_version: str | None = None
    pipeline_version: str | None = None

    # ------------------------------------------------------------------
    def effective_event_time(self) -> str | None:
        """Best available *memory* timestamp (never silently indexing time)."""
        return self.event_time_start or self.captured_at or self.created_at

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = Modality(self.modality).value
        if self.location is not None:
            d["location"] = self.location.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> MemoryEvent:
        d = dict(d)
        if d.get("location"):
            d["location"] = Region.from_dict(d["location"])
        if d.get("modality"):
            d["modality"] = Modality(d["modality"])
        known = set(MemoryEvent.__dataclass_fields__)
        return MemoryEvent(**{k: v for k, v in d.items() if k in known})

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def deserialize(s: str) -> MemoryEvent:
        return MemoryEvent.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    def time_range(self) -> tuple[float, float] | None:
        if self.timestamp_start is None:
            return None
        end = self.timestamp_end if self.timestamp_end is not None else self.timestamp_start
        return (float(self.timestamp_start), float(end))

    def has_evidence_anchor(self) -> bool:
        """True when this event can point back to an exact source location."""
        return (
            self.page is not None
            or self.section is not None
            or self.timestamp_start is not None
            or self.location is not None
        )


def format_timestamp(seconds: float) -> str:
    """Human-friendly HH:MM:SS (or MM:SS.ss) timestamp formatting."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{int(s):02d}"
    return f"{m:02d}:{s:05.2f}"
