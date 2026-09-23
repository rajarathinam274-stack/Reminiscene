"""Unified Memory Event representation.

Every modality (PDF, DOCX, PPTX, TXT/MD, audio, video, image/screenshot)
maps into a single ``MemoryEvent`` structure so that retrieval, storage and
evidence handling can be modality-agnostic.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Modality(str, Enum):
    PDF = "pdf"
    DOCUMENT = "document"   # docx / pptx / txt / markdown
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"         # photos, screenshots
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
    def from_dict(d: dict) -> "Region":
        b = d["bbox"]
        return Region(b[0], b[1], b[2], b[3])

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection_over_union(self, other: "Region") -> float:
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
    embedding: Optional[list[float]] = None

    # Temporal evidence (audio/video)
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None

    # Structural evidence (documents)
    page: Optional[int] = None
    section: Optional[str] = None

    # Spatial evidence (images/screenshots/PDF regions)
    location: Optional[Region] = None

    concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_event: Optional[str] = None

    created_at: Optional[str] = None  # ISO string; set by persistence layer

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = Modality(self.modality).value
        if self.location is not None:
            d["location"] = self.location.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "MemoryEvent":
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
    def deserialize(s: str) -> "MemoryEvent":
        return MemoryEvent.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    def time_range(self) -> Optional[tuple[float, float]]:
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
