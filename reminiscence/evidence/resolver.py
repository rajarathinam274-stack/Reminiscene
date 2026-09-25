"""Evidence resolution: map Memory Events to clickable source references.

Answer -> Evidence -> Original source (page / timestamp / region).
Never fabricate citations: an event without a resolvable anchor yields no
evidence entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..memory.events import MemoryEvent, format_timestamp
from ..storage.database import Database


@dataclass
class Evidence:
    memory_id: str
    source: str  # display name (file name)
    source_path: str  # local filesystem path for "open" action
    modality: str
    page: int | None = None
    section: str | None = None
    timestamp: str | None = None  # formatted HH:MM:SS
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    bbox: list[float] | None = None  # image/screenshot region
    snippet: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, "", [])}
        return d

    def locator(self) -> str:
        """Human-readable pointer used by the UI."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"§ {self.section}")
        if self.timestamp:
            parts.append(self.timestamp)
        if self.bbox:
            parts.append(
                f"[{int(self.bbox[0])},{int(self.bbox[1])}]-[{int(self.bbox[2])},{int(self.bbox[3])}]"
            )
        return " · ".join(parts)


class EvidenceResolver:
    def __init__(self, db: Database):
        self.db = db

    def resolve(self, ev: MemoryEvent, snippet_chars: int = 280) -> Evidence | None:
        src = self.db.get_source(ev.source_id)
        if src is None:
            return None  # orphaned event -> no fabricated evidence
        ts = None
        if ev.timestamp_start is not None:
            ts = format_timestamp(ev.timestamp_start)
            if ev.timestamp_end is not None and ev.timestamp_end != ev.timestamp_start:
                ts = f"{ts} → {format_timestamp(ev.timestamp_end)}"
        snippet = ev.content.strip().replace("\n", " ")
        if len(snippet) > snippet_chars:
            snippet = snippet[: snippet_chars - 1] + "…"
        return Evidence(
            memory_id=ev.id,
            source=src["name"],
            source_path=src["path"],
            modality=ev.modality.value,
            page=ev.page,
            section=ev.section,
            timestamp=ts,
            timestamp_start=ev.timestamp_start,
            timestamp_end=ev.timestamp_end,
            bbox=(ev.location.to_dict()["bbox"] if ev.location else None),
            snippet=snippet,
        )

    def resolve_many(self, events: list[MemoryEvent]) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[str] = set()
        for ev in events:
            e = self.resolve(ev)
            if e and e.memory_id not in seen:
                seen.add(e.memory_id)
                out.append(e)
        return out
