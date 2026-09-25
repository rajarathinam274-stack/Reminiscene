"""Query classification before expensive retrieval.

Categories (composable): FACT, SOURCE_LOOKUP, TEMPORAL_LOOKUP, CROSS_SOURCE,
VISUAL, EXPLANATION.  A tiny rule-based classifier ships by default; an ONNX
TinyBERT head can replace it via the same ``QueryClassifier`` interface once
benchmarked on-target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum


class QueryCategory(str, Enum):
    FACT = "FACT"
    SOURCE_LOOKUP = "SOURCE_LOOKUP"
    TEMPORAL_LOOKUP = "TEMPORAL_LOOKUP"
    CROSS_SOURCE = "CROSS_SOURCE"
    VISUAL = "VISUAL"
    EXPLANATION = "EXPLANATION"


@dataclass
class ClassifiedQuery:
    text: str
    categories: list[QueryCategory] = field(default_factory=list)
    time_range: tuple[datetime, datetime] | None = None
    source_hint: str | None = None
    modality_hint: str | None = None  # pdf | audio | video | image
    terms: list[str] = field(default_factory=list)
    # Phase 14: resolved graph entities {entity_type: [entity_id, ...]}
    entities: dict[str, list[str]] = field(default_factory=dict)
    temporal_field: str | None = None  # event_time | captured_at | any
    intent_confidence: float = 0.5

    @property
    def primary(self) -> QueryCategory:
        return self.categories[0] if self.categories else QueryCategory.FACT


_VISUAL_WORDS = {
    "diagram",
    "chart",
    "graph",
    "image",
    "picture",
    "photo",
    "screenshot",
    "slide",
    "figure",
    "visual",
    "drawing",
    "map",
    "table",
    "plot",
    "logo",
}
_VISUAL_PATTERNS = [
    r"\bwhat (does|did) .* look like",
    r"\bsaw\b.*\b(diagram|chart|image|slide|picture)\b",
]
_EXPLAIN_WORDS = {
    "explain",
    "difference",
    "compare",
    "why",
    "how does",
    "meaning",
    "summarize",
    "summary",
    "versus",
    "vs",
    "interpret",
    "understand",
}
_SOURCE_WORDS = {
    "where",
    "which file",
    "which document",
    "source",
    "came from",
    "located",
    "found it",
    "what pdf",
    "what doc",
    "in which",
}
_CROSS_WORDS = {
    "across",
    "all my",
    "every",
    "both",
    "between",
    "notes and",
    "connect",
    "related",
    "multiple",
    "corroborat",
}

_RELATIVE_TIME = {
    "yesterday": timedelta(days=1),
    "today": timedelta(days=0),
    "last week": timedelta(days=7),
    "this week": timedelta(days=7),
    "last month": timedelta(days=30),
    "this month": timedelta(days=30),
    "past month": timedelta(days=30),
    "recent": timedelta(days=7),
    "last year": timedelta(days=365),
}

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}


def _month_bounds(y: int, mo: int) -> tuple[datetime, datetime]:
    start = datetime(y, mo, 1, tzinfo=UTC)
    end = datetime(y + (mo == 12), (mo % 12) + 1, 1, tzinfo=UTC)
    return start, end - timedelta(microseconds=1)


class QueryClassifier:
    """Rule-based baseline implementing the required category set.

    Deterministic and dependency-free; a compact ONNX head can replace it
    behind the same interface once benchmarked on-target.
    """

    def __init__(self, entity_resolver=None):
        """entity_resolver: optional callable(text_lower) ->
        dict[str, list[str]] mapping entity_type -> entity_ids, used to hook
        the classifier into the memory graph without coupling to storage."""
        self.entity_resolver = entity_resolver

    def classify(self, text: str, now: datetime | None = None) -> ClassifiedQuery:
        now = now or datetime.now(UTC)
        low = text.lower()
        cats: list[QueryCategory] = []

        # temporal
        tr = self._time_range(low, now)
        if tr:
            cats.append(QueryCategory.TEMPORAL_LOOKUP)

        # visual
        if any(w in low for w in _VISUAL_WORDS) or any(re.search(p, low) for p in _VISUAL_PATTERNS):
            cats.append(QueryCategory.VISUAL)

        # source lookup ("where did I see...", quoted filenames)
        if any(w in low for w in _SOURCE_WORDS) or re.search(
            r"\.(pdf|mp4|mp3|docx|pptx|png|jpe?g)\b", low
        ):
            if QueryCategory.SOURCE_LOOKUP not in cats:
                cats.append(QueryCategory.SOURCE_LOOKUP)

        # cross-source
        if any(w in low for w in _CROSS_WORDS) or re.search(
            r"\b(my notes?|my docs?|my files?)\b", low
        ):
            cats.append(QueryCategory.CROSS_SOURCE)

        # explanation
        if any(w in low for w in _EXPLAIN_WORDS):
            cats.append(QueryCategory.EXPLANATION)

        if not cats:
            cats.append(QueryCategory.FACT)

        modality = None
        if ".pdf" in low:
            modality = "pdf"
        elif ".mp4" in low or "video" in low or "lecture" in low:
            modality = "video"
        elif ".mp3" in low or "recording" in low or "audio" in low:
            modality = "audio"
        elif any(k in low for k in ("screenshot", ".png", ".jpg", "image")):
            modality = "image"

        m = re.search(r"([A-Za-z0-9_\-\. ]+\.(?:pdf|mp4|mp3|docx|pptx|png|jpe?g))", low)
        source_hint = m.group(1).strip() if m else None

        terms = re.findall(r"[a-z0-9]{3,}", low)
        stop = {
            "what",
            "where",
            "when",
            "which",
            "about",
            "that",
            "this",
            "with",
            "from",
            "did",
            "does",
            "the",
            "and",
            "for",
            "was",
            "were",
            "how",
            "why",
        }
        terms = [t for t in terms if t not in stop]

        # entity resolution against the memory graph (optional hook)
        entities: dict[str, list[str]] = {}
        if self.entity_resolver is not None:
            try:
                entities = self.entity_resolver(low) or {}
            except Exception:
                entities = {}
        if entities and QueryCategory.CROSS_SOURCE not in cats:
            cats.append(QueryCategory.CROSS_SOURCE)

        intent_conf = min(
            1.0, 0.5 + 0.1 * len(cats) + (0.2 if tr else 0.0) + (0.1 if entities else 0.0)
        )

        return ClassifiedQuery(
            text=text,
            categories=cats,
            time_range=tr,
            source_hint=source_hint,
            modality_hint=modality,
            terms=terms,
            entities=entities,
            temporal_field="event_time" if tr else None,
            intent_confidence=intent_conf,
        )

    @staticmethod
    def _time_range(low: str, now: datetime) -> tuple[datetime, datetime] | None:
        # explicit month+year ("December 2022", "in July 2024") first —
        # calendar windows beat relative approximations.
        m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", low)
        if m:
            return _month_bounds(int(m.group(2)), _MONTHS[m.group(1)])
        m = re.search(r"\b(?:in|during)\s+(" + "|".join(_MONTHS) + r")\b", low)
        if m:
            y = now.year
            start, end = _month_bounds(y, _MONTHS[m.group(1)])
            if start > now:  # "in December" said in September means last December
                start, end = _month_bounds(y - 1, _MONTHS[m.group(1)])
            return start, end
        for phrase, delta in _RELATIVE_TIME.items():
            if phrase in low:
                start = now - delta
                # normalize 'last month' to calendar-ish window
                return (start.replace(hour=0, minute=0, second=0, microsecond=0), now)
        m = re.search(r"\bin (\d{4})\b", low)
        if m:
            y = int(m.group(1))
            return (datetime(y, 1, 1, tzinfo=UTC), datetime(y, 12, 31, 23, 59, tzinfo=UTC))
        m = re.search(r"\b(\w+) (\d{1,2}),? (\d{4})\b", low)
        if m:
            try:
                d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y")
                d = d.replace(tzinfo=now.tzinfo or UTC)
                return (d, d + timedelta(days=1))
            except ValueError:
                return None
        return None
