"""Semantic and structural chunking.

Never blindly split inputs into arbitrary fixed-size chunks:
- PDFs/documents: split by headings/paragraphs/sections, then merge small
  neighbors up to a token budget; oversized blocks are split on sentence
  boundaries with heading context preserved.
- Transcripts: split on topic/pause/speaker boundaries with a duration cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z\"'(0-9\u4e00-\u9fff])|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


@dataclass
class Chunk:
    text: str
    section: str | None = None
    page: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    concepts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _merge_blocks(
    blocks: list[tuple[str, int | None]],
    min_words: int,
    max_words: int,
) -> list[tuple[str, int | None]]:
    """Merge adjacent (text, page) blocks so chunks land in [min,max] words."""
    merged: list[tuple[str, int | None]] = []
    buf_text = ""
    buf_pages: list[int | None] = []

    def flush():
        nonlocal buf_text, buf_pages
        if buf_text.strip():
            page = next((p for p in buf_pages if p is not None), None)
            merged.append((buf_text.strip(), page))
        buf_text = ""
        buf_pages = []

    for text, page in blocks:
        if not text.strip():
            continue
        if buf_text and word_count(buf_text) + word_count(text) > max_words:
            flush()
        buf_text = (buf_text + "\n\n" + text).strip() if buf_text else text
        buf_pages.append(page)
        if word_count(buf_text) >= min_words:
            flush()
    flush()
    return merged


def _split_oversized(text: str, max_words: int, heading: str | None) -> list[str]:
    """Split an oversized block on sentence boundaries keeping the heading."""
    sentences = split_sentences(text)
    pieces: list[str] = []
    cur: list[str] = []
    cur_words = 0
    prefix_words = word_count(heading) if heading else 0
    for s in sentences:
        w = word_count(s)
        if cur and cur_words + w + prefix_words > max_words:
            pieces.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += w
    if cur:
        pieces.append(" ".join(cur))
    if heading:
        pieces = [f"{heading}\n{p}" for p in pieces]
    return pieces


# ---------------------------------------------------------------------------
# Document / PDF structure chunking
# ---------------------------------------------------------------------------

_HEADING_MAX_WORDS = 12
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 160:
        return False
    if re.match(r"^#{1,6}\s+", line):  # markdown
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S", line) and word_count(line) <= _HEADING_MAX_WORDS:
        return True
    words = _WORD_RE.findall(line)
    if len(words) == 0 or len(words) > _HEADING_MAX_WORDS:
        return False
    letters = [c for c in line if c.isalpha()]
    if line.endswith(":"):
        return True
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return True
    # Title case short line
    titled = sum(1 for w in words if w[:1].isupper()) / len(words)
    return titled >= 0.8 and len(words) <= 8


def structure_text(text: str) -> list[tuple[str | None, str]]:
    """Return (heading, paragraph) pairs from raw extracted text."""
    out: list[tuple[str | None, str]] = []
    heading: str | None = None
    for block in _PARAGRAPH_BREAK_RE.split(text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if looks_like_heading(lines[0]):
            heading = lines[0].lstrip("#").strip()
            rest = "\n".join(lines[1:]).strip()
            if rest:
                out.append((heading, rest))
        else:
            out.append((heading, block))
    return out


def chunk_document_text(
    text: str,
    min_words: int = 40,
    max_words: int = 220,
) -> list[Chunk]:
    """Structural chunking for plain document text (PDF pages joined, txt/md)."""
    pairs = structure_text(text)
    blocks = [(body, None) for _, body in pairs]
    # Pre-split oversized blocks (keep heading prefix for context)
    prepared: list[tuple[str, int | None]] = []
    for (head, body), (_, page) in zip(pairs, blocks):
        if word_count(body) > max_words:
            prepared.extend((p, page) for p in _split_oversized(body, max_words, head))
        else:
            prepared.append((body, page))
    chunks: list[Chunk] = []
    for text_, page in _merge_blocks(prepared, min_words, max_words):
        head = None
        first_line = text_.splitlines()[0].strip() if text_.splitlines() else ""
        if looks_like_heading(first_line) and "\n" in text_:
            head = first_line.lstrip("#").strip()
        chunks.append(Chunk(text=text_, page=page, section=head))
    return chunks


def chunk_pdf_pages(pages: list[str], min_words: int = 40, max_words: int = 220) -> list[Chunk]:
    """Chunk per-page extracted PDF text while preserving page numbers."""
    all_chunks: list[Chunk] = []
    current_heading: str | None = None
    for page_no, page_text in enumerate(pages, start=1):
        pairs = structure_text(page_text)
        prepared: list[tuple[str, int | None]] = []
        for head, body in pairs:
            if head:
                current_heading = head
            if word_count(body) > max_words:
                prepared.extend(
                    (p, page_no) for p in _split_oversized(body, max_words, head or current_heading)
                )
            else:
                prepared.append((body, page_no))
        for text_, page in _merge_blocks(prepared, min_words, max_words):
            first_line = text_.splitlines()[0].strip()
            section = None
            if current_heading and current_heading in text_:
                section = current_heading
            elif looks_like_heading(first_line) and "\n" in text_:
                section = first_line.lstrip("#").strip()
            all_chunks.append(Chunk(text=text_, page=page, section=section))
    return all_chunks


# ---------------------------------------------------------------------------
# Transcript chunking (audio / video)
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def chunk_transcript(
    segments: list[Segment],
    max_duration: float = 60.0,
    max_words: int = 180,
    pause_threshold: float = 2.5,
    min_words: int = 12,
) -> list[Chunk]:
    """Group ASR segments into semantically coherent chunks.

    Boundaries: long pauses, speaker changes, duration/word caps.
    Timestamps of each chunk span its member segments.
    """
    chunks: list[Chunk] = []
    cur_segs: list[Segment] = []

    def flush():
        if not cur_segs:
            return
        text = " ".join(s.text.strip() for s in cur_segs).strip()
        if not text:
            return
        speakers = {s.speaker for s in cur_segs if s.speaker}
        chunks.append(
            Chunk(
                text=text,
                timestamp_start=round(cur_segs[0].start, 3),
                timestamp_end=round(cur_segs[-1].end, 3),
                concepts=sorted(f"speaker:{sp}" for sp in speakers),
                metadata={"segment_count": len(cur_segs)},
            )
        )
        cur_segs.clear()

    for seg in segments:
        if not seg.text.strip():
            continue
        if cur_segs:
            prev = cur_segs[-1]
            boundary = (
                (seg.start - prev.end) >= pause_threshold
                or (seg.speaker and prev.speaker and seg.speaker != prev.speaker)
                or (seg.end - cur_segs[0].start) > max_duration
                or (sum(word_count(s.text) for s in cur_segs) + word_count(seg.text)) > max_words
            )
            if boundary:
                if word_count(" ".join(s.text for s in cur_segs)) < min_words:
                    # too small to stand alone: extend past the boundary
                    pass
                else:
                    flush()
        cur_segs.append(seg)
    flush()
    return chunks
