"""Ingestion pipeline: file -> metadata -> modality processing -> chunks ->
MemoryEvents -> embeddings -> SQLite persistence -> vector index.

Modality-specific extractors are pluggable.  When heavy native dependencies
(PyMuPDF, python-docx/pptx, ffmpeg, OCR models) are absent, the pipeline
degrades gracefully with clear errors instead of crashing — and text-based
formats (txt/md) always work.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..ai.embeddings.embedder import Embedder
from ..ai.scheduler import AIWorkloadScheduler
from ..memory.chunking import Chunk, Segment, chunk_document_text, chunk_pdf_pages, chunk_transcript
from ..memory.events import MemoryEvent, Modality, Region
from ..storage.database import Database
from ..storage.media_store import MediaObject, MediaStore
from ..storage.vector_index import VectorIndex
from ..workers.queue import Job, JobContext

# Reproducibility (Phase 14): every processed memory is traceable to the
# pipeline/extraction versions that produced it. Bump on behavior changes.
PIPELINE_VERSION = "1.0"
EXTRACTION_VERSION = "1.0"

log = logging.getLogger("reminiscence.ingestion")

SUPPORTED_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".markdown",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
}

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class UnsupportedFormat(Exception):
    pass


class MissingDependency(Exception):
    pass


def detect_modality(path: str | Path) -> Modality:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return Modality.PDF
    if ext in {".docx", ".pptx", ".txt", ".md", ".markdown"}:
        return Modality.DOCUMENT
    if ext in AUDIO_EXTS:
        return Modality.AUDIO
    if ext in VIDEO_EXTS:
        return Modality.VIDEO
    if ext in IMAGE_EXTS:
        return Modality.IMAGE
    raise UnsupportedFormat(f"Unsupported file format '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")


def content_hash(path: str | Path) -> str:
    """SHA-256 of file bytes — the *content identity* (Phase 14).

    Path-based ids break on move/copy; content hashes enable duplicate
    detection, moved-file recognition and reliable re-indexing.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_id_for(path: str | Path) -> str:
    p = Path(path).resolve()
    return hashlib.sha1(str(p).encode()).hexdigest()[:16]


def capture_time_for(path: str | Path) -> tuple[str | None, str]:
    """Best-effort *capture* time of a file (never import time).

    Priority: EXIF DateTimeOriginal (images) > filesystem mtime.
    Returns (iso_utc_or_None, time_source).
    """
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXTS:
        try:  # Pillow is optional
            from PIL import Image  # type: ignore

            with Image.open(p) as im:
                exif = im.getexif()
                for tag in (36867, 306):  # DateTimeOriginal, DateTime
                    val = exif.get(tag)
                    if val:
                        d = datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
                        return d.isoformat(), "exif"
        except Exception:
            pass
    return _mtime_iso(p), "filesystem_mtime"


def _mtime_iso(path: str | Path) -> str | None:
    try:
        m = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)
        return m.isoformat()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Extractors (per modality)
# ---------------------------------------------------------------------------


def extract_pdf_pages(path: str | Path) -> list[str]:
    """PyMuPDF page texts; falls back to a clear error when unavailable."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise MissingDependency(
            "PyMuPDF is required for PDF ingestion (pip install pymupdf)"
        ) from e
    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def extract_docx(path: str | Path) -> str:
    try:
        import docx
    except ImportError as e:
        raise MissingDependency("python-docx is required for DOCX ingestion") from e
    d = docx.Document(str(path))
    parts = []
    for p in d.paragraphs:
        style = (p.style.name or "").lower()
        text = p.text.strip()
        if not text:
            continue
        parts.append(f"# {text}" if "heading" in style else text)
    return "\n\n".join(parts)


def extract_pptx(path: str | Path) -> list[tuple[int, str]]:
    """Return per-slide text (slide number acts as 'page')."""
    try:
        from pptx import Presentation
    except ImportError as e:
        raise MissingDependency("python-pptx is required for PPTX ingestion") from e
    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(r.text for r in para.runs).strip()
                    if t:
                        lines.append(t)
        if lines:
            out.append((i, "\n".join(lines)))
    return out


def extract_plain_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def ffprobe_duration(path: str | Path) -> float | None:
    if shutil.which("ffmpeg") is None and shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def extract_audio_to_wav(path: str | Path, dest_dir: Path) -> Path:
    """FFmpeg audio extraction for video files / non-wav audio."""
    if shutil.which("ffmpeg") is None:
        raise MissingDependency("FFmpeg is required for audio/video ingestion")
    dest = dest_dir / (Path(path).stem + "_audio.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ],
        check=True,
        timeout=1800,
    )
    return dest


# ---------------------------------------------------------------------------
# ASR/OCR interfaces (pluggable; honest stubs when models missing)
# ---------------------------------------------------------------------------


class Transcriber:
    """Timestamped local transcription interface (Whisper-Base candidate)."""

    def transcribe(self, wav_path: str | Path) -> list[Segment]:
        raise NotImplementedError


class OCRBackend:
    """OCR interface preserving text/page/bbox/confidence."""

    def recognize(self, image_path: str | Path) -> list[dict]:
        """Return [{"text":..., "page":..., "bbox":[x0,y0,x1,y1], "confidence":...}]"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Adapters bridging ai.asr / ai.vision.ocr backends into the pipeline
# interfaces above (Sprint 4 wiring — keeps ingestion decoupled from engines)
# ---------------------------------------------------------------------------


class AsrAdapter(Transcriber):
    """Adapts a ``reminiscence.ai.asr`` transcriber to the pipeline interface.

    The rich backend returns a TranscriptResult carrying language and the
    *actual* execution provider; we surface plain Segments here and stash the
    metadata on ``last_result`` so callers (jobs, benchmarks) can record it.
    """

    def __init__(self, backend):
        if not hasattr(backend, "transcribe_segments"):
            raise TypeError(
                "AsrAdapter expects an ai.asr backend exposing transcribe_segments(); "
                "pass the pipeline-native Transcriber implementation directly instead."
            )
        self.backend = backend
        self.last_result = None

    def transcribe(self, wav_path: str | Path) -> list[Segment]:
        result = self.backend.transcribe_segments(wav_path)
        self.last_result = result
        return list(result.segments)


class OcrAdapter(OCRBackend):
    """Adapts a ``reminiscence.ai.vision.ocr`` backend to the pipeline dict shape."""

    def __init__(self, backend):
        self.backend = backend

    def recognize(self, image_path: str | Path) -> list[dict]:
        result = self.backend.recognize(image_path)
        return [r.to_dict() for r in result.regions] or [
            {"text": result.text, "page": result.page, "bbox": None, "confidence": None}
        ]


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    source_id: str
    events_created: int
    modality: Modality
    warnings: list[str]


class IngestionPipeline:
    def __init__(
        self,
        db: Database,
        index: VectorIndex,
        embedder: Embedder,
        scheduler: AIWorkloadScheduler,
        data_dir: str | Path,
        transcriber: Transcriber | None = None,
        ocr: OCRBackend | None = None,
        media_store: MediaStore | None = None,
    ):
        self.db = db
        self.index = index
        self.embedder = embedder
        self.scheduler = scheduler
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.transcriber = transcriber
        self.ocr = ocr
        # Sprint 2: authoritative content-addressed media storage. When a
        # MediaStore is provided, every imported file becomes a canonical
        # object (objects/<hash-prefix>/<sha256>) and SQLite keeps only the
        # reference key. Without one, ingestion still works (dev/tests).
        if media_store is None:
            media_store = MediaStore(self.data_dir / "media")
        self.media_store = media_store

    # ------------------------------------------------------------------
    def ingest(
        self, path: str | Path, job: Job | None = None, ctx: JobContext | None = None
    ) -> IngestionResult:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        warnings: list[str] = []

        def stage(name: str):
            if ctx:
                ctx.stage(name)
                ctx.check_cancelled()

        def done(name: str):
            if ctx:
                ctx.complete_stage(name)

        # 1. File identification + content identity (SHA-256, not path)
        stage("identify")
        modality = detect_modality(path)
        chash = content_hash(path)
        sid = source_id_for(path)
        done("identify")

        # 1a. Canonicalize into the MediaStore: identical bytes from different
        # paths collapse onto one stored object (dedup + moved-file support).
        media_obj = self.media_store.put(path)

        # 1b. Duplicate detection: identical bytes already ingested?
        existing_src = self.db.get_source_by_hash(chash)
        if existing_src and existing_src["id"] != sid:
            # Same content known under another path (moved or copied file):
            # reuse the existing source id, refresh its location, and do NOT
            # create a second set of memories.
            mime0 = mimetypes.guess_type(str(path))[0]
            sid = self.db.upsert_source(
                existing_src["id"],
                str(path),
                path.name,
                modality,
                mime_type=mime0,
                size_bytes=path.stat().st_size,
                content_hash=chash,
                file_modified_at=_mtime_iso(path),
            )
            self.db.add_media_asset(
                f"asset-{sid}",
                media_obj.key,
                media_obj.sha256,
                "original",
                source_id=sid,
                mime_type=mime0,
                size_bytes=media_obj.size,
            )
            return IngestionResult(
                source_id=sid,
                events_created=0,
                modality=modality,
                warnings=[
                    "same content already indexed at new path "
                    "— treated as moved/copied file, skipped"
                ],
            )

        # 2. Metadata extraction (incl. capture time — never import time)
        stage("metadata")
        mime = mimetypes.guess_type(str(path))[0]
        size = path.stat().st_size
        duration = None
        if modality in (Modality.AUDIO, Modality.VIDEO):
            duration = ffprobe_duration(path)
            if duration is None:
                warnings.append("ffprobe unavailable: media duration unknown")
        captured_iso, time_source = capture_time_for(path)
        self.db.upsert_source(
            sid,
            str(path),
            path.name,
            modality,
            mime_type=mime,
            size_bytes=size,
            duration_seconds=duration,
            content_hash=chash,
            captured_at=captured_iso,
            file_modified_at=_mtime_iso(path),
        )
        # Persist the canonical media reference (bytes stay on disk).
        self.db.add_media_asset(
            f"asset-{sid}",
            media_obj.key,
            media_obj.sha256,
            "original",
            source_id=sid,
            mime_type=mime,
            size_bytes=media_obj.size,
            duration_seconds=duration,
        )
        done("metadata")

        # 3+4. Modality processing & chunking
        stage("process")
        chunks: list[Chunk] = []
        if modality == Modality.PDF:
            self.scheduler.route("pdf_parse")
            pages = extract_pdf_pages(path)
            chunks = chunk_pdf_pages(pages)
            self.db.upsert_source(
                sid,
                str(path),
                path.name,
                modality,
                mime_type=mime,
                size_bytes=size,
                page_count=len(pages),
            )
        elif modality == Modality.DOCUMENT:
            self.scheduler.route("doc_parse")
            ext = path.suffix.lower()
            if ext == ".docx":
                text = extract_docx(path)
                chunks = chunk_document_text(text)
            elif ext == ".pptx":
                slides = extract_pptx(path)
                for n, s_text in slides:
                    for c in chunk_document_text(s_text, min_words=5, max_words=200):
                        c.page = n
                        c.section = f"Slide {n}"
                        chunks.append(c)
            else:
                text = extract_plain_text(path)
                chunks = chunk_document_text(text)
        elif modality == Modality.AUDIO:
            if self.transcriber is None:
                raise MissingDependency(
                    "No ASR model installed. Set up Whisper-Base (ONNX/QNN) via "
                    "scripts/setup_models.py to transcribe audio."
                )
            self.scheduler.route("asr")
            wav = path
            if path.suffix.lower() != ".wav":
                wav = extract_audio_to_wav(path, self.data_dir)
            segments = self.transcriber.transcribe(wav)
            chunks = chunk_transcript(segments)
            self.db.upsert_source(
                sid,
                str(path),
                path.name,
                modality,
                mime_type=mime,
                size_bytes=size,
                duration_seconds=duration or (segments[-1].end if segments else None),
            )
        elif modality == Modality.VIDEO:
            chunks = self._ingest_video(path, sid, warnings, media_obj)
        elif modality == Modality.IMAGE:
            if self.ocr is None:
                raise MissingDependency(
                    "No OCR model installed. Set up OCR (ONNX/QNN) via scripts/setup_models.py."
                )
            self.scheduler.route("ocr")
            for rec in self.ocr.recognize(path):
                c = Chunk(
                    text=rec["text"],
                    page=rec.get("page"),
                    metadata={"confidence": rec.get("confidence")},
                )
                if rec.get("bbox"):
                    b = rec["bbox"]
                    c.metadata["bbox"] = b
                chunks.append(c)
            if not chunks:
                warnings.append("OCR produced no text for this image")
        else:
            raise UnsupportedFormat(f"No extractor for modality {modality}")
        done("process")

        stage("chunk")
        if not chunks:
            warnings.append("No textual content extracted; nothing indexed")
        done("chunk")

        # 5. MemoryEvent creation
        stage("events")
        events: list[MemoryEvent] = []
        for c in chunks:
            ev = MemoryEvent(
                source_id=sid,
                modality=modality,
                content=c.text,
                timestamp_start=c.timestamp_start,
                timestamp_end=c.timestamp_end,
                page=c.page,
                section=c.section,
                location=Region(*c.metadata["bbox"]) if c.metadata.get("bbox") else None,
                concepts=list(c.concepts),
                metadata=dict(c.metadata),
            )
            if modality == Modality.IMAGE and ev.location is None:
                ev.metadata.setdefault("whole_image", True)
            # -- Phase 14 temporal semantics --------------------------------
            # Timestamped media: derive real event time from the source's
            # capture time + segment offset (never import time).
            if captured_iso and ev.timestamp_start is not None:
                try:
                    base = datetime.fromisoformat(captured_iso)
                    st = base + timedelta(seconds=float(ev.timestamp_start))
                    en = base + timedelta(seconds=float(ev.timestamp_end or ev.timestamp_start))
                    ev.event_time_start = st.isoformat()
                    ev.event_time_end = en.isoformat()
                    ev.time_source = f"{time_source}+transcript_offset"
                    ev.time_confidence = 0.9
                except (ValueError, TypeError):
                    pass
            elif captured_iso:
                ev.captured_at = captured_iso
                ev.event_time_start = captured_iso
                ev.time_source = time_source
                ev.time_confidence = 0.7 if time_source == "filesystem_mtime" else 0.95
            ev.modified_at = _mtime_iso(path)
            # reproducibility: trace every memory to its pipeline/model versions
            ev.extraction_version = EXTRACTION_VERSION
            ev.embedding_version = self.embedder.model_id
            ev.pipeline_version = PIPELINE_VERSION
            events.append(ev)
        done("events")

        # 6. Embedding generation (batched)
        stage("embed")
        if events:
            self.scheduler.route("embedding")
            texts = [e.content for e in events]
            vecs = self.embedder.embed(texts)
            for e, v in zip(events, vecs, strict=True):
                e.embedding = [float(x) for x in v]
                e.metadata["embedding_model"] = self.embedder.model_id
        done("embed")

        # 7. Persistence + 8. Vector indexing
        stage("index")
        n = self.db.add_events(events)
        import numpy as np

        ids = [e.id for e in events if e.embedding]
        mat = (
            np.asarray([e.embedding for e in events if e.embedding], dtype=np.float32)
            if ids
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
        if ids:
            self.index.add(ids, mat)
        done("index")

        # 9. Relationships where useful (same-source sequential + topic overlap)
        self._link_events(events)

        return IngestionResult(
            source_id=sid, events_created=n, modality=modality, warnings=warnings
        )

    # ------------------------------------------------------------------
    def _ingest_video(
        self, path: Path, sid: str, warnings: list[str], media_obj: MediaObject
    ) -> list[Chunk]:
        """Video -> audio track (ASR) + adaptive keyframes (scene change) + OCR."""
        from .video.keyframes import adaptive_keyframes  # local import: optional deps

        chunks: list[Chunk] = []
        frames_dir = self.data_dir / f"{sid}_frames"
        try:
            wav = extract_audio_to_wav(path, self.data_dir)
            if self.transcriber is not None:
                self.scheduler.route("asr")
                segments = self.transcriber.transcribe(wav)
                chunks.extend(chunk_transcript(segments))
            else:
                warnings.append("ASR model missing: video transcript skipped")
        except MissingDependency as e:
            warnings.append(str(e))
        try:
            self.scheduler.route("ocr")
            keyframes = adaptive_keyframes(path, frames_dir)
            for kf in keyframes:
                texts: list[str] = []
                if self.ocr is not None:
                    for rec in self.ocr.recognize(kf.path):
                        texts.append(rec["text"])
                # Persist the selected frame as a derived asset in the
                # MediaStore (content-addressed, survives temp cleanup).
                frame_key = None
                try:
                    kf_path = Path(kf.path)
                    if kf_path.is_file():
                        data = kf_path.read_bytes()
                        deriv = self.media_store.create_derivative(
                            media_obj.sha256, data, f"keyframe-{kf.frame_id}", extension=".jpg"
                        )
                        frame_key = deriv.key
                        self.db.add_media_asset(
                            f"asset-{sid}-{kf.frame_id}",
                            deriv.key,
                            deriv.sha256,
                            "derived",
                            source_id=sid,
                            mime_type="image/jpeg",
                            size_bytes=deriv.size,
                            parent_asset_id=f"asset-{sid}",
                            metadata={"timestamp": kf.time, "frame_id": kf.frame_id},
                        )
                except OSError:
                    frame_key = None  # frame file unreadable; keep event text-only
                chunks.append(
                    Chunk(
                        text="\n".join(t for t in texts if t) or f"[keyframe at {kf.time:.1f}s]",
                        timestamp_start=kf.time,
                        timestamp_end=kf.time + kf.duration,
                        metadata={
                            "keyframe": True,
                            "frame_id": kf.frame_id,
                            "score": round(kf.score, 3),
                            "media_key": frame_key,
                        },
                    )
                )
        except MissingDependency as e:
            warnings.append(f"Keyframe extraction skipped: {e}")
        return chunks

    def _link_events(self, events: list[MemoryEvent]) -> None:
        """SQLite relationships only where they add discovery value."""
        prev = None
        for ev in events:
            if prev is not None:
                self.db.add_relationship(prev.id, ev.id, "follows", 1.0)
            prev = ev
        # same_topic via concept/concept-token overlap between neighbors
        for i in range(len(events)):
            for j in range(i + 1, min(i + 6, len(events))):
                a, b = (
                    set(_content_tokens(events[i].content)),
                    set(_content_tokens(events[j].content)),
                )
                if a and b and len(a & b) / max(1, len(a | b)) > 0.25:
                    self.db.add_relationship(events[i].id, events[j].id, "same_topic", 0.6)


def _content_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]{4,}", text.lower())
