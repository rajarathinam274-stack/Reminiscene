"""SQLite persistence with FTS5 full-text search and simple migrations.

Logical tables: sources, memory_events, relationships, concepts, jobs,
models, settings, benchmark_runs.  Binary media stays on the filesystem;
only references are stored here.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..memory.events import MemoryEvent, Modality, Region

SCHEMA_VERSION = 5


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            modality TEXT NOT NULL,
            mime_type TEXT,
            size_bytes INTEGER,
            duration_seconds REAL,
            page_count INTEGER,
            imported_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            modality TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            embedding_model TEXT,
            timestamp_start REAL,
            timestamp_end REAL,
            page INTEGER,
            section TEXT,
            location TEXT,
            concepts TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            parent_event TEXT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_me_source ON memory_events(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_me_time ON memory_events(timestamp_start)",
        "CREATE INDEX IF NOT EXISTS idx_me_modality ON memory_events(modality)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_events_fts USING fts5(
            content,
            concepts,
            section,
            content='memory_events',
            content_rowid='rowid'
        )
        """,
        """
        CREATE TRIGGER IF NOT EXISTS me_ai AFTER INSERT ON memory_events BEGIN
            INSERT INTO memory_events_fts(rowid, content, concepts, section)
            VALUES (new.rowid, new.content, new.concepts, new.section);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS me_ad AFTER DELETE ON memory_events BEGIN
            INSERT INTO memory_events_fts(memory_events_fts, rowid, content, concepts, section)
            VALUES ('delete', old.rowid, old.content, old.concepts, old.section);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS me_au AFTER UPDATE ON memory_events BEGIN
            INSERT INTO memory_events_fts(memory_events_fts, rowid, content, concepts, section)
            VALUES ('delete', old.rowid, old.content, old.concepts, old.section);
            INSERT INTO memory_events_fts(rowid, content, concepts, section)
            VALUES (new.rowid, new.content, new.concepts, new.section);
        END
        """,
        """
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_memory TEXT NOT NULL,
            target_memory TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            UNIQUE(source_memory, target_memory, relationship_type)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS concepts (
            name TEXT PRIMARY KEY,
            kind TEXT,
            metadata TEXT DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            current_stage TEXT,
            stages TEXT DEFAULT '[]',
            completed_stages TEXT DEFAULT '[]',
            error TEXT,
            source_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            cancelled INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT,
            modality TEXT,
            task TEXT,
            model_path TEXT,
            format TEXT,
            quantization TEXT,
            runtime TEXT,
            execution_provider TEXT,
            supported_devices TEXT DEFAULT '[]',
            memory_requirement_mb REAL,
            expected_latency_ms REAL,
            license TEXT,
            status TEXT DEFAULT 'available',
            fallback_policy TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            machine TEXT,
            model_id TEXT,
            task TEXT,
            runtime TEXT,
            execution_provider TEXT,
            backend TEXT,
            input_desc TEXT,
            latency_ms REAL,
            throughput REAL,
            peak_memory_mb REAL,
            cpu_pct REAL,
            gpu_pct REAL,
            npu_pct REAL,
            fallback INTEGER DEFAULT 0,
            notes TEXT
        )
        """,
    ],
    # Phase 12/13: richer benchmark records.  Every result must be traceable
    # to the machine, runtime version, quantization and pipeline that produced
    # it, so reference/vendor numbers can never masquerade as ours.
    2: [
        "ALTER TABLE benchmark_runs ADD COLUMN model_version TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN quantization TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN input_size TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN p50_ms REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN p95_ms REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN iterations INTEGER",
        "ALTER TABLE benchmark_runs ADD COLUMN os_name TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN arch TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN cpu_model TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN ram_gb REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN runtime_version TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN available_providers TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN provider_used TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN accelerated INTEGER",
        "ALTER TABLE benchmark_runs ADD COLUMN fallback_reason TEXT",
        "ALTER TABLE benchmark_runs ADD COLUMN power_mw REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN thermal_celsius REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN battery_pct REAL",
        "ALTER TABLE benchmark_runs ADD COLUMN pipeline_version TEXT",
        "CREATE INDEX IF NOT EXISTS idx_bench_task ON benchmark_runs(task)",
        "CREATE INDEX IF NOT EXISTS idx_bench_provider ON benchmark_runs(execution_provider)",
    ],
    # Phase 14 (P2): temporal semantics, content identity, deduplication,
    # graph entities with evidence-backed relationships, versioning.
    # Non-destructive: old columns are kept; new columns are nullable so
    # existing rows remain valid (expand-only migration).
    3: [
        # -- source content identity (SHA-256 of bytes, not path) ----------
        "ALTER TABLE sources ADD COLUMN content_hash TEXT",
        "ALTER TABLE sources ADD COLUMN original_path TEXT",
        "ALTER TABLE sources ADD COLUMN current_path TEXT",
        "ALTER TABLE sources ADD COLUMN filename TEXT",
        "ALTER TABLE sources ADD COLUMN file_modified_at TEXT",
        "ALTER TABLE sources ADD COLUMN last_seen_at TEXT",
        "ALTER TABLE sources ADD COLUMN status TEXT DEFAULT 'active'",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_src_hash ON sources(content_hash)"
        " WHERE content_hash IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_src_current ON sources(current_path)",
        # -- memory event temporal semantics + versioning -------------------
        "ALTER TABLE memory_events ADD COLUMN event_time_start TEXT",
        "ALTER TABLE memory_events ADD COLUMN event_time_end TEXT",
        "ALTER TABLE memory_events ADD COLUMN captured_at TEXT",
        "ALTER TABLE memory_events ADD COLUMN modified_at TEXT",
        "ALTER TABLE memory_events ADD COLUMN time_source TEXT",
        "ALTER TABLE memory_events ADD COLUMN time_confidence REAL DEFAULT 1.0",
        "ALTER TABLE memory_events ADD COLUMN extraction_version TEXT",
        "ALTER TABLE memory_events ADD COLUMN embedding_version TEXT",
        "ALTER TABLE memory_events ADD COLUMN pipeline_version TEXT",
        "CREATE INDEX IF NOT EXISTS idx_me_event_time ON memory_events(event_time_start)",
        "CREATE INDEX IF NOT EXISTS idx_me_captured ON memory_events(captured_at)",
        # relationship provenance (method that inferred the edge)
        "ALTER TABLE relationships ADD COLUMN method TEXT",
        "ALTER TABLE relationships ADD COLUMN created_at TEXT",
        # -- lightweight graph entities --------------------------------------
        """
        CREATE TABLE IF NOT EXISTS persons (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            aliases TEXT DEFAULT '[]',
            confidence REAL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS places (
            id TEXT PRIMARY KEY,
            name TEXT,
            latitude REAL,
            longitude REAL,
            accuracy REAL,
            city TEXT,
            region TEXT,
            country TEXT,
            geocoding_source TEXT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events_entities (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            place_id TEXT REFERENCES places(id),
            description TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS entity_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL REFERENCES memory_events(id) ON DELETE CASCADE,
            entity_type TEXT NOT NULL,      -- person | place | event | concept
            entity_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,-- MENTIONS | OCCURRED_AT | PART_OF | ABOUT
            confidence REAL NOT NULL DEFAULT 1.0,
            method TEXT NOT NULL DEFAULT 'rule',   -- rule | ocr | transcript | manual | model
            evidence_id TEXT,               -- memory that justifies the link
            created_at TEXT NOT NULL,
            UNIQUE(memory_id, entity_type, entity_id, relationship_type)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_el_entity ON entity_links(entity_type, entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_el_memory ON entity_links(memory_id)",
        # -- inference telemetry (Phase 12/14): every real inference recorded
        """
        CREATE TABLE IF NOT EXISTS inference_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            task TEXT NOT NULL,
            runtime TEXT,
            provider TEXT,
            backend TEXT,
            started_at TEXT NOT NULL,
            duration_ms REAL,
            input_size TEXT,
            memory_mb REAL,
            success INTEGER NOT NULL DEFAULT 1,
            fallback_reason TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ir_model ON inference_records(model_id, started_at)",
    ],
    # Sprint 2 (MediaStore): canonical content-addressed media references.
    # Expand-only migration; no destructive changes to prior schema.
    4: [
        """
        CREATE TABLE IF NOT EXISTS media_assets (
            asset_id TEXT PRIMARY KEY,
            source_id TEXT REFERENCES sources(id) ON DELETE CASCADE,
            media_key TEXT NOT NULL,          -- relative key inside MediaStore
            sha256 TEXT NOT NULL,             -- content identity
            media_type TEXT NOT NULL,         -- original | thumbnail | derived
            mime_type TEXT,
            size_bytes INTEGER,
            width INTEGER,
            height INTEGER,
            duration_seconds REAL,
            frame_rate REAL,
            audio_channels INTEGER,
            parent_asset_id TEXT REFERENCES media_assets(asset_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ma_source ON media_assets(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_ma_sha ON media_assets(sha256)",
        "CREATE INDEX IF NOT EXISTS idx_ma_type ON media_assets(media_type)",
        "ALTER TABLE sources ADD COLUMN media_key TEXT",
    ],
    # Sprint 5 (persistent jobs): durable job state with retries, priority,
    # payload persistence and incremental-indexing version columns.
    5: [
        "ALTER TABLE jobs ADD COLUMN payload TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE jobs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 3",
        "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE jobs ADD COLUMN started_at TEXT",
        "ALTER TABLE jobs ADD COLUMN finished_at TEXT",
        "ALTER TABLE jobs ADD COLUMN worker_version TEXT",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
        # -- incremental re-indexing provenance ------------------------------
        "ALTER TABLE sources ADD COLUMN extraction_version TEXT",
        "ALTER TABLE sources ADD COLUMN embedding_model TEXT",
        "ALTER TABLE sources ADD COLUMN embedding_version TEXT",
        "ALTER TABLE sources ADD COLUMN pipeline_version TEXT",
    ],
}


def _vec_to_blob(v: Iterable[float]) -> bytes:
    import array

    a = array.array("f", v)
    return a.tobytes()


def _blob_to_vec(b: bytes | None) -> list[float] | None:
    if b is None:
        return None
    import array

    a = array.array("f")
    a.frombytes(b)
    return list(a)


class Database:
    """Thread-safe (single-writer lock) SQLite wrapper for REMINISCENCE."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._migrate()

    # -- migrations -----------------------------------------------------
    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.execute("PRAGMA user_version")
            version = cur.fetchone()[0]
            for target in sorted(_MIGRATIONS):
                if target > version:
                    for stmt in _MIGRATIONS[target]:
                        self._conn.execute(stmt)
                    self._conn.execute(f"PRAGMA user_version = {target}")
            self._conn.commit()

    def get_schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sources ----------------------------------------------------------
    def upsert_source(
        self,
        source_id: str,
        path: str,
        name: str,
        modality: Modality | str,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        duration_seconds: float | None = None,
        page_count: int | None = None,
        metadata: dict | None = None,
        content_hash: str | None = None,
        captured_at: str | None = None,
        file_modified_at: str | None = None,
        extraction_version: str | None = None,
        embedding_model: str | None = None,
        embedding_version: str | None = None,
        pipeline_version: str | None = None,
    ) -> str:
        """Insert or update a source.

        Phase 14: identity is *content-based* (SHA-256 ``content_hash``), not
        path-based.  Re-importing the same bytes from a new location updates
        ``current_path``/``last_seen_at`` on the existing row instead of
        creating a duplicate source — this handles moved and copied files.
        """
        modality = Modality(modality).value
        now = _utcnow()
        with self._lock:
            existing = None
            if content_hash:
                existing = self._conn.execute(
                    "SELECT id FROM sources WHERE content_hash=? AND id<>?",
                    (content_hash, source_id),
                ).fetchone()
            if existing:
                # Same content already known under another id: refresh its
                # location; do NOT create a second source.
                sid = existing["id"]
                self._conn.execute(
                    """UPDATE sources SET current_path=?, path=?, filename=?,
                       last_seen_at=?, metadata=? WHERE id=?""",
                    (path, path, name, now, json.dumps(metadata or {}), sid),
                )
                self._conn.commit()
                return sid
            self._conn.execute(
                """
                INSERT INTO sources(id, path, name, modality, mime_type, size_bytes,
                                    duration_seconds, page_count, imported_at, metadata,
                                    content_hash, original_path, current_path, filename,
                                    file_modified_at, last_seen_at, status,
                                    extraction_version, embedding_model,
                                    embedding_version, pipeline_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active',?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    path=excluded.path, name=excluded.name, modality=excluded.modality,
                    mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
                    duration_seconds=excluded.duration_seconds,
                    page_count=excluded.page_count, metadata=excluded.metadata,
                    content_hash=COALESCE(excluded.content_hash, sources.content_hash),
                    current_path=excluded.current_path,
                    filename=excluded.filename,
                    last_seen_at=excluded.last_seen_at,
                    extraction_version=COALESCE(excluded.extraction_version,
                                                sources.extraction_version),
                    embedding_model=COALESCE(excluded.embedding_model,
                                             sources.embedding_model),
                    embedding_version=COALESCE(excluded.embedding_version,
                                               sources.embedding_version),
                    pipeline_version=COALESCE(excluded.pipeline_version,
                                              sources.pipeline_version)
                """,
                (
                    source_id,
                    path,
                    name,
                    modality,
                    mime_type,
                    size_bytes,
                    duration_seconds,
                    page_count,
                    now,
                    json.dumps(metadata or {}),
                    content_hash,
                    path,
                    path,
                    name,
                    file_modified_at,
                    now,
                    extraction_version,
                    embedding_model,
                    embedding_version,
                    pipeline_version,
                ),
            )
            if captured_at:
                self._conn.execute(
                    "UPDATE sources SET metadata=json_set("
                    "CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END,"
                    "'$.captured_at', ?) WHERE id=?",
                    (captured_at, source_id),
                )
            self._conn.commit()
        return source_id

    def get_source_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        """Duplicate detection: find an already-known source by SHA-256."""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM sources WHERE content_hash=?", (content_hash,)
            ).fetchone()

    def get_source(self, source_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()

    def list_sources(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM sources ORDER BY imported_at DESC").fetchall()

    def delete_source(self, source_id: str) -> int:
        """Explicit delete control: removes source + its memory events."""
        with self._lock:
            ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM memory_events WHERE source_id=?", (source_id,)
                )
            ]
            for mid in ids:
                self._conn.execute("DELETE FROM memory_events WHERE id=?", (mid,))
            n = self._conn.execute("DELETE FROM sources WHERE id=?", (source_id,)).rowcount
            self._conn.commit()
        return n

    # -- media assets (Sprint 2 / MediaStore references) ---------------------
    def add_media_asset(
        self,
        asset_id: str,
        media_key: str,
        sha256: str,
        media_type: str,
        *,
        source_id: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
        frame_rate: float | None = None,
        audio_channels: int | None = None,
        parent_asset_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Persist a reference to one object stored in the MediaStore.

        Bytes live on disk; SQLite only keeps the relative key + metadata.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO media_assets(
                    asset_id, source_id, media_key, sha256, media_type, mime_type,
                    size_bytes, width, height, duration_seconds, frame_rate,
                    audio_channels, parent_asset_id, created_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    asset_id,
                    source_id,
                    media_key,
                    sha256,
                    media_type,
                    mime_type,
                    size_bytes,
                    width,
                    height,
                    duration_seconds,
                    frame_rate,
                    audio_channels,
                    parent_asset_id,
                    _utcnow(),
                    json.dumps(metadata or {}),
                ),
            )
            if source_id is not None and media_type == "original":
                self._conn.execute(
                    "UPDATE sources SET media_key=? WHERE id=?", (media_key, source_id)
                )
            self._conn.commit()
        return asset_id

    def get_media_assets(self, source_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM media_assets WHERE source_id=? ORDER BY created_at",
                (source_id,),
            ).fetchall()

    def list_media_keys(self) -> list[str]:
        """All referenced media keys (for orphan detection against MediaStore)."""
        with self._lock:
            return [
                r["media_key"] for r in self._conn.execute("SELECT media_key FROM media_assets")
            ]

    def find_source_by_media_key(self, media_key: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT s.* FROM sources s JOIN media_assets m ON m.source_id=s.id "
                "WHERE m.media_key=? LIMIT 1",
                (media_key,),
            ).fetchone()

    # -- memory events ------------------------------------------------------
    def add_event(self, ev: MemoryEvent) -> str:
        if ev.created_at is None:
            ev.created_at = _utcnow()
        loc = json.dumps(ev.location.to_dict()) if ev.location else None
        emb_blob = _vec_to_blob(ev.embedding) if ev.embedding else None
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_events(
                    id, source_id, modality, content, embedding, embedding_model,
                    timestamp_start, timestamp_end, page, section, location,
                    concepts, metadata, parent_event, created_at,
                    event_time_start, event_time_end, captured_at, modified_at,
                    time_source, time_confidence,
                    extraction_version, embedding_version, pipeline_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ev.id,
                    ev.source_id,
                    Modality(ev.modality).value,
                    ev.content,
                    emb_blob,
                    ev.metadata.get("embedding_model"),
                    ev.timestamp_start,
                    ev.timestamp_end,
                    ev.page,
                    ev.section,
                    loc,
                    json.dumps(ev.concepts),
                    json.dumps(ev.metadata),
                    ev.parent_event,
                    ev.created_at,
                    ev.event_time_start,
                    ev.event_time_end,
                    ev.captured_at,
                    ev.modified_at,
                    ev.time_source,
                    ev.time_confidence,
                    ev.extraction_version,
                    ev.embedding_version,
                    ev.pipeline_version,
                ),
            )
            self._conn.commit()
        return ev.id

    def add_events(self, events: Iterable[MemoryEvent]) -> int:
        n = 0
        for ev in events:
            self.add_event(ev)
            n += 1
        return n

    def get_event(self, event_id: str) -> MemoryEvent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_events WHERE id=?", (event_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
        keys = row.keys()
        return MemoryEvent(
            id=row["id"],
            source_id=row["source_id"],
            modality=Modality(row["modality"]),
            content=row["content"],
            embedding=_blob_to_vec(row["embedding"]),
            timestamp_start=row["timestamp_start"],
            timestamp_end=row["timestamp_end"],
            page=row["page"],
            section=row["section"],
            location=Region.from_dict(json.loads(row["location"])) if row["location"] else None,
            concepts=json.loads(row["concepts"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            parent_event=row["parent_event"],
            created_at=row["created_at"],
            event_time_start=row["event_time_start"] if "event_time_start" in keys else None,
            event_time_end=row["event_time_end"] if "event_time_end" in keys else None,
            captured_at=row["captured_at"] if "captured_at" in keys else None,
            modified_at=row["modified_at"] if "modified_at" in keys else None,
            time_source=row["time_source"] if "time_source" in keys else None,
            time_confidence=(
                row["time_confidence"]
                if "time_confidence" in keys and row["time_confidence"] is not None
                else 1.0
            ),
            extraction_version=row["extraction_version"] if "extraction_version" in keys else None,
            embedding_version=row["embedding_version"] if "embedding_version" in keys else None,
            pipeline_version=row["pipeline_version"] if "pipeline_version" in keys else None,
        )

    def all_events(self) -> list[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memory_events").fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_with_embeddings(self) -> list[tuple[str, list[float]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memory_events WHERE embedding IS NOT NULL"
            ).fetchall()
        return [(r["id"], _blob_to_vec(r["embedding"]) or []) for r in rows]

    def recent_events(self, limit: int = 20) -> list[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_in_range(self, start_iso: str, end_iso: str) -> list[MemoryEvent]:
        """Temporal lookup against *memory* time, not indexing time.

        A memory matches if its event/capture window overlaps [start, end].
        MemoryEvents without explicit temporal fields (legacy rows) fall back
        to created_at so old data remains reachable.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory_events
                WHERE COALESCE(event_time_start, captured_at, created_at) >= ?
                  AND COALESCE(event_time_start, captured_at, created_at) <= ?
                ORDER BY COALESCE(event_time_start, captured_at, created_at)
                """,
                (start_iso, end_iso),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_by_entity(self, entity_type: str, entity_id: str) -> list[MemoryEvent]:
        """Graph candidate channel: memories linked to an entity."""
        with self._lock:
            ids = [
                r["memory_id"]
                for r in self._conn.execute(
                    "SELECT memory_id FROM entity_links WHERE entity_type=? AND entity_id=?",
                    (entity_type, entity_id),
                )
            ]
            out = []
            for mid in ids:
                ev = self.get_event(mid)
                if ev is not None:
                    out.append(ev)
        return out

    # -- graph entities -------------------------------------------------------
    def upsert_person(
        self,
        person_id: str,
        display_name: str,
        aliases: list[str] | None = None,
        confidence: float = 1.0,
    ) -> str:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                """INSERT INTO persons(id, display_name, aliases, confidence,
                                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     display_name=excluded.display_name,
                     aliases=excluded.aliases,
                     confidence=excluded.confidence,
                     updated_at=excluded.updated_at""",
                (person_id, display_name, json.dumps(aliases or []), confidence, now, now),
            )
            self._conn.commit()
        return person_id

    def list_persons(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM persons ORDER BY display_name").fetchall()

    def upsert_place(
        self,
        place_id: str,
        name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        accuracy: float | None = None,
        city: str | None = None,
        region: str | None = None,
        country: str | None = None,
        geocoding_source: str | None = None,
    ) -> str:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO places(id, name, latitude, longitude, accuracy,
                        city, region, country, geocoding_source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    place_id,
                    name,
                    latitude,
                    longitude,
                    accuracy,
                    city,
                    region,
                    country,
                    geocoding_source,
                    _utcnow(),
                ),
            )
            self._conn.commit()
        return place_id

    def list_places(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM places").fetchall()

    def link_entity(
        self,
        memory_id: str,
        entity_type: str,
        entity_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        method: str = "rule",
        evidence_id: str | None = None,
    ) -> None:
        """Evidence-backed relationship: every inferred link stores the
        confidence and the method/evidence that justified it."""
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO entity_links(memory_id, entity_type, entity_id,
                       relationship_type, confidence, method, evidence_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    entity_type,
                    entity_id,
                    relationship_type,
                    confidence,
                    method,
                    evidence_id,
                    _utcnow(),
                ),
            )
            self._conn.commit()

    def entity_links_for(self, memory_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM entity_links WHERE memory_id=?", (memory_id,)
            ).fetchall()

    # -- inference telemetry ----------------------------------------------------
    def record_inference(
        self,
        model_id: str,
        task: str,
        runtime: str | None,
        provider: str | None,
        backend: str | None,
        duration_ms: float | None,
        input_size: str | None = None,
        memory_mb: float | None = None,
        success: bool = True,
        fallback_reason: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO inference_records(model_id, task, runtime, provider, backend,
                       started_at, duration_ms, input_size, memory_mb, success, fallback_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id,
                    task,
                    runtime,
                    provider,
                    backend,
                    _utcnow(),
                    duration_ms,
                    input_size,
                    memory_mb,
                    1 if success else 0,
                    fallback_reason,
                ),
            )
            self._conn.commit()
        return int(cur.lastrowid or -1)

    def inference_summary(self, limit: int = 200) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                """SELECT model_id, task, provider, COUNT(*) AS n,
                          AVG(duration_ms) AS avg_ms, MAX(duration_ms) AS max_ms
                   FROM inference_records GROUP BY model_id, task, provider
                   ORDER BY n DESC LIMIT ?""",
                (limit,),
            ).fetchall()

    # -- lexical (FTS5) ------------------------------------------------------
    @staticmethod
    def _fts_match_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression from free text.

        Each token becomes an independent quoted prefix term joined by
        implicit AND.  Previously the whole query was wrapped as one phrase
        prefix ("a b c"*), which made multi-word queries miss documents where
        the terms were present but not adjacent in exactly that order.
        """
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", query)
        if not tokens:
            return ""
        return " ".join('"' + t.replace('"', '""') + '"*' for t in tokens)

    def _fts_rows(self, match_expr: str, limit: int) -> list[sqlite3.Row]:
        """Execute one FTS5 MATCH query with bm25 ranking."""
        with self._lock:
            return self._conn.execute(
                """
                SELECT me.*, bm25(memory_events_fts, 1.0, 0.5, 0.3) AS rank
                FROM memory_events_fts
                JOIN memory_events me ON me.rowid = memory_events_fts.rowid
                WHERE memory_events_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()

    def fts_search(self, query: str, limit: int = 50) -> list[tuple[MemoryEvent, float]]:
        """Returns (event, bm25_score) — lower bm25 == better match.

        Robustness: strict AND over all terms is the first attempt; if it
        yields nothing we retry with OR so a single unmatched term (e.g. an
        interrogative like "take", or a year token that lives only in event
        time rather than content) cannot zero out an otherwise strong hit.
        bm25 still ranks documents containing more/rarer terms first.
        """
        if not query.strip():
            return []
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", query)
        if not tokens:
            return []
        quoted = [t.replace('"', '""') for t in tokens]
        and_expr = " ".join(f'"{t}"*' for t in quoted)
        rows = self._fts_rows(and_expr, limit)
        if not rows and len(tokens) > 1:
            or_expr = " OR ".join(f'"{t}"*' for t in quoted[:8])
            rows = self._fts_rows(or_expr, limit)
        return [(self._row_to_event(r), r["rank"]) for r in rows]

    # -- relationships ---------------------------------------------------------
    def add_relationship(self, src: str, dst: str, rel_type: str, confidence: float = 1.0) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO relationships
                   (source_memory, target_memory, relationship_type, confidence)
                   VALUES (?,?,?,?)""",
                (src, dst, rel_type, confidence),
            )
            self._conn.commit()

    def related(self, event_id: str) -> list[tuple[str, str, float]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT source_memory, target_memory, relationship_type, confidence
                   FROM relationships
                   WHERE source_memory=? OR target_memory=?""",
                (event_id, event_id),
            ).fetchall()
        return [(r["source_memory"], r["target_memory"], r["confidence"]) for r in rows]

    # -- jobs -------------------------------------------------------------------
    def create_job(
        self,
        job_id: str,
        kind: str,
        stages: list[str],
        source_path: str = "",
        payload: dict | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        worker_version: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs
                   (id, kind, status, progress, current_stage, stages, completed_stages,
                    source_path, created_at, updated_at, payload, attempts, max_attempts,
                    priority, worker_version)
                   VALUES (?,?,'queued',0,?,?, '[]',?,?,?,?,0,?,?,?)""",
                (
                    job_id,
                    kind,
                    stages[0] if stages else None,
                    json.dumps(stages),
                    source_path,
                    _utcnow(),
                    _utcnow(),
                    json.dumps(payload or {}),
                    max_attempts,
                    priority,
                    worker_version,
                ),
            )
            self._conn.commit()

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        current_stage: str | None = None,
        completed_stages: list[str] | None = None,
        error: str | None = None,
        attempts: int | None = None,
    ) -> None:
        sets, vals = ["updated_at=?"], [_utcnow()]
        if status is not None:
            sets.append("status=?")
            vals.append(status)
            if status == "running":
                sets.append("started_at=COALESCE(started_at, ?)")
                vals.append(_utcnow())
            elif status in ("done", "failed", "cancelled"):
                sets.append("finished_at=?")
                vals.append(_utcnow())
        if progress is not None:
            sets.append("progress=?")
            vals.append(progress)
        if current_stage is not None:
            sets.append("current_stage=?")
            vals.append(current_stage)
        if completed_stages is not None:
            sets.append("completed_stages=?")
            vals.append(json.dumps(completed_stages))
        if error is not None:
            sets.append("error=?")
            vals.append(error)
        if attempts is not None:
            sets.append("attempts=?")
            vals.append(attempts)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {','.join(sets)} WHERE id=?", (*vals, job_id))
            self._conn.commit()

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET cancelled=1, updated_at=? WHERE id=?", (_utcnow(), job_id)
            )
            self._conn.commit()

    # -- durable job recovery ---------------------------------------------------
    def recover_jobs(self) -> list[sqlite3.Row]:
        """Return jobs that must be re-queued after a process restart.

        Includes queued/retrying work plus RUNNING rows (a running row found
        at startup means the previous process died mid-job; it is resumable
        because handlers are expected to skip already-completed stages).
        Ordered by priority DESC, created_at ASC.
        """
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM jobs
                   WHERE status IN ('queued', 'retrying', 'running') AND cancelled = 0
                   ORDER BY priority DESC, created_at ASC"""
            ).fetchall()

    def mark_interrupted_running_failed(self, reason: str = "interrupted by shutdown") -> int:
        """Optional stricter recovery mode: fail orphaned running jobs."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? "
                "WHERE status='running' AND cancelled=0",
                (reason, _utcnow()),
            )
            self._conn.commit()
            return cur.rowcount

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def list_jobs(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()

    def is_job_cancelled(self, job_id: str) -> bool:
        row = self.get_job(job_id)
        return bool(row and row["cancelled"])

    # -- settings -----------------------------------------------------------
    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings(key,value) VALUES (?,?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- benchmark runs -------------------------------------------------------
    def record_benchmark(self, data: dict[str, Any]) -> int:
        cols = [
            "run_ts",
            "machine",
            "model_id",
            "task",
            "runtime",
            "execution_provider",
            "backend",
            "input_desc",
            "latency_ms",
            "throughput",
            "peak_memory_mb",
            "cpu_pct",
            "gpu_pct",
            "npu_pct",
            "fallback",
            "notes",
            # Phase 12/13 traceability fields (migration v2)
            "model_version",
            "quantization",
            "input_size",
            "p50_ms",
            "p95_ms",
            "iterations",
            "os_name",
            "arch",
            "cpu_model",
            "ram_gb",
            "runtime_version",
            "available_providers",
            "provider_used",
            "accelerated",
            "fallback_reason",
            "power_mw",
            "thermal_celsius",
            "battery_pct",
            "pipeline_version",
        ]
        vals = [data.get(c) for c in cols]
        placeholders = ",".join("?" * len(cols))
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO benchmark_runs ({','.join(cols)}) VALUES ({placeholders})", vals
            )
            self._conn.commit()
        return int(cur.lastrowid or -1)

    def benchmark_history(self, limit: int = 100) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY run_ts DESC LIMIT ?", (limit,)
            ).fetchall()
