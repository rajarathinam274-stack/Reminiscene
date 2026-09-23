"""SQLite persistence with FTS5 full-text search and simple migrations.

Logical tables: sources, memory_events, relationships, concepts, jobs,
models, settings, benchmark_runs.  Binary media stays on the filesystem;
only references are stored here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..memory.events import MemoryEvent, Modality, Region

SCHEMA_VERSION = 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    ]
}


def _vec_to_blob(v: Iterable[float]) -> bytes:
    import array

    a = array.array("f", v)
    return a.tobytes()


def _blob_to_vec(b: Optional[bytes]) -> Optional[list[float]]:
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
        mime_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        page_count: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        modality = Modality(modality).value
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sources(id, path, name, modality, mime_type, size_bytes,
                                    duration_seconds, page_count, imported_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    path=excluded.path, name=excluded.name, modality=excluded.modality,
                    mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
                    duration_seconds=excluded.duration_seconds,
                    page_count=excluded.page_count, metadata=excluded.metadata
                """,
                (
                    source_id, path, name, modality, mime_type, size_bytes,
                    duration_seconds, page_count, _utcnow(),
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return source_id

    def get_source(self, source_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()

    def list_sources(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM sources ORDER BY imported_at DESC").fetchall()

    def delete_source(self, source_id: str) -> int:
        """Explicit delete control: removes source + its memory events."""
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM memory_events WHERE source_id=?", (source_id,)
            )]
            for mid in ids:
                self._conn.execute("DELETE FROM memory_events WHERE id=?", (mid,))
            n = self._conn.execute("DELETE FROM sources WHERE id=?", (source_id,)).rowcount
            self._conn.commit()
        return n

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
                    concepts, metadata, parent_event, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ev.id, ev.source_id, Modality(ev.modality).value, ev.content,
                    emb_blob, ev.metadata.get("embedding_model"),
                    ev.timestamp_start, ev.timestamp_end, ev.page, ev.section, loc,
                    json.dumps(ev.concepts), json.dumps(ev.metadata),
                    ev.parent_event, ev.created_at,
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

    def get_event(self, event_id: str) -> Optional[MemoryEvent]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memory_events WHERE id=?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
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
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_events WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
                (start_iso, end_iso),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # -- lexical (FTS5) ------------------------------------------------------
    def fts_search(self, query: str, limit: int = 50) -> list[tuple[MemoryEvent, float]]:
        """Returns (event, bm25_score) — lower bm25 == better match."""
        if not query.strip():
            return []
        safe = '"' + query.replace('"', '""') + '"*'
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT me.*, bm25(memory_events_fts, 1.0, 0.5, 0.3) AS rank
                FROM memory_events_fts
                JOIN memory_events me ON me.rowid = memory_events_fts.rowid
                WHERE memory_events_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe, limit),
            ).fetchall()
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
    def create_job(self, job_id: str, kind: str, stages: list[str], source_path: str = "") -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs
                   (id, kind, status, progress, current_stage, stages, completed_stages,
                    source_path, created_at, updated_at)
                   VALUES (?,?,'queued',0,?,?, '[]',?,?,?)""",
                (job_id, kind, stages[0] if stages else None, json.dumps(stages),
                 source_path, _utcnow(), _utcnow()),
            )
            self._conn.commit()

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        completed_stages: Optional[list[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        sets, vals = ["updated_at=?"], [_utcnow()]
        if status is not None:
            sets.append("status=?"); vals.append(status)
        if progress is not None:
            sets.append("progress=?"); vals.append(progress)
        if current_stage is not None:
            sets.append("current_stage=?"); vals.append(current_stage)
        if completed_stages is not None:
            sets.append("completed_stages=?"); vals.append(json.dumps(completed_stages))
        if error is not None:
            sets.append("error=?"); vals.append(error)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {','.join(sets)} WHERE id=?", (*vals, job_id))
            self._conn.commit()

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET cancelled=1 WHERE id=?", (job_id,))
            self._conn.commit()

    def get_job(self, job_id: str) -> Optional[sqlite3.Row]:
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
        cols = ["run_ts", "machine", "model_id", "task", "runtime", "execution_provider",
                "backend", "input_desc", "latency_ms", "throughput", "peak_memory_mb",
                "cpu_pct", "gpu_pct", "npu_pct", "fallback", "notes"]
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
