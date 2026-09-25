"""Background job queue + worker architecture (never block the UI).

Threaded queue with typed jobs.  Each job exposes:
job_id, status, progress, current_stage, elapsed_time, error, cancellation.
Workers register handlers per job kind (pdf/document/asr/ocr/vision/
embedding/index) so ingestion stages run off the UI thread.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any

log = logging.getLogger("reminiscence.workers")


def _pkg_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:
        return "0.0.0"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    stages: list[str]
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0..1 overall
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    error: str | None = None
    result: Any = None
    started_at: float | None = None
    finished_at: float | None = None
    cancelled: bool = False
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 0

    def elapsed_time(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 3)

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "current_stage": self.current_stage,
            "completed_stages": list(self.completed_stages),
            "stages": list(self.stages),
            "elapsed_time": self.elapsed_time(),
            "error": self.error,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "priority": self.priority,
            "cancellation_state": "cancelled" if self.cancelled else "active",
        }


HandlerFn = Callable[["Job", "JobContext"], Any]


class JobContext:
    """Passed to handlers: report progress/stage, check cancellation."""

    def __init__(self, job: Job, queue: JobQueue):
        self.job = job
        self._queue = queue

    def stage(self, name: str) -> None:
        j = self.job
        if name in j.stages and name not in j.completed_stages:
            j.current_stage = name
            idx = j.stages.index(name)
            j.progress = idx / max(1, len(j.stages))
            self._queue.notify(j)

    def complete_stage(self, name: str) -> None:
        j = self.job
        if name in j.stages and name not in j.completed_stages:
            j.completed_stages.append(name)
            j.progress = len(j.completed_stages) / max(1, len(j.stages))
            self._queue.notify(j)

    def set_progress(self, p: float) -> None:
        self.job.progress = max(0.0, min(1.0, p))
        self._queue.notify(self.job)

    def cancelled(self) -> bool:
        return self.job.cancelled

    def check_cancelled(self) -> None:
        if self.job.cancelled:
            raise JobCancelled(f"Job {self.job.id} cancelled")


class JobCancelled(Exception):
    pass


class JobQueue:
    """Threaded job queue with optional durable SQLite persistence.

    When a ``Database`` is supplied, every job state transition is mirrored
    into the ``jobs`` table (payload, attempts, progress, stage).  On restart,
    :meth:`recover` re-queues unfinished work so ingestion survives crashes.
    Failed jobs are retried with exponential backoff up to ``max_attempts``;
    handlers should be idempotent and skip stages already in
    ``job.completed_stages``.
    """

    def __init__(
        self,
        n_workers: int = 2,
        on_update: Callable[[dict], None] | None = None,
        db: Any | None = None,
        worker_version: str = "",
    ):
        self._q: Queue[tuple[int, float, int, Job]] = Queue()
        self._handlers: dict[str, HandlerFn] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self.on_update = on_update  # UI callback (thread-safe sink)
        self._db = db
        self.worker_version = worker_version or f"reminiscence-{_pkg_version()}"
        self._seq = itertools.count()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        for i in range(n_workers):
            t = threading.Thread(target=self._worker_loop, name=f"job-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # ------------------------------------------------------------------
    def register_handler(self, kind: str, fn: HandlerFn) -> None:
        self._handlers[kind] = fn

    def submit(
        self,
        kind: str,
        payload: dict,
        stages: list[str],
        priority: int = 0,
        max_attempts: int = 3,
        source_path: str = "",
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            kind=kind,
            payload=payload,
            stages=list(stages),
            priority=priority,
            max_attempts=max_attempts,
        )
        with self._lock:
            self._jobs[job.id] = job
        if self._db is not None:
            try:
                self._db.create_job(
                    job.id,
                    kind,
                    stages,
                    source_path=source_path or str(payload.get("path", "")),
                    payload={"priority": priority, **payload},
                    priority=priority,
                    max_attempts=max_attempts,
                    worker_version=self.worker_version,
                )
            except Exception:
                log.exception("failed to persist job %s", job.id)
        self._enqueue(job, delay=0.0)
        self.notify(job)
        return job

    def _enqueue(self, job: Job, delay: float) -> None:
        # priority DESC first, then FIFO among equal priorities
        self._q.put((-job.priority, time.time() + delay, next(self._seq), job))

    def recover(self) -> list[Job]:
        """Re-queue persisted unfinished jobs after a process restart."""
        if self._db is None:
            return []
        recovered: list[Job] = []
        for row in self._db.recover_jobs():
            job = Job(
                id=row["id"],
                kind=row["kind"],
                payload=json.loads(row["payload"] if "payload" in row.keys() else "{}"),
                stages=json.loads(row["stages"]),
                status=JobStatus.QUEUED,
                progress=row["progress"] or 0.0,
                current_stage=row["current_stage"],
                completed_stages=json.loads(row["completed_stages"] or "[]"),
                attempts=row["attempts"] if "attempts" in row.keys() else 0,
                max_attempts=row["max_attempts"] if "max_attempts" in row.keys() else 3,
                priority=row["priority"] if "priority" in row.keys() else 0,
            )
            with self._lock:
                self._jobs[job.id] = job
            self._db.update_job(job.id, status="queued")
            self._enqueue(job, delay=0.0)
            recovered.append(job)
        if recovered:
            log.info("recovered %d unfinished job(s) from previous run", len(recovered))
        return recovered

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.cancelled = True
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.error = "cancelled before start"
        self._persist(job)
        self.notify(job)
        return True

    def _persist(self, job: Job) -> None:
        if self._db is None:
            return
        try:
            self._db.update_job(
                job.id,
                status=job.status.value,
                progress=job.progress,
                current_stage=job.current_stage,
                completed_stages=list(job.completed_stages),
                error=job.error,
                attempts=job.attempts,
            )
        except Exception:
            log.exception("failed to persist job state for %s", job.id)

    def notify(self, job: Job) -> None:
        self._persist(job)
        if self.on_update:
            try:
                self.on_update(job.snapshot())
            except Exception:
                log.exception("UI update callback failed")

    def stats(self) -> list[dict]:
        with self._lock:
            return [
                j.snapshot()
                for j in sorted(self._jobs.values(), key=lambda x: x.id, reverse=True)[:50]
            ]

    def shutdown(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                _prio, ready_at, _seq, job = self._q.get(timeout=0.2)
            except Empty:
                continue
            wait = ready_at - time.time()
            if wait > 0:
                # retry backoff: park without consuming a worker slot forever
                self._q.put((_prio, ready_at, _seq, job))
                time.sleep(min(wait, 0.1))
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        handler = self._handlers.get(job.kind)
        job.started_at = time.time()
        if job.cancelled:
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            self.notify(job)
            return
        if handler is None:
            job.status = JobStatus.FAILED
            job.error = f"No worker registered for job kind '{job.kind}'"
            job.finished_at = time.time()
            self.notify(job)
            return
        job.status = JobStatus.RUNNING
        job.attempts += 1
        self.notify(job)
        ctx = JobContext(job, self)
        retryable = True
        try:
            job.result = handler(job, ctx)
            if not job.cancelled:
                job.status = JobStatus.DONE
                job.progress = 1.0
        except JobCancelled:
            job.status = JobStatus.CANCELLED
            retryable = False
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            log.error(
                "job %s failed (attempt %d/%d):\n%s",
                job.id,
                job.attempts,
                job.max_attempts,
                traceback.format_exc(),
            )
            if job.attempts < job.max_attempts and not job.cancelled:
                job.status = JobStatus.RETRYING
                delay = min(60.0, 2.0 ** (job.attempts - 1))
                self.notify(job)
                self._enqueue(job, delay=delay)
                return
            job.status = JobStatus.FAILED
            retryable = False
        finally:
            if job.status != JobStatus.RETRYING:
                job.finished_at = time.time()
            self.notify(job)
        del retryable
