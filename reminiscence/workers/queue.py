"""Background job queue + worker architecture (never block the UI).

Threaded queue with typed jobs.  Each job exposes:
job_id, status, progress, current_stage, elapsed_time, error, cancellation.
Workers register handlers per job kind (pdf/document/asr/ocr/vision/
embedding/index) so ingestion stages run off the UI thread.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Queue, Empty
from typing import Any, Callable, Optional

log = logging.getLogger("reminiscence.workers")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
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
    progress: float = 0.0                    # 0..1 overall
    current_stage: Optional[str] = None
    completed_stages: list[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Any = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancelled: bool = False

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
            "cancellation_state": "cancelled" if self.cancelled else "active",
        }


HandlerFn = Callable[["Job", "JobContext"], Any]


class JobContext:
    """Passed to handlers: report progress/stage, check cancellation."""

    def __init__(self, job: Job, queue: "JobQueue"):
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
    def __init__(self, n_workers: int = 2, on_update: Optional[Callable[[dict], None]] = None):
        self._q: "Queue[Job]" = Queue()
        self._handlers: dict[str, HandlerFn] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self.on_update = on_update      # UI callback (thread-safe sink)
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        for i in range(n_workers):
            t = threading.Thread(target=self._worker_loop, name=f"job-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # ------------------------------------------------------------------
    def register_handler(self, kind: str, fn: HandlerFn) -> None:
        self._handlers[kind] = fn

    def submit(self, kind: str, payload: dict, stages: list[str]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, payload=payload, stages=list(stages))
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        self.notify(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
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
            self.notify(job)
            return True

    def notify(self, job: Job) -> None:
        if self.on_update:
            try:
                self.on_update(job.snapshot())
            except Exception:
                log.exception("UI update callback failed")

    def stats(self) -> list[dict]:
        with self._lock:
            return [j.snapshot() for j in sorted(self._jobs.values(),
                                                 key=lambda x: x.id, reverse=True)[:50]]

    def shutdown(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except Empty:
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
        self.notify(job)
        ctx = JobContext(job, self)
        try:
            job.result = handler(job, ctx)
            if not job.cancelled:
                job.status = JobStatus.DONE
                job.progress = 1.0
        except JobCancelled:
            job.status = JobStatus.CANCELLED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {e}"
            log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        finally:
            job.finished_at = time.time()
            self.notify(job)
