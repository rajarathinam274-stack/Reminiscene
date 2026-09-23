"""Core application service: wires storage, AI, ingestion, retrieval, evidence.

This is the headless 'engine' the PySide6 UI binds to.  It never touches the
network and never blocks callers on long work (ingestion goes through the
background JobQueue).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config.settings import Settings, get_settings
from ...ai.embeddings.embedder import Embedder, HashingTFIDFEmbedder, get_embedder
from ...ai.registry import ModelRegistry
from ...ai.scheduler import AIWorkloadScheduler, SchedulerConfig
from ...evidence.generator import (Answer, AnswerGenerator,
                                   ExtractiveGroundedAnswerer)
from ...evidence.resolver import EvidenceResolver
from ...ingestion.pipeline import IngestionPipeline, IngestionResult
from ...memory.events import MemoryEvent
from ...retrieval.hybrid import Candidate, HybridRetriever, RetrievalWeights
from ...retrieval.query_classifier import ClassifiedQuery, QueryClassifier
from ...storage.database import Database
from ...storage.vector_index import NumpyVectorIndex, VectorIndex
from ...workers.queue import Job, JobQueue


@dataclass
class EnginePaths:
    data_dir: Path
    db_path: Path
    models_dir: Path

    @staticmethod
    def default() -> "EnginePaths":
        s = get_settings()
        return EnginePaths(s.data_dir, s.db_path, s.models_dir)


class ReminiscenceEngine:
    def __init__(
        self,
        paths: Optional[EnginePaths] = None,
        embedder: Optional[Embedder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        registry: Optional[ModelRegistry] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        settings: Optional[Settings] = None,
    ):
        self.paths = paths or EnginePaths.default()
        settings: Settings = settings or get_settings()
        self.settings = settings
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.paths.db_path)
        self.embedder = embedder or get_embedder(
            prefer_model_path=settings.embedding_model_path,
            dim=settings.embedding_dim)
        self.index: VectorIndex = NumpyVectorIndex(dim=self.embedder.dim)
        self.registry = registry or ModelRegistry()
        scheduler_config = scheduler_config or SchedulerConfig(
            allow_vlm=settings.allow_vlm,
            prefer_npu_tasks=list(settings.prefer_npu_tasks))
        self.scheduler = AIWorkloadScheduler(self.registry, scheduler_config)
        self.classifier = QueryClassifier()
        self.retriever = HybridRetriever(
            self.db, self.index, self.embedder, weights=self._retrieval_weights())
        self.evidence = EvidenceResolver(self.db)
        self.answerer = answer_generator or ExtractiveGroundedAnswerer()
        self.jobs = JobQueue(n_workers=settings.n_workers)
        self.pipeline = IngestionPipeline(
            db=self.db, index=self.index, embedder=self.embedder,
            scheduler=self.scheduler, data_dir=self.paths.data_dir,
        )
        self.jobs.register_handler("ingest", self._ingest_handler)
        self._restore_index()

    # ------------------------------------------------------------------
    def _retrieval_weights(self) -> RetrievalWeights:
        """Env vars override persisted DB settings, which override defaults."""
        s = self.settings
        env_set = any(
            f"RETRIEVAL_{k.upper()}" in os.environ
            for k in ("alpha", "beta", "gamma", "delta", "epsilon"))
        if env_set:
            return RetrievalWeights(
                alpha=s.retrieval_alpha, beta=s.retrieval_beta,
                gamma=s.retrieval_gamma, delta=s.retrieval_delta,
                epsilon=s.retrieval_epsilon)
        return RetrievalWeights.from_db(self.db)

    # ------------------------------------------------------------------
    def _restore_index(self) -> None:
        pairs = self.db.events_with_embeddings()
        if not pairs:
            return
        ids = [p[0] for p in pairs]
        mat = np.asarray([p[1] for p in pairs], dtype=np.float32)
        self.index.add(ids, mat)
        logging.getLogger("reminiscence.engine").info("restored %d vectors", len(ids))

    # -- ingestion -------------------------------------------------------
    def submit_ingestion(self, path: str) -> Job:
        stages = ["identify", "metadata", "process", "chunk", "events", "embed", "index"]
        return self.jobs.submit("ingest", {"path": path}, stages)

    def _ingest_handler(self, job: Job, ctx) -> dict:
        res = self.pipeline.ingest(job.payload["path"], job=job, ctx=ctx)
        return {"source_id": res.source_id, "events": res.events_created,
                "warnings": res.warnings}

    def ingest_now(self, path: str) -> IngestionResult:
        """Synchronous variant used by tests/scripts."""
        return self.pipeline.ingest(path)

    # -- search / recall ---------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[Candidate]:
        cq = self.classifier.classify(query)
        return self.retriever.retrieve(cq, top_k=top_k)

    def ask(self, question: str, max_evidence: int = 5) -> Answer:
        """Full loop: classify -> hybrid retrieve -> evidence pack -> local answer."""
        candidates = self.search(question, top_k=max_evidence * 2)
        events = [c.event for c in candidates]
        evidences = self.evidence.resolve_many(events)[:max_evidence]
        ans = self.answerer.generate(question, evidences)
        return ans

    # -- memory browsing / timeline -----------------------------------------
    def recent_memories(self, limit: int = 20) -> list[MemoryEvent]:
        return self.db.recent_events(limit)

    def sources(self):
        return self.db.list_sources()

    def delete_source(self, source_id: str) -> int:
        rows = self.db._conn.execute(
            "SELECT id FROM memory_events WHERE source_id=?", (source_id,)).fetchall()
        ids = [r["id"] for r in rows]
        self.index.remove(ids)
        return self.db.delete_source(source_id)

    def timeline(self, start_iso: str, end_iso: str) -> list[MemoryEvent]:
        return self.db.events_in_range(start_iso, end_iso)

    def related(self, event_id: str) -> list[tuple[str, str, float]]:
        return self.db.related(event_id)

    # -- status / privacy -----------------------------------------------------
    def offline_status(self) -> dict:
        """No telemetry, no cloud deps — core workflow is fully local."""
        return {
            "mode": "local-first",
            "cloud_dependencies": False,
            "telemetry_enabled": self.db.get_setting("telemetry", False),
            "npu_verified": self.scheduler.npu_available,
            "embedding_model": self.embedder.model_id,
            "data_dir": str(self.paths.data_dir),
        }

    def performance_snapshot(self) -> dict:
        return {
            "routing_decisions": self.scheduler.describe(),
            "index_size": len(self.index),
            "recent_benchmarks": [dict(r) for r in self.db.benchmark_history(10)],
        }

    def close(self) -> None:
        self.jobs.shutdown()
        self.db.close()
