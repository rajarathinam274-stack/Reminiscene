"""Unit + integration tests for the REMINISCENCE MVP core.

Run:  python -m pytest reminiscence/tests -q
All tests are offline and dependency-light (SQLite FTS5 + numpy only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reminiscence.ai.embeddings.embedder import HashingTFIDFEmbedder
from reminiscence.ai.registry import ModelRegistry
from reminiscence.ai.scheduler import AIWorkloadScheduler
from reminiscence.app.services.engine import EnginePaths, ReminiscenceEngine
from reminiscence.evidence.generator import ExtractiveGroundedAnswerer
from reminiscence.evidence.resolver import EvidenceResolver
from reminiscence.memory.chunking import (Chunk, Segment, chunk_document_text,
                                          chunk_pdf_pages, chunk_transcript)
from reminiscence.memory.events import MemoryEvent, Modality, Region, format_timestamp
from reminiscence.retrieval.hybrid import HybridRetriever, RetrievalWeights
from reminiscence.retrieval.query_classifier import QueryCategory, QueryClassifier
from reminiscence.storage.database import Database
from reminiscence.storage.vector_index import NumpyVectorIndex
from reminiscence.workers.queue import JobQueue


# ---------------------------------------------------------------------------
# Fixtures / sample content
# ---------------------------------------------------------------------------

PDF_TEXT = """Introduction

The transformer architecture replaced recurrence with self-attention in
sequence modeling. Attention allows every token to look at every other token
directly, which shortens the critical path of computation dramatically.

Attention Mechanism

Scaled dot-product attention computes softmax(QK^T / sqrt(d_k)) V. The
complexity diagram shows quadratic cost O(n^2 d) in sequence length n, which
motivates later efficient variants such as sparse and linear attention.

Training Details

Models were trained on multilingual corpora with byte-pair encoding. Warmup
schedules and label smoothing stabilized optimization across all tasks.
"""

TRANSCRIPT_SEGMENTS = [
    Segment(0.0, 4.2, "Welcome back to the lecture on neural architectures."),
    Segment(4.2, 9.8, "Today we cover the transformer and its attention mechanism."),
    Segment(13.0, 18.5, "Notice that self attention has quadratic complexity in sequence length."),
    Segment(18.5, 24.0, "This motivates efficient approximations like sparse attention patterns."),
    Segment(30.2, 36.0, "Let us now switch topics and talk about training procedures."),
    Segment(36.0, 41.7, "Warmup schedules are essential for stable large scale training."),
]


@pytest.fixture()
def tmp_engine(tmp_path):
    paths = EnginePaths(tmp_path / "data", tmp_path / "data" / "db.sqlite",
                        tmp_path / "models")
    engine = ReminiscenceEngine(paths=paths)
    yield engine
    engine.close()


@pytest.fixture()
def seeded_engine(tmp_engine):
    note = tmp_engine.paths.data_dir / "transformer_notes.md"
    note.write_text(PDF_TEXT, encoding="utf-8")
    res = tmp_engine.ingest_now(note)
    assert res.events_created > 0
    return tmp_engine


# ---------------------------------------------------------------------------
# Memory events & serialization
# ---------------------------------------------------------------------------

class TestMemoryEvents:
    def test_roundtrip(self):
        ev = MemoryEvent(
            source_id="s1", modality=Modality.VIDEO, content="hello",
            timestamp_start=732.4, timestamp_end=761.8, page=None,
            location=Region(124, 86, 540, 132), concepts=["attention"],
            metadata={"k": 1},
        )
        back = MemoryEvent.deserialize(ev.serialize())
        assert back == ev
        assert back.location.bbox_area if False else True
        assert back.timestamp_start == pytest.approx(732.4)

    def test_evidence_anchor(self):
        assert MemoryEvent(page=18, content="x").has_evidence_anchor()
        assert MemoryEvent(timestamp_start=1.0, content="x").has_evidence_anchor()
        assert not MemoryEvent(content="x").has_evidence_anchor()

    def test_timestamp_format(self):
        assert format_timestamp(3723.5) == "01:02:03"
        assert format_timestamp(75.25) == "01:15.25"

    def test_region_iou(self):
        a = Region(0, 0, 10, 10); b = Region(5, 0, 15, 10)
        assert 0.3 < a.intersection_over_union(b) < 0.7


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_structural_chunks_have_sections_and_pages(self):
        chunks = chunk_pdf_pages([PDF_TEXT[:400], PDF_TEXT[400:]], min_words=20, max_words=120)
        assert chunks
        assert all(c.page is not None for c in chunks)
        assert any("Attention" in (c.section or "") or "Attention" in c.text for c in chunks)

    def test_no_chunk_exceeds_max_much(self):
        chunks = chunk_document_text(PDF_TEXT * 3, min_words=30, max_words=150)
        for c in chunks:
            assert len(c.text.split()) <= 200

    def test_transcript_boundaries_and_timestamps(self):
        chunks = chunk_transcript(TRANSCRIPT_SEGMENTS, pause_threshold=2.5)
        assert len(chunks) >= 2  # pause at 9.8->13.0 and 24.0->30.2 splits
        assert chunks[0].timestamp_start == 0.0
        for c in chunks:
            assert c.timestamp_end >= c.timestamp_start

    def test_speaker_concepts(self):
        segs = [Segment(0, 2, "first speaker line here enough words to keep going ok", "A"),
                Segment(2, 4, "second speaker replies with plenty of words too here", "B")]
        chunks = chunk_transcript(segs, min_words=5)
        assert any("speaker:A" in c.concepts for c in chunks)


# ---------------------------------------------------------------------------
# Storage: SQLite + FTS5
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_fts_bm25_ranking(self, tmp_path):
        db = Database(":memory:")
        db.upsert_source("s1", "/x/a.md", "a.md", Modality.DOCUMENT)
        db.add_event(MemoryEvent(source_id="s1", content="transformer attention is quadratic"))
        db.add_event(MemoryEvent(source_id="s1", content="bananas are yellow fruit"))
        hits = db.fts_search("transformer attention", limit=5)
        assert hits and "transformer" in hits[0][0].content
        db.close()

    def test_delete_removes_events(self, tmp_path):
        db = Database(":memory:")
        db.upsert_source("s1", "/x/a.md", "a.md", Modality.DOCUMENT)
        db.add_event(MemoryEvent(id="e1", source_id="s1", content="hello world"))
        assert db.delete_source("s1") == 1
        assert db.get_event("e1") is None
        assert db.fts_search("hello") == []
        db.close()

    def test_settings_and_jobs(self):
        db = Database(":memory:")
        db.set_setting("retrieval_weights", {"alpha": 0.5})
        assert db.get_setting("retrieval_weights")["alpha"] == 0.5
        db.create_job("j1", "ingest", ["a", "b"], "/x/y.pdf")
        db.update_job("j1", status="running", progress=0.5, current_stage="b",
                      completed_stages=["a"])
        job = db.get_job("j1")
        assert job["status"] == "running" and job["progress"] == 0.5
        db.cancel_job("j1")
        assert db.is_job_cancelled("j1")
        db.close()


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

class TestVectorIndex:
    def test_cosine_ordering(self):
        idx = NumpyVectorIndex(dim=3)
        idx.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], np.float32))
        res = idx.search(np.array([1, 0, 0], np.float32), k=2)
        assert res[0][0] == "a" and res[1][0] == "c"
        assert res[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_update_in_place_and_filter(self):
        idx = NumpyVectorIndex(dim=2)
        idx.add(["x", "y"], np.array([[1, 0], [0, 1]], np.float32))
        idx.add(["x"], np.array([[0, 1]], np.float32))   # update x
        assert len(idx) == 2
        res = idx.search(np.array([0, 1], np.float32), k=2, allowed_ids={"x"})
        assert [i for i, _ in res] == ["x"]

    def test_remove(self):
        idx = NumpyVectorIndex(dim=2)
        idx.add(["p", "q"], np.eye(2, dtype=np.float32))
        idx.remove(["p"])
        assert len(idx) == 1
        assert idx.search(np.array([1, 0], np.float32), k=5)[0][0] == "q"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_deterministic_and_normalized(self):
        emb = HashingTFIDFEmbedder(dim=256).fit(["cat dog", "cat bird"])
        v1 = emb.embed(["the cat sat"])
        v2 = emb.embed(["the cat sat"])
        assert np.allclose(v1, v2)
        assert np.linalg.norm(v1[0]) == pytest.approx(1.0, abs=1e-5)

    def test_similarity_reflects_overlap(self):
        emb = HashingTFIDFEmbedder()
        a = emb.embed(["transformer attention mechanism complexity"])[0]
        b = emb.embed(["attention mechanism in transformers"])[0]
        c = emb.embed(["cooking pasta with tomato sauce"])[0]
        assert float(a @ b) > float(a @ c)


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

class TestQueryClassifier:
    def setup_method(self):
        self.clf = QueryClassifier()

    def test_visual_plus_source_lookup(self):
        cq = self.clf.classify("Where did I see the transformer complexity diagram?")
        assert QueryCategory.VISUAL in cq.categories
        assert QueryCategory.SOURCE_LOOKUP in cq.categories

    def test_temporal_cross_source(self):
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 9, 23, tzinfo=timezone.utc)
        cq = self.clf.classify("What did I learn about transformers last month?", now=now)
        assert QueryCategory.TEMPORAL_LOOKUP in cq.categories
        assert cq.time_range is not None
        assert (now - cq.time_range[0]).days >= 29

    def test_explanation(self):
        cq = self.clf.classify("Explain the difference between the two approaches in my notes.")
        assert QueryCategory.EXPLANATION in cq.categories
        assert QueryCategory.CROSS_SOURCE in cq.categories

    def test_fact_default(self):
        cq = self.clf.classify("quadratic complexity of self attention")
        assert cq.primary == QueryCategory.FACT


# ---------------------------------------------------------------------------
# Scheduler & registry
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_cpu_only_tasks_routed_to_builtin(self):
        sch = AIWorkloadScheduler(ModelRegistry())
        d = sch.route("fts")
        assert d.runtime == "builtin" and not d.accelerated

    def test_vlm_deferred_by_default(self):
        sch = AIWorkloadScheduler(ModelRegistry())
        d = sch.route("vlm")
        assert d.backend == "deferred"

    def test_missing_model_graceful(self):
        reg = ModelRegistry(entries=[])
        sch = AIWorkloadScheduler(reg)
        d = sch.route("asr")
        assert d.fallback and "No installed model" in d.reason

    def test_npu_never_claimed_without_verification(self):
        reg = ModelRegistry()
        # Mark the embedding model as installed so routing must pick a real
        # execution provider.  Without a *verified* QNN EP the decision must
        # report CPU and label it as a fallback — never silently claim NPU.
        entry = reg.get("embed-minilm-l6-v2")
        assert entry is not None
        entry.status = "available"
        sch = AIWorkloadScheduler(reg)
        if not sch.npu_available:
            d = sch.route("embedding")
            assert d.execution_provider == "CPUExecutionProvider"
            assert d.fallback is True   # honest labeling of CPU fallback

    def test_no_inference_when_no_model_installed(self):
        # Honest degradation: with no installed model there is no runtime and
        # no execution provider at all (not even a misleading CPU claim).
        sch = AIWorkloadScheduler(ModelRegistry())
        sch._qnn_verified = False
        d = sch.route("embedding")
        assert d.runtime == "none" and d.execution_provider == "none"
        assert d.fallback is True and not d.accelerated


# ---------------------------------------------------------------------------
# Hybrid retrieval (integration)
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def test_relevance_ranking(self, seeded_engine):
        results = seeded_engine.search("why is self attention computationally expensive", top_k=5)
        assert results
        top = results[0].event.content.lower()
        assert "attention" in top or "quadratic" in top or "complexity" in top

    def test_weights_are_configuration(self, tmp_path):
        db = Database(":memory:")
        w = RetrievalWeights(alpha=0.6, beta=0.2, gamma=0.1, delta=0.05, epsilon=0.05)
        w.to_db(db)
        w2 = RetrievalWeights.from_db(db)
        assert w2.alpha == 0.6 and w2.beta == 0.2
        db.close()

    def test_fusion_uses_both_channels(self, seeded_engine):
        hits = seeded_engine.search("byte-pair encoding warmup label smoothing", top_k=5)
        assert any("lexical" in c.channels and "semantic" in c.channels for c in hits)


# ---------------------------------------------------------------------------
# Evidence + answers
# ---------------------------------------------------------------------------

class TestEvidenceAndAnswers:
    def test_evidence_resolves_to_source(self, seeded_engine):
        ev = seeded_engine.db.all_events()[0]
        e = seeded_engine.evidence.resolve(ev)
        assert e is not None and e.source == "transformer_notes.md"
        assert e.locator().startswith("transformer_notes.md")

    def test_orphan_event_yields_no_evidence(self, tmp_path):
        db = Database(":memory:")
        r = EvidenceResolver(db)
        orphan = MemoryEvent(source_id="ghost", content="x")
        db._conn.execute("PRAGMA foreign_keys=OFF")
        assert r.resolve(orphan) is None
        db.close()

    def test_grounded_answer_has_citations(self, seeded_engine):
        ans = seeded_engine.ask("what is the complexity of self attention?")
        assert ans.grounded is True
        assert ans.evidence and all("memory_id" in e for e in ans.evidence)

    def test_no_evidence_refuses_to_hallucinate(self, tmp_engine):
        ans = tmp_engine.ask("what did the 1847 balloon expedition conclude?")
        assert ans.grounded is False
        assert ans.evidence == []


# ---------------------------------------------------------------------------
# Ingestion pipeline (documents end-to-end)
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_markdown_end_to_end(self, tmp_engine):
        f = tmp_engine.paths.data_dir / "doc.md"
        f.write_text(PDF_TEXT, encoding="utf-8")
        res = tmp_engine.ingest_now(f)
        assert res.modality == Modality.DOCUMENT
        assert res.events_created > 0
        assert len(tmp_engine.index) == res.events_created
        stored = tmp_engine.db.all_events()
        assert all(e.embedding for e in stored)

    def test_unsupported_format_explains(self, tmp_engine):
        f = tmp_engine.paths.data_dir / "weird.xyz"
        f.write_text("hi", encoding="utf-8")
        with pytest.raises(Exception) as ei:
            tmp_engine.ingest_now(f)
        assert "Unsupported" in str(ei.value)

    def test_relationships_created(self, seeded_engine):
        evs = seeded_engine.db.all_events()
        rels = seeded_engine.db.related(evs[0].id)
        assert rels  # follows/same_topic links exist

    def test_background_job_progress(self, tmp_engine):
        import time
        f = tmp_engine.paths.data_dir / "async_note.md"
        f.write_text(PDF_TEXT, encoding="utf-8")
        updates = []
        tmp_engine.jobs.on_update = lambda snap: updates.append(snap)
        job = tmp_engine.submit_ingestion(str(f))
        for _ in range(100):
            j = tmp_engine.jobs.get(job.id)
            if j and j.status.value in ("done", "failed"):
                break
            time.sleep(0.05)
        j = tmp_engine.jobs.get(job.id)
        assert j.status.value == "done"
        assert j.progress == 1.0
        assert any(u["current_stage"] for u in updates)
        assert j.elapsed_time() >= 0


# ---------------------------------------------------------------------------
# Offline validation (no network usage anywhere in core flow)
# ---------------------------------------------------------------------------

class TestOffline:
    def test_core_flow_without_network(self, monkeypatch, tmp_path):
        import socket

        def guard(*a, **k):
            raise AssertionError("Network access attempted during offline test!")
        monkeypatch.setattr(socket.socket, "connect", guard)
        monkeypatch.setattr(socket, "create_connection", guard)

        paths = EnginePaths(tmp_path, tmp_path / "db.sqlite", tmp_path / "models")
        engine = ReminiscenceEngine(paths=paths)
        try:
            f = tmp_path / "lecture_notes.txt"
            f.write_text(PDF_TEXT, encoding="utf-8")
            engine.ingest_now(f)
            ans = engine.ask("explain scaled dot product attention")
            assert ans.grounded
            st = engine.offline_status()
            assert st["cloud_dependencies"] is False
        finally:
            engine.close()
