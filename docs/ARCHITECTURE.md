# Reminiscence — Architecture (LLM-Free Memory Intelligence)

**Central principle:** Reminiscence retrieves and reasons over the user's
actual memories first, and generates language only as an optional
presentation layer. The source of truth is **MemoryEvents + Memory Graph +
EvidencePack**, never a generative model.

> **An LLM is NOT required for any core capability.**
> `REMINISCENCE_ENABLE_LLM=false` is the default and fully supported state.

---

## 1. Deterministic Pipeline

```text
                    USER
                     │
                     ▼
               ┌───────────┐
               │   Query   │
               └─────┬─────┘
                     │
                     ▼
        Query Understanding (deterministic rules;
        optional compact ONNX classifier — never an LLM)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Temporal    Entity     Intent
          │          │          │
          └──────────┼──────────┘
                     ▼
              Hybrid Retrieval
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    FTS5          Vector         Graph / Entities
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Evidence Resolver
                     │
              ┌──────┴──────┐
              │             │
          Conflicts     Confidence
              │             │
              └──────┬──────┘
                     ▼
            Deterministic Answer Engine
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
     Fact        Timeline      Extractive Summary
       │             │             │
       └─────────────┼─────────────┘
                     ▼
               Provenance Chain
                     │
                     ▼
              PySide6 UI / CLI

              Optional plugin:
              Local LLM (presentation layer only)
```

## 2. Ingestion

```text
Sources (PDF/DOCX/PPTX/TXT/MD/images/audio/video)
   ↓
Security Validation (size limits, magic bytes, path safety, temp isolation)
   ↓
Fingerprint (SHA-256 content identity → dedup / moved-file detection)
   ↓
Metadata + Temporal Extraction (EXIF/mtime/transcript; event vs capture vs import time)
   ↓
Extraction  ┌─ PDF text+layout   ┌─ OCR (regions + confidence)
            ├─ DOCX/PPTX         ├─ ASR (Whisper ONNX, timestamped segments)
            └─ plain/markdown    └─ Video keyframes (adaptive)
   ↓
Chunking (semantic/structural — headings, topics, pauses, slides)
   ↓
MemoryEvent (unified representation, provenance preserved)
   ↓
Entity extraction (Person/Place/Org/Date/Topic…)
   ↓
Embedding (MiniLM ONNX or deterministic hashing fallback — embedding ≠ LLM)
   ↓
SQLite + FTS5 + VectorIndex + Graph links
   ↓
Commit (jobs are durable, restart-recoverable, idempotent)
```

## 3. Module Map & Dependency Direction

```text
UI (app/ui, app/main.py CLI)
  ↓ only through
Services (app/services/engine.py facade)
  ↓
Domain (memory/, ingestion/, retrieval/, evidence/, graph hooks, benchmarks/)
  ↓
Infrastructure (storage/database.py, storage/vector_index.py,
                storage/media_store.py, ai/runtime.py, workers/queue.py)
```

Enforced by `reminiscence/tests/test_architecture.py`:
- domain/storage modules never import UI code;
- UI/CLI never touch `sqlite3` directly.

Key existing abstractions (do not duplicate):

| Abstraction | Location | Role |
|---|---|---|
| `MemoryEvent` | `memory/events.py` | unified multimodal record with temporal + reproducibility fields |
| `Embedder` (ABC) | `ai/embeddings/embedder.py` | replaceable embeddings (`HashingTFIDF`, `OnnxMiniLM`) |
| `VectorIndex` (ABC) | `storage/vector_index.py` | replaceable ANN (`NumpyVectorIndex` today) |
| `HybridRetriever` | `retrieval/hybrid.py` | α–ζ weighted fusion, configurable via DB settings |
| `QueryClassifier` | `retrieval/query_classifier.py` | rule-based intents + temporal parsing |
| `EvidenceResolver` | `evidence/resolver.py` | event → locatable evidence (page/ts/region) |
| `AnswerGenerator` (ABC) | `evidence/generator.py` | `ExtractiveGroundedAnswerer` (default), `OnnxLocalLLM` (optional) |
| `ModelRegistry`/`AIWorkloadScheduler` | `ai/registry.py`, `ai/scheduler.py` | honest provider routing, verified NPU claims only |
| `OnnxRuntimeAdapter` | `ai/runtime.py` | detects actual execution provider (QNN/CPU) |
| `JobQueue` | `workers/queue.py` | persistent jobs, retries, recovery, cancellation |
| `Database` | `storage/database.py` | SQLite + FTS5 + migrations (v5), entity/graph tables |
| `MediaStore` | `storage/media_store.py` | content-addressed media, atomic writes, safe deletion |

## 4. Retrieval Scoring (configurable, not "optimal")

```text
final_score = α·semantic + β·lexical + γ·temporal + δ·modality
            + ε·source   + ζ·graph
```

Weights live in the `settings` table (`RetrievalWeights.from_db/to_db`).
Default values are a starting baseline only; they must be tuned against the
evaluation dataset, never presented as optimal. Query-class-specific weight
profiles are planned (see MISSING_IMPLEMENTATION_PLAN.log §P1-Retrieval).

## 5. Temporal Semantics

Preferred hierarchy for "when did this happen":

```text
event_time_start → captured_at → created_at (indexing time only)
```

Every temporal field records `time_source` (exif/filename/transcript/import/
manual/inferred) and `time_confidence`. Temporal filtering uses
`COALESCE(event_time_start, captured_at, created_at)` — never raw `created_at`.

## 6. Evidence Rules

- No factual claim without ≥1 linked evidence item.
- Insufficient evidence ⇒ explicit refusal ("I don't have enough recorded
  evidence…"), never fabrication.
- Conflicting sources ⇒ surfaced, both shown; silent resolution forbidden.
- Provenance chain: answer → evidence → memory event → source → original file.

## 7. AI Runtime & Snapdragon/QNN

- Embeddings, ASR, OCR run through ONNX Runtime adapters that **detect the
  actual execution provider**. NPU acceleration is claimed only when the
  runtime reports `QNNExecutionProvider` in use.
- CPU fallback is always allowed for normal operation but is explicitly
  labeled and flagged in benchmark mode.
- Large VLM/LLM inference is deferred/optional by scheduler policy; heavy
  models are never silently routed to CPU.

## 8. Storage

- **SQLite**: sources, memory_events (+FTS5 external-content index),
  relationships, concepts, persons/places/events_entities, entity_links,
  jobs, models, settings, benchmark_runs, inference_records, media_assets.
- **Filesystem**: all binary media under `MediaStore`
  (`objects/<hash-prefix>/<hash>`, thumbnails, derived, temp). DB stores
  references only.
- **Vector index**: behind `VectorIndex` interface; NumPy exact search today,
  HNSW/FAISS pluggable later.

## 9. Privacy / Security Posture

Local-first: no telemetry, no cloud dependency, sanitized logs (no raw
content unless `REMINISCENCE_DEBUG_CONTENT=1`), secure deletion, checksummed
model artifacts, path-traversal protection, resource limits on ingest.

## 10. What Is NOT Required

- ❌ Generative LLM (core answers are extractive/deterministic)
- ❌ Cloud APIs, hosted vector DBs, remote OCR/ASR
- ❌ Network access at runtime (model download is an explicit opt-in step)
