# Reminiscence

> **Local-first multimodal personal AI memory engine** with an animated PySide6 desktop UI, hybrid memory search, evidence-grounded answers, and Snapdragon-aware AI acceleration.

Reminiscence is a privacy-first **personal memory assistant**, **multimodal memory search engine**, and local knowledge system. It turns photos, videos, audio, documents, notes, timestamps, and metadata into a unified memory index for searchable, contextual, and evidence-grounded recall.

The architecture is designed for **offline AI**, **on-device AI**, **Snapdragon Windows/ARM64**, ONNX Runtime/QNN provider routing, and efficient desktop interaction without requiring a cloud backend.

## Highlights

- 🧠 Multimodal personal memory engine for documents, images, audio, video, and notes
- 🔎 Hybrid semantic + lexical + temporal + graph-aware memory retrieval
- 🧩 Evidence-grounded answers with source references
- 🕸️ Entity and memory graph foundations for people, places, events, and concepts
- 🖥️ Native **PySide6 desktop UI** with lightweight animated transitions
- ⚡ Snapdragon-aware runtime status with explicit QNN/NPU vs CPU reporting
- 🔒 Local-first privacy model with no implicit cloud model downloads
- 💾 SQLite + FTS5 persistence with vector-index abstraction
- 🧵 Background ingestion jobs with retry/recovery foundations
- 📊 Local benchmark and provider telemetry
- 🧪 Headless engine remains independently testable from the GUI

## Animated desktop UI

The native frontend lives under `reminiscence/app/ui/` and uses PySide6 rather than a browser runtime.

Current UI foundation includes:

- animated memory workspace
- fast local search box
- recent-memory cards
- memory/source/provider metrics
- timeline/sources/graph/settings navigation shell
- Snapdragon/QNN capability indicator
- selectable evidence text
- low-cost opacity transitions designed to avoid heavy animation workloads

Install the UI extra and launch:

```bash
python -m pip install -e ".[ui]"
reminiscence-gui
```

The GUI binds directly to `ReminiscenceEngine`; retrieval, ingestion, storage, and AI routing are not duplicated in the presentation layer.

## Snapdragon and on-device AI

Snapdragon support is capability-driven, not branding-driven. The application reports the runtime provider actually available to the workload.

- QNN/NPU is shown only when the runtime verifies NPU availability.
- CPU fallback remains a supported path.
- Windows ARM64 is the target packaging path for QNN experimentation.
- Snapdragon performance claims require on-device measurement and profiling.
- The UI never labels CPU execution as NPU acceleration.

This makes the frontend suitable for Snapdragon-powered Windows PCs while preserving portability to ordinary x64 development machines.

## Architecture

    Personal files
         |
    Ingestion pipeline
      /  |  |  \
 Documents Audio Images Video
      |      |     |     |
    text    ASR   OCR keyframes
      \      |     |     /
       \___ Memory Events ___/
              |
       SQLite + FTS5 + vectors
              |
    lexical / semantic / temporal
              |
        graph retrieval
              |
       Evidence Resolver
              |
     grounded answer layer
              |
       PySide6 desktop UI

## Implemented foundation

- MemoryEvent with event/capture timestamps and pipeline versioning
- SHA-256 content identity and duplicate/move detection foundations
- SQLite persistence and schema migrations
- SQLite FTS5 lexical retrieval
- vector-index abstraction
- hybrid lexical/semantic/temporal/graph retrieval
- query intent and temporal parsing foundations
- people/place/entity-link storage foundations
- evidence resolution and grounded answer generation
- deterministic Memory Graph service
- durable SQLite-backed background jobs with retry/recovery
- model registry/runtime and model checksum verification
- ONNX/QNN provider detection and explicit fallback behavior
- document, audio, image, and video ingestion foundations
- benchmark telemetry with runtime/provider context
- offline-by-default model policy
- animated PySide6 desktop UI foundation

## Current limitations

The repository is still an implementation-stage project. The following remain incomplete or require target-device validation:

- production ASR inference and Snapdragon/QNN validation
- production OCR/vision inference
- full video semantic understanding
- production local LLM runtime and complete EvidencePack contract
- retrieval-quality evaluation dataset/metrics
- backup/export and secure deletion
- complete incremental indexing
- hardened security/CI gates
- release packaging and target-device validation

Authoritative backlog: `docs/MISSING_IMPLEMENTATION_PLAN.log`.

## Development

The project targets Python 3.11+.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

For the native desktop UI:

```bash
python -m pip install -e ".[ui]"
reminiscence-gui
```

The CLI exposes import, search, ask, timeline, source inspection, deletion, status, and benchmarking operations.

## Privacy model

Reminiscence is designed around local-first processing, no implicit network model downloads, explicit model verification, provider-aware inference reporting, no raw personal-content logging by default, and user-controlled deletion/export requirements.

Biometric processing and network-backed enrichment remain opt-in design requirements.

## Search/discovery keywords

Local AI memory, personal AI assistant, private AI, offline AI, on-device AI, multimodal memory, personal knowledge management, memory search, semantic search, hybrid retrieval, evidence-grounded AI, local LLM, PySide6 desktop app, Snapdragon AI, Snapdragon NPU, Qualcomm AI, ONNX Runtime, QNN, Windows on ARM, privacy-first AI, personal knowledge graph.

## License

See LICENSE.
