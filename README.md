# Reminiscence

> Local-first multimodal personal memory engine for searchable, contextual, and evidence-grounded memories.

Reminiscence turns photos, videos, audio, documents, notes, timestamps, and metadata into a unified local memory index. It combines lexical, semantic, temporal, and graph-aware retrieval while keeping the default architecture offline-first.

## Current architecture

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
- durable SQLite-backed background jobs with retry/recovery
- model registry/runtime and model checksum verification
- ONNX/QNN provider detection and explicit fallback behavior
- document, audio, image, and video ingestion foundations
- benchmark telemetry with runtime/provider context
- offline-by-default model policy

## Current limitations

The repository is still an implementation-stage project. The following are not yet production-complete:

- production ASR inference and Snapdragon/QNN validation
- production OCR/vision inference
- full video semantic understanding
- complete Memory Graph service and entity resolution
- local LLM runtime and complete EvidencePack contract
- retrieval-quality evaluation dataset/metrics
- PySide6 desktop UI
- backup/export and secure deletion
- complete incremental indexing
- hardened security/CI gates
- release packaging and target-device validation

Authoritative backlog: docs/MISSING_IMPLEMENTATION_PLAN.log

## Development

The project targets Python 3.11+.

When the packaging configuration is present on your working branch:

    python -m venv .venv
    pip install -e ".[dev]"
    pytest

The CLI is intended to expose import, search, ask, timeline, source inspection, deletion, status, and benchmarking operations.

## Privacy model

Reminiscence is designed around local-first processing, no implicit network model downloads, explicit model verification, provider-aware inference reporting, no raw personal-content logging by default, and user-controlled deletion/export requirements.

Biometric processing and network-backed enrichment should remain opt-in.

## Snapdragon / NPU

Snapdragon acceleration is treated as a measurable runtime capability. The system must report the actual execution provider and must not claim NPU acceleration when a workload fell back to CPU.

## Repository hygiene

Generated Python bytecode, virtual environments, local databases, logs, build artifacts, and environment files are ignored by Git. Use .env.example for documented configuration values; never commit secrets.

## Project status

Next implementation sequence:

1. configuration and CI/security stabilization
2. production ASR/OCR
3. incremental ingestion and retrieval evaluation
4. Memory Graph service
5. local LLM + EvidencePack
6. backup/export + secure deletion
7. PySide6 desktop UI
8. Snapdragon/QNN validation
9. release engineering

## License

See LICENSE.
