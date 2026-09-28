# Reminiscence — Security Notes (Phase 0 audit snapshot)

Local-first guarantees:
- No cloud APIs, no telemetry, no mandatory network at runtime.
- Internet is only used by an explicit opt-in model download step.

Implemented controls:
- Path-traversal protection in MediaStore (canonical paths, prefix checks).
- Atomic writes + safe temp files; explicit delete API.
- SHA-256 content identity; checksum-verified model artifacts (unverified
  artifacts are never activated).
- File-size limits on ingestion; malformed-file handling per extractor.
- Sanitized structured logging; raw content logged only if
  REMINISCENCE_DEBUG_CONTENT=1.
- Prompt-injection guard (`ai/llm/prompt_guard.py`): imported documents are
  untrusted DATA, never instructions.
- Evidence rule: no fabricated citations; insufficient evidence -> refusal.
- CI: gitleaks secret scanning + pip-audit dependency scan (.github/workflows).

Open hardening work tracked in docs/MISSING_IMPLEMENTATION_PLAN.log §P5/P7:
secure-deletion verification, backup integrity, fuzzing of malformed media,
zip-bomb pixel limits, corrupted-DB recovery tests, target-hardware validation.
