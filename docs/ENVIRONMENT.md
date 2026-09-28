# Reminiscence — Environment & Configuration

Reminiscence runs fully offline. Copy `.env.example` to `.env` (never committed).

## Variables

| Variable | Default | Purpose |
|---|---|---|
| REMINISCENCE_DATA_DIR | ./data | root for db/media/logs/cache |
| REMINISCENCE_DB_PATH | ./data/reminiscence.sqlite | SQLite database file |
| REMINISCENCE_MODEL_DIR | ./models | local model artifacts (verified via Model Manager) |
| REMINISCENCE_MEDIA_DIR | ./data/media | MediaStore content-addressed objects |
| REMINISCENCE_LOG_LEVEL | INFO | logging verbosity |
| REMINISCENCE_DEBUG_CONTENT | 0 | if 1, logs may include raw snippets (debug only) |
| REMINISCENCE_TELEMETRY | 0 | no telemetry exists; flag reserved and must stay 0 by default |
| REMINISCENCE_PREFER_NPU | 1 | prefer QNN EP when *detected*; never fabricates acceleration |
| REMINISCENCE_ALLOW_VLM | 0 | allow heavy VLM reasoning calls (opt-in) |
| REMINISCENCE_EMBEDDING_MODEL | all-MiniLM-L6-v2 | configured embedding model |
| REMINISCENCE_ASR_MODEL | whisper-base | ASR model id |
| REMINISCENCE_OCR_MODEL | (empty) | OCR backend selection |
| REMINISCENCE_LLM_MODEL | (empty) | OPTIONAL local LLM; empty = disabled |
| REMINISCENCE_ENABLE_LLM | false | master switch; core works with false |
| REMINISCENCE_WORKERS | 4 | background job worker count |
| REMINISCENCE_MAX_FILE_SIZE | 536870912 | ingest size limit (512 MiB) |
| REMINISCENCE_VECTOR_INDEX | numpy | vector index implementation (interface-pluggable) |

## Install

```bash
pip install -e .            # core (CPU-only, offline-capable)
pip install -e ".[documents,media,vision]"   # optional extractors
pip install -e ".[ai]"      # ONNX Runtime + tokenizers
pip install -e ".[npu]"     # QNN execution provider (Snapdragon targets)
pip install -e ".[ui]"      # PySide6 desktop shell
pip install -e ".[dev]"     # pytest/ruff/mypy/pre-commit
```

Models are NEVER downloaded silently. Use the Model Manager opt-in path
(`reminiscence/ai/model_manager.py`) with manifests + SHA-256 verification.

## Offline validation procedure

1. Import a representative library; build indexes.
2. Disable Wi-Fi.
3. Run search / ask / timeline queries.
4. Verify answers carry evidence; verify `status` shows offline OK.
5. Record benchmark results on the actual machine.
