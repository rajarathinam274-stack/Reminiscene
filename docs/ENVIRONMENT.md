# Environment Variables

All variables are optional and preserve offline-first behavior.

| Variable | Default | Purpose |
|---|---|---|
| REMINISCENCE_DATA_DIR | ~/.reminiscence | Local application data root |
| REMINISCENCE_DB_PATH | <data-dir>/reminiscence.db | SQLite database |
| REMINISCENCE_MODEL_DIR | <data-dir>/models | Verified model cache |
| REMINISCENCE_MEDIA_DIR | <data-dir>/media | Managed media storage |
| REMINISCENCE_LOG_LEVEL | INFO | Log level |
| REMINISCENCE_DEBUG_CONTENT | 0 | Developer-only raw-content logging |
| REMINISCENCE_TELEMETRY | 0 | Local telemetry flag |
| REMINISCENCE_PREFER_NPU | 1 | Prefer QNN after runtime verification |
| REMINISCENCE_ALLOW_VLM | 0 | Optional visual reasoning |
| REMINISCENCE_EMBEDDING_MODEL | empty | Local embedding model |
| REMINISCENCE_ASR_MODEL | empty | Local ASR model |
| REMINISCENCE_OCR_MODEL | empty | Local OCR model |
| REMINISCENCE_LLM_MODEL | empty | Local LLM |
| REMINISCENCE_WORKERS | 2 | Background worker count |
| REMINISCENCE_MAX_FILE_SIZE | 536870912 | Maximum input bytes |
| REMINISCENCE_VECTOR_INDEX | numpy | Vector backend |

The core engine requires no API key. Network model acquisition remains an
explicit model-management operation.
