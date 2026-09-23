# Environment Variables

REMINISCENCE is controlled through `REMINISCENCE_*` environment variables so
builds and deployments can be tuned **without touching code**. All variables
are parsed and validated once at startup in
[`reminiscence/app/config/settings.py`](../reminiscence/app/config/settings.py)
(`Settings` / `get_settings()`).

## Precedence

```
explicit CLI flag / constructor argument   (highest)
        ↓
REMINISCENCE_* environment variable
        ↓
built-in default                            (lowest)
```

Example: `--data-dir` wins over `REMINISCENCE_DATA_DIR`, which wins over
`~/.reminiscence`.

## Validation & error messages

Invalid values **fail fast** — the CLI exits with code `2` and prints:

```
configuration error: <message>
Fix or unset the offending REMINISCENCE_* variable; see docs/ENVIRONMENT.md for valid values.
```

Every error message names the exact variable, e.g.:

| Message | Cause |
|---|---|
| `REMINISCENCE_N_WORKERS='abc' is not an integer` | non-numeric int value |
| `REMINISCENCE_EMBEDDING_DIM=32 must be >= 64 and <= 4096` | out of range |
| `REMINISCENCE_ALLOW_VLM='maybe' is not a boolean; use one of 1/0, true/false, yes/no, on/off` | bad bool |
| `REMINISCENCE_LOG_LEVEL='verbose' invalid; use DEBUG, INFO, WARNING, ERROR or CRITICAL` | bad log level |
| `Retrieval weights must sum to ~1.0 (got 1.500). Adjust REMINISCENCE_RETRIEVAL_* variables.` | inconsistent α β γ δ ε |

Inspect the resolved configuration at any time:

```bash
python -m reminiscence.app.main status   # includes a "settings" block
```

## Variable reference

### Storage

| Variable | Type | Default | Range / values | Purpose |
|---|---|---|---|---|
| `REMINISCENCE_DATA_DIR` | path | `~/.reminiscence` (`%USERPROFILE%\.reminiscence` on Windows) | any writable path (`~` expanded) | Root for database (`reminiscence.db`), vector index and downloaded models. Overridden by the `--data-dir` CLI flag. |

### Logging & privacy

| Variable | Type | Default | Range / values | Purpose |
|---|---|---|---|---|
| `REMINISCENCE_LOG_LEVEL` | string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive) | Structured local logging verbosity. The `--verbose` CLI flag forces `DEBUG`. |
| `REMINISCENCE_DEBUG_CONTENT` | bool | `false` | `1/0`, `true/false`, `yes/no`, `on/off` | Logs **raw document content**. Development only — privacy-sensitive; keep unset in production (design spec §20). |

### Ingestion & workers

| Variable | Type | Default | Range / values | Purpose |
|---|---|---|---|---|
| `REMINISCENCE_N_WORKERS` | int | `2` | 1 – 16 | Background ingestion worker threads (JobQueue). Raise on high-core Snapdragon devices; lower to reduce memory pressure. |
| `REMINISCENCE_MAX_FILE_MB` | int | `512` | ≥ 1 | Maximum accepted size per imported file. Larger files are rejected with an actionable error (§22). |

### Embeddings

| Variable | Type | Default | Range / values | Purpose |
|---|---|---|---|---|
| `REMINISCENCE_EMBEDDING_DIM` | int | `512` | 64 – 4096 | Embedding vector dimensionality. ⚠️ Changing it invalidates existing indexes — re-import after changing. |
| `REMINISCENCE_EMBEDDING_MODEL_PATH` | path | unset (local hash-TF-IDF model) | path to a `.bin`/model directory | Optional local embedding model override. Never downloads from the network (local-first, §2.1). |

### AI scheduler — Snapdragon routing

| Variable | Type | Default | Range / values | Purpose |
|---|---|---|---|---|
| `REMINISCENCE_ALLOW_VLM` | bool | `false` | truthy/falsy set above | Permit vision-language-model workloads (advanced users only, §2.5). |
| `REMINISCENCE_PREFER_NPU_TASKS` | csv | `embedding,asr,ocr,classifier` | comma-separated task names | Tasks routed to the NPU when Qualcomm QNN/QAI runtimes are present; falls back to CPU/GPU transparently (§2.4). |

### Hybrid retrieval weights (must sum to ≈ 1.0)

Score = α·semantic + β·lexical + γ·temporal + δ·modality + ε·source

| Variable | Component | Default | Range |
|---|---|---|---|
| `REMINISCENCE_RETRIEVAL_ALPHA` | semantic (vector similarity) | `0.45` | 0.0 – 1.0 |
| `REMINISCENCE_RETRIEVAL_BETA` | lexical (BM25) | `0.35` | 0.0 – 1.0 |
| `REMINISCENCE_RETRIEVAL_GAMMA` | temporal recency | `0.10` | 0.0 – 1.0 |
| `REMINISCENCE_RETRIEVAL_DELTA` | modality relevance | `0.05` | 0.0 – 1.0 |
| `REMINISCENCE_RETRIEVAL_EPSILON` | source relevance | `0.05` | 0.0 – 1.0 |

Tuning examples (design priority §34 — search quality first):

```bash
# More keyword-exact matching (technical jargon, names, codes):
export REMINISCENCE_RETRIEVAL_ALPHA=0.30
export REMINISCENCE_RETRIEVAL_BETA=0.50
# (γ δ ε unchanged → total stays 1.0)

# Recall-oriented browsing (favor recency):
export REMINISCENCE_RETRIEVAL_ALPHA=0.40 REMINISCENCE_RETRIEVAL_BETA=0.25 \
       REMINISCENCE_RETRIEVAL_GAMMA=0.25
```

## Using a `.env` file

The engine reads the process environment directly; it does **not** parse
`.env` files itself. Load [`.env.example`](../.env.example) with your shell or
runtime, e.g.:

```bash
# bash / zsh
set -a && . ./.env && set +a
python -m reminiscence.app.main status

# Windows PowerShell
Get-Content .env | ForEach-Object { if ($_ -match '^\s*([^#][^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2]) } }
```

Copy the template and edit:

```bash
cp .env.example .env      # .env is git-ignored; never commit real config
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `configuration error: ... exit code 2` at startup | One env var holds an invalid value — the message names it; correct or `unset` it. |
| Searches return nothing after changing `EMBEDDING_DIM` | Existing vectors were built with the old dimension; delete and re-import sources. |
| Files rejected as too large | Raise `REMINISCENCE_MAX_FILE_MB` (check disk budget first). |
| High RAM during bulk import | Lower `REMINISCENCE_N_WORKERS` to `1`. |
| Raw text visible in logs | Ensure `REMINISCENCE_DEBUG_CONTENT` is unset in production. |
