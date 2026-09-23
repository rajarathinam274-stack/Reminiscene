# REMINISCENCE — Deployment Procedure (Step by Step)

**Target platform:** Windows on Snapdragon (HP PCs) · **Architecture:** local-first / offline-first
**Design baseline:** Product & UX Design Specification v1.0

> Reminder (spec §18): never fabricate benchmark or hardware numbers — only show measurements actually collected on the deployment machine.

---

## Step 0 — Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | ≥ 3.10 (verified on 3.12) |
| Git | any recent version |
| OS | Windows 11 on Snapdragon (production target); Linux/macOS for development |
| Disk | ≥ 2 GB free for app + `~/.reminiscence` data dir (SQLite DB, vector index, models) |
| Network | **Not required at runtime** — core workflow is fully local (no cloud dependencies, no telemetry) |

No database server, Docker, Node.js, or API keys are needed. The default embedder
is local (`local-hash-tfidf-512`), so a fresh install works with Wi-Fi off.

---

## Step 1 — Obtain the code

```powershell
git clone https://github.com/<your-org>/reminiscence.git
cd reminiscence
```

## Step 2 — Create an isolated environment

```powershell
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
```

## Step 3 — Install dependencies

Core engine requires only `numpy`. Optional extras:

```powershell
pip install numpy              # core (storage/vector index/benchmarks)
pip install PySide6            # desktop GUI (app/ui) — spec §6 application shell
pip install pytest             # only for running the test suite
```

For Snapdragon NPU acceleration (spec §2.4, §17), install the Qualcomm AI Stack /
QNN execution provider per HP/Snapdragon device documentation. Without it the app
still runs fully — the scheduler simply reports `npu_verified: false` and routes to CPU.

## Step 4 — Configure

All build/deployment tunables are `REMINISCENCE_*` environment variables, validated
at startup (fail-fast with actionable errors). **Full reference:
[`docs/ENVIRONMENT.md`](ENVIRONMENT.md).** A ready-to-copy template lives in
[`.env.example`](../.env.example) (`cp .env.example .env`, then load it into the
shell — see the doc for one-liners for bash and PowerShell).

Quick overview:

| Variable | Default | Purpose |
|---|---|---|
| `REMINISCENCE_DATA_DIR` | `%USERPROFILE%\.reminiscence` | storage root (DB, vector index, models); `--data-dir` CLI flag wins over it |
| `REMINISCENCE_LOG_LEVEL` | `INFO` | verbosity (`DEBUG`…`CRITICAL`) |
| `REMINISCENCE_DEBUG_CONTENT` | `false` | logs raw document content — **development only**, privacy-sensitive; keep unset/false in production (spec §20) |
| `REMINISCENCE_N_WORKERS` | `2` | ingestion worker threads (1–16) |
| `REMINISCENCE_MAX_FILE_MB` | `512` | max import file size |
| `REMINISCENCE_EMBEDDING_DIM` | `512` | vector dimensionality (changing it requires re-import) |
| `REMINISCENCE_EMBEDDING_MODEL_PATH` | unset | optional local embedding model |
| `REMINISCENCE_ALLOW_VLM` | `false` | allow vision-language workloads |
| `REMINISCENCE_PREFER_NPU_TASKS` | `embedding,asr,ocr,classifier` | NPU routing preferences (CPU/GPU fallback) |
| `REMINISCENCE_RETRIEVAL_{ALPHA,BETA,GAMMA,DELTA,EPSILON}` | `0.45/0.35/0.10/0.05/0.05` | hybrid retrieval weights (must sum to ≈1.0) |

Ensure the deploying user has read/write access to the data directory. There are no
secrets to manage locally; if cloud AI backends are ever enabled, inject keys via the
environment — never commit them (`.env` is git-ignored). Verify the resolved
configuration with `python -m reminiscence.app.main status` (Step 5).

## Step 5 — Verify installation

```powershell
python -m reminiscence.app.main status
```

Expected output (values vary per machine):

```json
{
  "offline_status": {
    "mode": "local-first",
    "cloud_dependencies": false,
    "telemetry_enabled": false,
    "npu_verified": false,
    "embedding_model": "local-hash-tfidf-512",
    "data_dir": "C:\\Users\\<user>\\.reminiscence"
  },
  "index_size": 0
}
```

✅ Success criteria: exit code 0, `mode: local-first`, data directory created.

## Step 6 — Run the test suite (pre-production gate)

```powershell
python -m pytest reminiscence/tests -q
```

All tests must pass before proceeding. Then capture a local performance baseline
(measured on THIS machine only — spec §18):

```powershell
python -m reminiscence.app.main benchmark
```

## Step 7 — Smoke-test the full memory loop (spec §27)

```powershell
echo "Grandma's apple pie recipe from 1974" > sample.txt
python -m reminiscence.app.main import sample.txt      # REMEMBER
python -m reminiscence.app.main search "recipe"        # SEARCH / RETRIEVE
python -m reminiscence.app.main ask "what recipes do I have?"   # ANSWER + evidence JSON
python -m reminiscence.app.main timeline --days 30     # chronological recall
python -m reminiscence.app.main sources                # list ingested sources
python -m reminiscence.app.main delete <source_id>     # verify explicit deletion (spec §19 Privacy)
```

✅ Success criteria: import reports memory events, `ask` returns `"grounded": true`
with evidence references, delete removes the source and its vectors.

## Step 8 — Launch the application

**Desktop GUI (production, spec §6):**
```powershell
python -m reminiscence.app.ui
```

**Headless / service use:** run the CLI commands above, or bind to
`ReminiscenceEngine` (`reminiscence/app/services/engine.py`) from your own host —
the UI and CLI share the same engine, ingestion jobs run on a background JobQueue.

## Step 9 — Offline validation (demo requirement, spec §31/§33)

1. Turn Wi-Fi / networking **OFF** on the device.
2. Repeat Step 7 (import → search → ask).
3. Confirm `status` still shows `cloud_dependencies: false` and results return normally.

This is the acceptance proof for the local-first principle (§2.1).

## Step 10 — Production hardening & operations

- **Backups:** schedule copies of the data directory (`reminiscence.db` + index + models); test restores quarterly.
- **Least privilege:** dedicated non-admin user; data directory not shared.
- **Logging:** keep `REMINISCENCE_DEBUG_CONTENT` unset; raw personal content must never reach logs (§20).
- **Performance panel:** expose only measured values from `benchmark_history` (§18).
- **Upgrades:**
  ```powershell
  git pull
  .venv\Scripts\activate
  python -m pytest reminiscence/tests -q
  python -m reminiscence.app.main status   # confirm migration OK, then relaunch
  ```
  Back up the data directory before every upgrade.

---

## Rollback

```powershell
git checkout <previous-tag>
# restore data-dir backup taken before upgrade
python -m pytest reminiscence/tests -q && python -m reminiscence.app.main status
```

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `No module named numpy` | venv not activated | activate `.venv` first |
| Empty search results after import | wrong `--data-dir` between commands | use the same data dir consistently |
| GUI does not start | PySide6 missing / no display | `pip install PySide6`; on headless hosts use the CLI |
| `npu_verified: false` on Snapdragon PC | QNN runtime not installed | install Qualcomm AI Stack; app remains functional on CPU (§2.4) |
| Corrupted index | interrupted write | restore data-dir backup; index auto-restores from DB embeddings on next start |
