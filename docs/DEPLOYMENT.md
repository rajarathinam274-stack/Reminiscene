# Reminiscence Deployment

Reminiscence is local-first. The supported deployment model is a native Python
environment; Docker is provided for reproducible headless CLI jobs and CI, not
for the PySide6 desktop UI or Snapdragon NPU runtime.

## Native installation

Requirements: Python 3.11+, FFmpeg for audio/video, and optional local ONNX
model artifacts for ASR/OCR/embeddings.

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    cp .env.example .env
    reminiscence status

## Data layout

Keep personal data outside the source tree:

    ~/.reminiscence/
    ├── reminiscence.db
    ├── media/
    └── models/

Set REMINISCENCE_DATA_DIR, REMINISCENCE_DB_PATH, and REMINISCENCE_MODEL_DIR
when another location is required.

## Docker

Docker is headless only:

    docker compose build
    docker compose run --rm reminiscence status
    docker compose run --rm -v "$PWD/imports:/imports:ro" reminiscence import /imports

The container does not expose a network service. Mount only required data and
input directories.

## Models and Snapdragon

Models are not downloaded during application startup. Model artifacts must be
installed and verified separately. Snapdragon/QNN support is target-device
dependent and must be verified by the runtime before acceleration is reported.

The Docker image uses a standard Linux CPU runtime and must not be used to claim
Snapdragon NPU performance.

## Release checklist

- Review dependency versions before release.
- Keep .env and model secrets outside Git.
- Back up SQLite and media together.
- Verify model SHA-256 manifests.
- Run the full offline test suite.
- Validate actual execution-provider telemetry on target hardware.
