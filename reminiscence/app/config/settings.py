"""Environment-backed runtime configuration for Reminiscence."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default

def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    model_dir: Path
    media_dir: Path
    log_level: str
    debug_content: bool
    telemetry: bool
    prefer_npu: bool
    allow_vlm: bool
    workers: int
    max_file_size: int
    vector_index: str

    @classmethod
    def from_env(cls) -> "Settings":
        data = _path("REMINISCENCE_DATA_DIR", Path.home() / ".reminiscence")
        return cls(
            data_dir=data,
            db_path=_path("REMINISCENCE_DB_PATH", data / "reminiscence.db"),
            model_dir=_path("REMINISCENCE_MODEL_DIR", data / "models"),
            media_dir=_path("REMINISCENCE_MEDIA_DIR", data / "media"),
            log_level=os.getenv("REMINISCENCE_LOG_LEVEL", "INFO").upper(),
            debug_content=_bool("REMINISCENCE_DEBUG_CONTENT", False),
            telemetry=_bool("REMINISCENCE_TELEMETRY", False),
            prefer_npu=_bool("REMINISCENCE_PREFER_NPU", True),
            allow_vlm=_bool("REMINISCENCE_ALLOW_VLM", False),
            workers=max(1, _int("REMINISCENCE_WORKERS", 2)),
            max_file_size=max(1, _int("REMINISCENCE_MAX_FILE_SIZE", 512 * 1024 * 1024)),
            vector_index=os.getenv("REMINISCENCE_VECTOR_INDEX", "numpy"),
        )
