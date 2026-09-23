"""Environment-driven configuration for REMINISCENCE.

All tunables are read from ``REMINISCENCE_*`` environment variables so that
builds and deployments can be controlled without touching code or the CLI.
Precedence (highest wins): explicit constructor arguments > environment
variables > built-in defaults.

Every value is validated at startup; invalid values raise ``ConfigError``
with an actionable message instead of silently misbehaving.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "REMINISCENCE_"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


class ConfigError(RuntimeError):
    """Raised when an environment variable holds an invalid value."""


def _get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(PREFIX + name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    raise ConfigError(
        f"{PREFIX}{name}={raw!r} is not a boolean; use one of "
        "1/0, true/false, yes/no, on/off"
    )


def _get_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ConfigError(f"{PREFIX}{name}={raw!r} is not an integer") from None
    if val < minimum or (maximum is not None and val > maximum):
        bound = f">= {minimum}" + (f" and <= {maximum}" if maximum is not None else "")
        raise ConfigError(f"{PREFIX}{name}={val} must be {bound}")
    return val


def _get_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = _get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        raise ConfigError(f"{PREFIX}{name}={raw!r} is not a number") from None
    if not minimum <= val <= maximum:
        raise ConfigError(f"{PREFIX}{name}={val} must be between {minimum} and {maximum}")
    return val


def _get_path(name: str, default: Path) -> Path:
    raw = _get(name)
    if raw is None:
        return default
    return Path(raw).expanduser()


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all environment-controlled settings."""

    # --- Storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _get_path(
        "DATA_DIR", Path.home() / ".reminiscence"))

    # --- Logging / privacy ---------------------------------------------------
    log_level: str = field(default_factory=lambda: (
        (_get("LOG_LEVEL") or "INFO").upper()))
    debug_content: bool = field(default_factory=lambda: _get_bool(
        "DEBUG_CONTENT", False))

    # --- Ingestion / workers -----------------------------------------------
    n_workers: int = field(default_factory=lambda: _get_int(
        "N_WORKERS", 2, minimum=1, maximum=16))
    max_file_mb: int = field(default_factory=lambda: _get_int(
        "MAX_FILE_MB", 512, minimum=1))

    # --- Embeddings ----------------------------------------------------------
    embedding_dim: int = field(default_factory=lambda: _get_int(
        "EMBEDDING_DIM", 512, minimum=64, maximum=4096))
    embedding_model_path: str | None = field(default_factory=lambda: _get(
        "EMBEDDING_MODEL_PATH"))

    # --- AI scheduler (Snapdragon routing) ----------------------------------
    allow_vlm: bool = field(default_factory=lambda: _get_bool(
        "ALLOW_VLM", False))
    prefer_npu_tasks: tuple[str, ...] = field(default_factory=lambda: tuple(
        t.strip() for t in (_get("PREFER_NPU_TASKS") or
                            "embedding,asr,ocr,classifier").split(",") if t.strip()))

    # --- Hybrid retrieval weights (see retrieval.hybrid.RetrievalWeights) ---
    retrieval_alpha: float = field(default_factory=lambda: _get_float(
        "RETRIEVAL_ALPHA", 0.45))   # semantic
    retrieval_beta: float = field(default_factory=lambda: _get_float(
        "RETRIEVAL_BETA", 0.35))    # lexical (BM25)
    retrieval_gamma: float = field(default_factory=lambda: _get_float(
        "RETRIEVAL_GAMMA", 0.10))   # temporal
    retrieval_delta: float = field(default_factory=lambda: _get_float(
        "RETRIEVAL_DELTA", 0.05))   # modality relevance
    retrieval_epsilon: float = field(default_factory=lambda: _get_float(
        "RETRIEVAL_EPSILON", 0.05))  # source relevance

    def __post_init__(self) -> None:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(
                f"{PREFIX}LOG_LEVEL={self.log_level!r} invalid; use "
                "DEBUG, INFO, WARNING, ERROR or CRITICAL")
        total = (self.retrieval_alpha + self.retrieval_beta +
                 self.retrieval_gamma + self.retrieval_delta +
                 self.retrieval_epsilon)
        if abs(total - 1.0) > 0.01:
            raise ConfigError(
                "Retrieval weights must sum to ~1.0 (got "
                f"{total:.3f}). Adjust {PREFIX}RETRIEVAL_* variables.")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reminiscence.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def describe(self) -> dict:
        """JSON-safe summary for `status` output (no secrets, no content)."""
        return {
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "debug_content": self.debug_content,
            "n_workers": self.n_workers,
            "max_file_mb": self.max_file_mb,
            "embedding_dim": self.embedding_dim,
            "embedding_model_path": self.embedding_model_path,
            "allow_vlm": self.allow_vlm,
            "prefer_npu_tasks": list(self.prefer_npu_tasks),
            "retrieval_weights": {
                "alpha": self.retrieval_alpha, "beta": self.retrieval_beta,
                "gamma": self.retrieval_gamma, "delta": self.retrieval_delta,
                "epsilon": self.retrieval_epsilon,
            },
        }


_cached: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Return the process-wide Settings snapshot (env read once by default)."""
    global _cached
    if _cached is None or refresh:
        _cached = Settings()
    return _cached
