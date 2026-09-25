"""Model manager: manifests, local cache, checksum verification, loading (P0).

Governs the lifecycle of every AI model artifact on disk.  The hard rule is:

    **An unverified model artifact is never activated.**

States::

    DISCOVERED        -> artifact file found in cache, not yet checked
    VERIFIED          -> sha256 + size match the manifest
    AVAILABLE         -> verified and loadable (runtime deps present)
    LOADED            -> runtime session created
    ACTIVE            -> currently serving inference
    MISSING           -> no artifact in cache (optional network download path)
    CHECKSUM_FAILED   -> artifact present but hash mismatch (never activated)
    UNSUPPORTED       -> manifest declares providers this machine lacks
    LOAD_FAILED       -> verification passed but runtime failed to load
    PROVIDER_UNAVAILABLE -> runtime present, requested provider absent

Network downloads are *opt-in* (``allow_network=True``); normal application
execution never silently fetches models.  All failures surface as explicit,
typed states so callers can degrade gracefully and report honestly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

CHUNK = 1024 * 1024


class ModelState(StrEnum):
    DISCOVERED = "discovered"
    VERIFIED = "verified"
    AVAILABLE = "available"
    LOADED = "loaded"
    ACTIVE = "active"
    MISSING = "missing"
    CHECKSUM_FAILED = "checksum_failed"
    UNSUPPORTED = "unsupported"
    LOAD_FAILED = "load_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


class ModelManagerError(Exception):
    """Base error for model manager failures."""


class ChecksumMismatchError(ModelManagerError):
    """Artifact does not match its manifest checksum — activation blocked."""


@dataclass(frozen=True)
class ModelManifest:
    """Declarative description of a model artifact and its requirements."""

    model_id: str
    version: str
    task: str  # e.g. "embeddings", "asr", "ocr", "llm"
    format: str  # "onnx"
    quantization: str  # "fp32" | "fp16" | "int8" | "q4" ...
    file: str  # filename relative to the model directory
    sha256: str  # expected content hash (hex, lowercase)
    size: int  # expected size in bytes
    minimum_ram_mb: int
    supported_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    license: str = "unknown"
    source: str = ""  # human-readable origin (URL or local path)
    dim: int | None = None  # embedding dimension when applicable
    model_path: Path | None = None  # resolved at discovery time

    @property
    def key(self) -> str:
        return f"{self.model_id}@{self.version}"

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "model_path"}
        return d

    @staticmethod
    def from_dict(d: dict, base_dir: Path | None = None) -> ModelManifest:
        payload = {k: v for k, v in d.items() if k in ModelManifest.__dataclass_fields__}
        m = ModelManifest(**payload)
        if base_dir is not None:
            object.__setattr__(m, "model_path", base_dir / m.file)
        return m


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


class ModelVerifier:
    """Checksum + size verification against a manifest."""

    @staticmethod
    def verify(manifest: ModelManifest) -> ModelState:
        path = manifest.model_path
        if path is None or not path.is_file():
            return ModelState.MISSING
        if path.stat().st_size != manifest.size:
            return ModelState.CHECKSUM_FAILED
        if sha256_file(path) != manifest.sha256.lower():
            return ModelState.CHECKSUM_FAILED
        return ModelState.VERIFIED


class ModelDownloader:
    """Opt-in network fetcher. Never invoked implicitly by the manager."""

    @staticmethod
    def download(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
        if not url.startswith(("http://", "https://")):
            raise ModelManagerError(f"refusing non-HTTP source URL: {url!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
        except Exception as exc:  # noqa: BLE001 - surfaced as typed failure
            tmp.unlink(missing_ok=True)
            raise ModelManagerError(f"download failed for {url}: {exc}") from exc
        return dest


class ModelCache:
    """Local model directory layout and manifest bookkeeping."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, manifest: ModelManifest) -> Path:
        candidate = (self.root / manifest.file).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise ModelManagerError("manifest file escapes model cache root")
        return candidate

    def save_manifest(self, manifest: ModelManifest) -> Path:
        p = self.root / f"{manifest.model_id}.manifest.json"
        p.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return p

    def load_manifests(self) -> list[ModelManifest]:
        out: list[ModelManifest] = []
        for p in sorted(self.root.glob("*.manifest.json")):
            try:
                out.append(
                    ModelManifest.from_dict(json.loads(p.read_text(encoding="utf-8")), self.root)
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return out


@dataclass
class ManagedModel:
    manifest: ModelManifest
    state: ModelState
    detail: str = ""
    runtime: object | None = None

    @property
    def activatable(self) -> bool:
        return self.state in (ModelState.AVAILABLE, ModelState.LOADED, ModelState.ACTIVE)


class ModelLoader:
    """Creates ONNX runtime sessions for *verified* models only."""

    @staticmethod
    def load(manifest: ModelManifest, preferred_providers: list[str] | None = None):
        from reminiscence.ai.runtime import OnnxRuntimeAdapter

        adapter = OnnxRuntimeAdapter(
            str(manifest.model_path),
            preferred_providers=preferred_providers or [p for p in manifest.supported_providers],
        )
        info = adapter.load()
        return adapter, info


class ModelManager:
    """Orchestrates discovery -> verification -> availability -> loading."""

    def __init__(self, cache_root: Path | str, *, allow_network: bool = False):
        self.cache = ModelCache(Path(cache_root))
        self.allow_network = allow_network
        self._models: dict[str, ManagedModel] = {}

    # -- registration ------------------------------------------------------
    def register(self, manifest: ModelManifest) -> ManagedModel:
        manifest = ModelManifest(
            **{
                **{k: v for k, v in manifest.__dict__.items()},
                "model_path": self.cache.artifact_path(manifest),
            }
        )
        self.cache.save_manifest(manifest)
        mm = ManagedModel(manifest=manifest, state=ModelState.DISCOVERED)
        mm.state = ModelVerifier.verify(manifest)
        mm.detail = {
            ModelState.MISSING: "artifact not found in local cache",
            ModelState.CHECKSUM_FAILED: "sha256/size mismatch — activation blocked",
        }.get(mm.state, "")
        self._models[manifest.key] = mm
        return mm

    def discover(self) -> list[ManagedModel]:
        """Reload all manifests from cache and re-verify artifacts."""
        for manifest in self.cache.load_manifests():
            self.register(manifest)
        return list(self._models.values())

    # -- status ------------------------------------------------------------
    def status(self, model_key: str) -> ManagedModel:
        if model_key not in self._models:
            raise ModelManagerError(f"unknown model: {model_key}")
        return self._models[model_key]

    def all_status(self) -> dict[str, ManagedModel]:
        return dict(self._models)

    # -- acquisition -------------------------------------------------------
    def ensure_available(self, model_key: str, available_providers: list[str]) -> ManagedModel:
        """Move a model to AVAILABLE, or an explicit failure state.

        Network download happens only when ``allow_network`` was set at
        construction AND the manifest carries an HTTP(S) source URL.
        """
        mm = self.status(model_key)
        if (
            mm.state == ModelState.MISSING
            and self.allow_network
            and mm.manifest.source.startswith("http")
        ):
            try:
                ModelDownloader.download(mm.manifest.source, mm.manifest.model_path)  # type: ignore[arg-type]
                mm.state = ModelVerifier.verify(mm.manifest)
            except ModelManagerError as exc:
                mm.state = ModelState.MISSING
                mm.detail = str(exc)
                return mm
        if mm.state == ModelState.CHECKSUM_FAILED:
            return mm  # never activate unverified artifacts
        if mm.state == ModelState.VERIFIED:
            unsupported = [
                p
                for p in mm.manifest.supported_providers
                if p not in available_providers and p != "CPUExecutionProvider"
            ]
            if "CPUExecutionProvider" not in available_providers and unsupported:
                mm.state = ModelState.UNSUPPORTED
                mm.detail = f"none of {mm.manifest.supported_providers} present"
            else:
                mm.state = ModelState.AVAILABLE
        return mm

    # -- loading -----------------------------------------------------------
    def load(self, model_key: str, preferred_providers: list[str] | None = None) -> ManagedModel:
        mm = self.status(model_key)
        if mm.state not in (ModelState.AVAILABLE, ModelState.LOADED):
            if mm.state == ModelState.LOADED:
                return mm
            raise ModelManagerError(
                f"cannot load {model_key}: state={mm.state.value} ({mm.detail or 'not verified'})"
            )
        try:
            adapter, info = ModelLoader.load(mm.manifest, preferred_providers)
        except Exception as exc:  # runtime import/load failure
            mm.state = (
                ModelState.PROVIDER_UNAVAILABLE
                if "provider" in str(exc).lower()
                else ModelState.LOAD_FAILED
            )
            mm.detail = str(exc)
            raise ModelManagerError(f"load failed for {model_key}: {exc}") from exc
        mm.runtime = adapter
        mm.state = ModelState.LOADED
        mm.detail = f"active provider: {info.execution_provider}"
        return mm

    def activate(self, model_key: str) -> ManagedModel:
        mm = self.status(model_key)
        if mm.state != ModelState.LOADED:
            raise ModelManagerError(f"cannot activate {model_key} from state {mm.state.value}")
        mm.state = ModelState.ACTIVE
        return mm

    def unload(self, model_key: str) -> None:
        mm = self.status(model_key)
        mm.runtime = None
        mm.state = (
            ModelState.AVAILABLE if mm.state in (ModelState.LOADED, ModelState.ACTIVE) else mm.state
        )
