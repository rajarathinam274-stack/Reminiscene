"""AI runtime abstraction over ONNX Runtime (with QNN EP awareness).

Design rules:
- Never claim NPU acceleration merely because a model is Qualcomm-compatible.
- Always detect the *actual* execution provider in use and expose it.
- CPU fallback is allowed for normal operation but must be clearly labeled;
  benchmark mode flags it.
- If onnxruntime is not installed, everything degrades gracefully to a
  pure-CPU stub so the rest of the app/tests keep working offline.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeInfo:
    runtime: str  # "onnxruntime" | "stub"
    version: str | None
    available_providers: list[str] = field(default_factory=list)
    selected_provider: str = "CPUExecutionProvider"
    backend: str = "cpu"  # cpu | gpu | htp/npu
    arch: str = field(default_factory=lambda: platform.machine())
    os_name: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    @property
    def accelerated(self) -> bool:
        return self.selected_provider not in ("CPUExecutionProvider",) and self.backend != "cpu"


_PREFERRED_ORDER = [
    "QNNExecutionProvider",  # Qualcomm NPU (Hexagon HTP)
    "NnapiExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

_BACKEND_BY_PROVIDER = {
    "QNNExecutionProvider": "HTP",
    "NnapiExecutionProvider": "NNAPI/GPU",
    "CUDAExecutionProvider": "GPU",
    "CPUExecutionProvider": "cpu",
}


class OnnxRuntimeAdapter:
    """Lazy wrapper around an onnxruntime InferenceSession."""

    def __init__(self, model_path: str, preferred_providers: list[str] | None = None):
        self.model_path = model_path
        self.preferred = preferred_providers or _PREFERRED_ORDER
        self._session = None
        self._info: RuntimeInfo | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def available() -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def qnn_available() -> bool:
        """True only if the QNN EP is genuinely registered in this ORT build."""
        try:
            import onnxruntime as ort

            return "QNNExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    # ------------------------------------------------------------------
    def load(self) -> RuntimeInfo:
        if self._info is not None:
            return self._info
        info = RuntimeInfo(
            runtime="stub", version=None, backend="cpu", selected_provider="CPUExecutionProvider"
        )
        try:
            import onnxruntime as ort

            info.runtime = "onnxruntime"
            info.version = ort.__version__
            avail = list(ort.get_available_providers())
            info.available_providers = avail
            providers = [p for p in self.preferred if p in avail] or ["CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self.model_path, sess_options=sess_opts, providers=providers
            )
            # The session reports which providers were actually attached.
            active = list(self._session.get_providers())
            info.selected_provider = active[0] if active else "CPUExecutionProvider"
            info.backend = _BACKEND_BY_PROVIDER.get(info.selected_provider, "cpu")
        except ImportError:
            info.runtime = "stub"
        except Exception as e:  # model missing / corrupt -> explicit failure
            raise RuntimeError(f"Failed to load ONNX model {self.model_path}: {e}") from e
        self._info = info
        return info

    @property
    def info(self) -> RuntimeInfo:
        return self.load()

    # ------------------------------------------------------------------
    def run(self, inputs: dict[str, Any]) -> list[Any]:
        info = self.load()
        if self._session is None:
            raise RuntimeError(
                f"Model '{os.path.basename(self.model_path)}' unavailable: onnxruntime "
                "is not installed. Install ARM64 ONNX Runtime (and the QNN EP for NPU)."
            )
        return self._session.run(None, inputs)

    def input_names(self) -> list[str]:
        self.load()
        if self._session is None:
            return []
        return [i.name for i in self._session.get_inputs()]
