"""AI Workload Scheduler — a distinct service that routes inference.

Responsibilities:
- pick the smallest suitable model for a task (via ModelRegistry)
- pick runtime/execution provider (NPU only when *verified* available)
- avoid unnecessary expensive inference (e.g. VLM only on demand)
- record every actual execution decision for auditability/benchmarking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .registry import ModelEntry, ModelRegistry
from .runtime import OnnxRuntimeAdapter, RuntimeInfo

log = logging.getLogger("reminiscence.scheduler")


@dataclass
class RoutingDecision:
    ts: str
    task: str
    model_id: Optional[str]
    runtime: str
    execution_provider: str
    backend: str
    accelerated: bool
    fallback: bool
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SchedulerConfig:
    # Explicit opt-in switches; defaults keep heavy paths off.
    allow_vlm: bool = False            # stage-2 visual reasoning only on demand
    prefer_npu_tasks: list[str] = field(
        default_factory=lambda: ["embedding", "asr", "ocr", "classifier"]
    )
    cpu_only_tasks: list[str] = field(
        default_factory=lambda: ["pdf_parse", "doc_parse", "fts", "vector_search"]
    )


class AIWorkloadScheduler:
    """Routes tasks to models/providers using capability metadata + measurement."""

    NON_MODEL_TASKS = {
        "pdf_parse": ("cpu", "PyMuPDF text/structure extraction"),
        "doc_parse": ("cpu", "python-docx / python-pptx parsing"),
        "fts": ("cpu", "SQLite FTS5 lexical search"),
        "vector_search": ("cpu", "embedded ANN (BLAS matmul)"),
    }

    def __init__(self, registry: ModelRegistry, config: Optional[SchedulerConfig] = None):
        self.registry = registry
        self.config = config or SchedulerConfig()
        self.decisions: list[RoutingDecision] = []
        self._qnn_verified: Optional[bool] = None

    # ------------------------------------------------------------------
    @property
    def npu_available(self) -> bool:
        """Verified — not assumed — NPU availability through QNN EP."""
        if self._qnn_verified is None:
            self._qnn_verified = OnnxRuntimeAdapter.qnn_available()
            log.info("QNN/NPU verified available: %s", self._qnn_verified)
        return self._qnn_verified

    def _record(self, d: RoutingDecision) -> RoutingDecision:
        self.decisions.append(d)
        log.info(
            "route task=%s model=%s ep=%s backend=%s fallback=%s reason=%s",
            d.task, d.model_id, d.execution_provider, d.backend, d.fallback, d.reason,
        )
        return d

    # ------------------------------------------------------------------
    def route(self, task: str, require_acceleration: bool = False) -> RoutingDecision:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        if task in self.NON_MODEL_TASKS:
            backend, reason = self.NON_MODEL_TASKS[task]
            return self._record(RoutingDecision(
                ts=now, task=task, model_id=None, runtime="builtin",
                execution_provider="CPUExecutionProvider", backend=backend,
                accelerated=False, fallback=False, reason=reason,
            ))

        if task == "vlm" and not self.config.allow_vlm:
            # Stage 1 policy: do NOT run a large VLM by default.
            return self._record(RoutingDecision(
                ts=now, task=task, model_id=None, runtime="none",
                execution_provider="none", backend="deferred",
                accelerated=False, fallback=False,
                reason="VLM disabled by tiered-vision policy; use OCR stage first",
            ))

        entry = self.registry.best_for(task)
        if entry is None:
            # No *installed* model, but we can still describe the intended
            # routing target from the catalogue (status stays honest).
            # There is no runtime and no execution provider at all — we never
            # claim even a CPU EP for inference that does not happen.
            planned = self._planned_candidate(task)
            d = RoutingDecision(
                ts=now, task=task, model_id=planned.id if planned else None,
                runtime="none", execution_provider="none", backend="none",
                accelerated=False, fallback=True,
                reason=f"No installed model for task '{task}'. "
                       "Offer setup path; feature degrades gracefully.",
            )
            if require_acceleration:
                raise RuntimeError(d.reason)
            return self._record(d)

        want_npu = self.npu_available and task in self.config.prefer_npu_tasks \
            and "npu" in entry.supported_devices
        ep = "QNNExecutionProvider" if want_npu else "CPUExecutionProvider"
        backend = "HTP" if want_npu else "cpu"
        fallback = (not want_npu) and ("npu" in entry.supported_devices)
        reason = ("NPU verified via QNN EP" if want_npu
                  else "CPU fallback: NPU not verified/not preferred for task")
        if require_acceleration and not want_npu:
            msg = f"Benchmark requires accelerated path for '{task}' but NPU unavailable."
            self._record(RoutingDecision(now, task, entry.id, entry.runtime, "CPUExecutionProvider",
                                         "cpu", False, True, msg))
            raise RuntimeError(msg)
        return self._record(RoutingDecision(
            ts=now, task=task, model_id=entry.id, runtime=entry.runtime,
            execution_provider=ep, backend=backend, accelerated=want_npu,
            fallback=fallback, reason=reason,
        ))

    # ------------------------------------------------------------------
    def _planned_candidate(self, task: str) -> Optional[ModelEntry]:
        """Catalogue entry for a task regardless of install status."""
        cands = sorted(self.registry.by_task(task), key=lambda e: e.memory_requirement_mb)
        return cands[0] if cands else None

    def select_model(self, task: str) -> Optional[ModelEntry]:
        return self.registry.best_for(task)

    def describe(self) -> list[dict]:
        return [d.to_dict() for d in self.decisions]
