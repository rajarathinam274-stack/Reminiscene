"""Snapdragon benchmark subsystem.

Rules:
- Numbers come from measurements on the CURRENT machine only.
- Detect actual execution provider; flag CPU fallback in benchmark mode.
- Never present hosted/reference numbers as REMINISCENCE results.

This module is pure-Python (works anywhere); model-specific benches plug in
callables and report the runtime info they actually observe.
"""

from __future__ import annotations

import os
import platform
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..ai.runtime import OnnxRuntimeAdapter, RuntimeInfo
from ..storage.database import Database


@dataclass
class BenchmarkResult:
    run_ts: str
    machine: str
    model_id: str
    task: str
    runtime: str
    execution_provider: str
    backend: str
    input_desc: str
    latency_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    throughput: Optional[float] = None      # items/sec
    peak_memory_mb: Optional[float] = None
    cpu_pct: Optional[float] = None
    gpu_pct: Optional[float] = None
    npu_pct: Optional[float] = None         # None == not observable here
    fallback: bool = False
    notes: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["fallback"] = int(self.fallback)
        return d


def current_machine() -> str:
    return f"{platform.system()} {platform.release()} {platform.machine()} " \
           f"({os.cpu_count()} CPUs)"


def _percentile(vals: list[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


class BenchmarkRunner:
    """Runs cold/warm/batch/memory benchmarks against pluggable callables."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db
        self.results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    def _base(self, model_id: str, task: str, input_desc: str,
              rt: Optional[RuntimeInfo] = None) -> BenchmarkResult:
        res = BenchmarkResult(
            run_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            machine=current_machine(),
            model_id=model_id, task=task, input_desc=input_desc,
            runtime=rt.runtime if rt else "builtin",
            execution_provider=rt.selected_provider if rt else "CPUExecutionProvider",
            backend=rt.backend if rt else "cpu",
        )
        if rt is not None:
            # Honest fallback labeling: intended accel path vs actual EP.
            res.fallback = ("npu" in getattr(rt, "preferred_devices", ["npu"])) and not rt.accelerated \
                if hasattr(rt, "preferred_devices") else (rt.selected_provider == "CPUExecutionProvider"
                                                          and "QNNExecutionProvider" in rt.available_providers)
        return res

    # ------------------------------------------------------------------
    def bench_fn(
        self,
        fn: Callable[[], Any],
        model_id: str,
        task: str,
        input_desc: str,
        iterations: int = 5,
        warmup: int = 1,
        measure_memory: bool = True,
        rt_info: Optional[RuntimeInfo] = None,
        require_acceleration: bool = False,
    ) -> BenchmarkResult:
        """Cold start + warm latency + throughput + peak memory for one callable."""
        res = self._base(model_id, task, input_desc, rt_info)

        # Cold start
        t0 = time.perf_counter()
        out = fn()
        cold = (time.perf_counter() - t0) * 1000
        res.latency_ms = round(cold, 2)

        for _ in range(warmup):
            fn()

        latencies: list[float] = []
        if measure_memory:
            tracemalloc.start()
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - t0) * 1000)
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            res.peak_memory_mb = round(peak / 1e6, 2)

        res.p50_ms = round(_percentile(latencies, 50) or 0, 2)
        res.p95_ms = round(_percentile(latencies, 95) or 0, 2)
        mean = sum(latencies) / len(latencies)
        res.throughput = round(1000.0 / mean, 2) if mean > 0 else None

        # Utilization observability
        try:
            import psutil
            res.cpu_pct = psutil.cpu_percent(interval=None)
        except ImportError:
            res.cpu_pct = None
        res.gpu_pct = None
        res.npu_pct = None  # populated on-target where QNN profiling exists

        if require_acceleration and rt_info is not None and not rt_info.accelerated:
            res.notes = ("FLAG: intended accelerator path NOT used — measured on CPU. "
                         "Benchmark result is a CPU baseline, not an NPU result.")
            if res.execution_provider == "CPUExecutionProvider":
                res.fallback = True
            raise AssertionError(res.notes)

        self.results.append(res)
        if self.db:
            self.db.record_benchmark(res.to_row())
        return res

    # ------------------------------------------------------------------
    def bench_onnx_model(self, adapter: OnnxRuntimeAdapter, inputs: dict,
                         model_id: str, task: str, input_desc: str,
                         iterations: int = 5, require_acceleration: bool = False) -> BenchmarkResult:
        rt = adapter.load()
        return self.bench_fn(
            lambda: adapter.run(inputs),
            model_id=model_id, task=task, input_desc=input_desc,
            iterations=iterations, rt_info=rt,
            require_acceleration=require_acceleration,
        )

    # ------------------------------------------------------------------
    def report(self) -> str:
        lines = ["REMINISCENCE BENCHMARK REPORT", "=" * 40,
                 f"Machine: {current_machine()}"]
        for r in self.results:
            lines += [
                "",
                f"MODEL: {r.model_id}   TASK: {r.task}",
                f"RUNTIME: {r.runtime}   EP: {r.execution_provider}   BACKEND: {r.backend}",
                f"INPUT: {r.input_desc}",
                f"COLD LATENCY: {r.latency_ms} ms   P50: {r.p50_ms}   P95: {r.p95_ms}",
                f"THROUGHPUT: {r.throughput} it/s   PEAK MEM: {r.peak_memory_mb} MB",
                f"FALLBACK: {'YES' if r.fallback else 'No'}   NOTES: {r.notes or '-'}",
            ]
        return "\n".join(lines)
