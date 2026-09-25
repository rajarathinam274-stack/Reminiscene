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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    latency_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    throughput: float | None = None  # items/sec
    peak_memory_mb: float | None = None
    cpu_pct: float | None = None
    gpu_pct: float | None = None
    npu_pct: float | None = None  # None == not observable here
    fallback: bool = False
    notes: str = ""
    # Phase 12/13 traceability (every number tied to its exact environment)
    model_version: str | None = None
    quantization: str | None = None
    input_size: str | None = None
    iterations: int | None = None
    os_name: str | None = None
    arch: str | None = None
    cpu_model: str | None = None
    ram_gb: float | None = None
    runtime_version: str | None = None
    available_providers: str | None = None
    provider_used: str | None = None
    accelerated: bool | None = None
    fallback_reason: str | None = None
    power_mw: float | None = None
    thermal_celsius: float | None = None
    battery_pct: float | None = None
    pipeline_version: str | None = None

    def to_row(self) -> dict:
        d = asdict(self)
        d["fallback"] = int(self.fallback)
        if self.accelerated is not None:
            d["accelerated"] = int(self.accelerated)
        return d


def current_machine() -> str:
    return f"{platform.system()} {platform.release()} {platform.machine()} ({os.cpu_count()} CPUs)"


def _cpu_model() -> str | None:
    try:
        if platform.system() == "Windows":
            import subprocess

            out = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in out.stdout.splitlines() if l.strip()][1:]
            return lines[0] if lines else None
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                low = line.lower()
                if low.startswith(("model name", "hardware", "cpu architecture")):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            psz = os.sysconf("SC_PAGE_SIZE")
            return round(pages * psz / 1e9, 1)
        except Exception:
            return None


def hardware_context() -> dict:
    """Machine facts recorded with every benchmark so results are traceable."""
    return {
        "os_name": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "ram_gb": _ram_gb(),
    }


def power_thermal_snapshot() -> dict:
    """Best-effort power/thermal/battery capture.

    Returns None-valued fields wherever the metric is genuinely not
    observable on this machine — we never estimate or fabricate them.
    """
    snap = {"power_mw": None, "thermal_celsius": None, "battery_pct": None}
    # Battery (psutil exposes it everywhere it can be read locally)
    try:
        import psutil

        b = psutil.sensors_battery()
        if b is not None:
            snap["battery_pct"] = float(b.percent)
    except Exception:
        pass
    # Thermal zones (Linux; Snapdragon target exposes QCRIT/thermal sysfs)
    try:
        best = None
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                t = int(zone.read_text().strip()) / 1000.0
                best = t if best is None else max(best, t)
            except Exception:
                continue
        snap["thermal_celsius"] = round(best, 1) if best is not None else None
    except Exception:
        pass
    # Power (Intel RAPL via psutil where present; HTP power only on-target)
    try:
        import psutil

        for k, v in (getattr(psutil, "sensors_power", lambda: {})() or {}).items():
            if isinstance(v, (int, float)) and v > 0:
                snap["power_mw"] = float(v)
                break
    except Exception:
        pass
    return snap


class _CpuSampler:
    """Background CPU utilization sampler covering the measured window."""

    def __init__(self):
        self._stop = None
        self._thread = None
        self._samples: list[float] = []

    def start(self):
        try:
            import psutil
        except ImportError:
            return
        import threading

        self._samples = []
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                self._samples.append(psutil.cpu_percent(interval=None))
                self._stop.wait(0.25)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._stop is None:
            return None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        vals = [v for v in self._samples if v >= 0]
        return round(sum(vals) / len(vals), 1) if vals else None


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


class BenchmarkRunner:
    """Runs cold/warm/batch/memory benchmarks against pluggable callables."""

    def __init__(self, db: Database | None = None):
        self.db = db
        self.results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    def _base(
        self, model_id: str, task: str, input_desc: str, rt: RuntimeInfo | None = None
    ) -> BenchmarkResult:
        res = BenchmarkResult(
            run_ts=datetime.now(UTC).isoformat(timespec="seconds"),
            machine=current_machine(),
            model_id=model_id,
            task=task,
            input_desc=input_desc,
            runtime=rt.runtime if rt else "builtin",
            execution_provider=rt.selected_provider if rt else "CPUExecutionProvider",
            backend=rt.backend if rt else "cpu",
        )
        if rt is not None:
            # Honest fallback labeling: intended accel path vs actual EP.
            res.fallback = (
                ("npu" in getattr(rt, "preferred_devices", ["npu"])) and not rt.accelerated
                if hasattr(rt, "preferred_devices")
                else (
                    rt.selected_provider == "CPUExecutionProvider"
                    and "QNNExecutionProvider" in rt.available_providers
                )
            )
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
        rt_info: RuntimeInfo | None = None,
        require_acceleration: bool = False,
        model_version: str | None = None,
        quantization: str | None = None,
        input_size: str | None = None,
        fallback_reason: str | None = None,
        pipeline_version: str | None = None,
    ) -> BenchmarkResult:
        """Cold start + warm latency + throughput + peak memory for one callable."""
        res = self._base(model_id, task, input_desc, rt_info)
        res.model_version = model_version
        res.quantization = quantization
        res.input_size = input_size or f"{iterations}x '{input_desc}'"
        res.iterations = iterations
        if pipeline_version is None:
            # stamp automatically so every measured result is reproducible
            from ..ingestion.pipeline import PIPELINE_VERSION

            pipeline_version = PIPELINE_VERSION
        res.pipeline_version = pipeline_version
        ctx = hardware_context()
        res.os_name = ctx["os_name"]
        res.arch = ctx["arch"]
        res.cpu_model = ctx["cpu_model"]
        res.ram_gb = ctx["ram_gb"]

        # Cold start
        t0 = time.perf_counter()
        out = fn()
        cold = (time.perf_counter() - t0) * 1000
        res.latency_ms = round(cold, 2)

        for _ in range(warmup):
            fn()

        latencies: list[float] = []
        sampler = _CpuSampler()
        sampler.start()
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
        cpu_avg = sampler.stop()

        res.p50_ms = round(_percentile(latencies, 50) or 0, 2)
        res.p95_ms = round(_percentile(latencies, 95) or 0, 2)
        mean = sum(latencies) / len(latencies)
        res.throughput = round(1000.0 / mean, 2) if mean > 0 else None

        # Utilization observability — measured window average when available
        res.cpu_pct = cpu_avg
        if res.cpu_pct is None:
            try:
                import psutil

                res.cpu_pct = psutil.cpu_percent(interval=None)
            except ImportError:
                res.cpu_pct = None
        snap = power_thermal_snapshot()
        res.power_mw = snap["power_mw"]
        res.thermal_celsius = snap["thermal_celsius"]
        res.battery_pct = snap["battery_pct"]
        res.gpu_pct = None
        res.npu_pct = None  # populated on-target where QNN profiling exists

        if require_acceleration and rt_info is not None and not rt_info.accelerated:
            res.notes = (
                "FLAG: intended accelerator path NOT used — measured on CPU. "
                "Benchmark result is a CPU baseline, not an NPU result."
            )
            if res.execution_provider == "CPUExecutionProvider":
                res.fallback = True
                res.fallback_reason = fallback_reason or "QNN EP not verified on this machine"
            raise AssertionError(res.notes)

        self.results.append(res)
        if self.db:
            self.db.record_benchmark(res.to_row())
        return res

    # ------------------------------------------------------------------
    def bench_onnx_model(
        self,
        adapter: OnnxRuntimeAdapter,
        inputs: dict,
        model_id: str,
        task: str,
        input_desc: str,
        iterations: int = 5,
        require_acceleration: bool = False,
    ) -> BenchmarkResult:
        rt = adapter.load()
        return self.bench_fn(
            lambda: adapter.run(inputs),
            model_id=model_id,
            task=task,
            input_desc=input_desc,
            iterations=iterations,
            rt_info=rt,
            require_acceleration=require_acceleration,
        )

    # ------------------------------------------------------------------
    def report(self) -> str:
        lines = ["REMINISCENCE BENCHMARK REPORT", "=" * 40, f"Machine: {current_machine()}"]
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
