"""Model registry: every model exposes full capability metadata.

Entries carry id/name/version/format/quantization/runtime/EP/supported
devices/memory/latency/license/status/fallback policy — as required by the
engineering spec.  The application resolves models through the registry so
swapping implementations never touches retrieval/business logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace


@dataclass
class ModelEntry:
    id: str
    name: str
    version: str
    modality: str  # text | audio | image | video | any
    task: str  # embedding | asr | ocr | vlm | llm | classifier
    model_path: str | None = None
    format: str = "onnx"
    quantization: str = "int8"
    runtime: str = "onnxruntime"
    execution_provider: str = "auto"  # auto -> resolved at load time
    supported_devices: list[str] = field(default_factory=lambda: ["cpu", "npu"])
    memory_requirement_mb: float = 0.0
    expected_latency_ms: float | None = None
    license: str = "unknown"
    status: str = "not_installed"  # available | not_installed | error
    fallback_policy: str = "cpu"  # cpu | fail

    def to_dict(self) -> dict:
        return asdict(self)


# Default catalogue.  These are *candidates* — actual measured latency and
# verified EP come from the benchmark subsystem on the target machine, never
# from copied Qualcomm AI Hub numbers.
DEFAULT_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="embed-minilm-l6-v2",
        name="all-MiniLM-L6-v2",
        version="onnx-int8-1",
        modality="text",
        task="embedding",
        quantization="int8",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=80,
        license="Apache-2.0",
        fallback_policy="cpu",
    ),
    ModelEntry(
        id="asr-whisper-base",
        name="Whisper-Base",
        version="onnx-q4-1",
        modality="audio",
        task="asr",
        quantization="int8/fp16",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=900,
        license="MIT",
        fallback_policy="cpu",
    ),
    ModelEntry(
        id="ocr-det-v5",
        name="PaddleOCR-Det-v5 (ONNX)",
        version="onnx-int8-1",
        modality="image",
        task="ocr",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=150,
        license="Apache-2.0",
        fallback_policy="cpu",
    ),
    ModelEntry(
        id="ocr-rec-v5",
        name="PaddleOCR-Rec-v5 (ONNX)",
        version="onnx-int8-1",
        modality="image",
        task="ocr",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=120,
        license="Apache-2.0",
        fallback_policy="cpu",
    ),
    ModelEntry(
        id="vlm-qwen25-vl-7b",
        name="Qwen2.5-VL-7B-Instruct",
        version="qnn-1",
        modality="image",
        task="vlm",
        format="qnn/onnx",
        quantization="int4",
        supported_devices=["npu", "cpu"],
        memory_requirement_mb=6000,
        license="apache-2.0",
        fallback_policy="fail",  # never silently run a 7B VLM on CPU for stage 1
    ),
    ModelEntry(
        id="llm-phi35-mini",
        name="Phi-3.5-mini (local)",
        version="q4-k1",
        modality="text",
        task="llm",
        format="gguf",
        quantization="Q4_K_M",
        runtime="onnxruntime / llama.cpp",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=3500,
        license="MIT",
        fallback_policy="cpu",
    ),
    ModelEntry(
        id="clf-query-tinybert",
        name="TinyBERT query classifier",
        version="onnx-int8-1",
        modality="text",
        task="classifier",
        supported_devices=["cpu", "npu"],
        memory_requirement_mb=60,
        license="MIT",
        fallback_policy="cpu",
    ),
]


class ModelRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None):
        self._entries: dict[str, ModelEntry] = {}
        # Always copy entries so per-instance status changes (install marking,
        # benchmarks, tests) never mutate the shared DEFAULT_MODELS catalogue.
        for e in DEFAULT_MODELS if entries is None else entries:
            self.register(replace(e))

    def register(self, entry: ModelEntry) -> None:
        self._entries[entry.id] = entry

    def get(self, model_id: str) -> ModelEntry | None:
        return self._entries.get(model_id)

    def by_task(self, task: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.task == task]

    def best_for(self, task: str, prefer_accelerated: bool = True) -> ModelEntry | None:
        """Select the smallest suitable *available* model for a task."""
        candidates = sorted(
            (e for e in self.by_task(task) if e.status == "available"),
            key=lambda e: e.memory_requirement_mb,
        )
        return candidates[0] if candidates else None

    def all(self) -> list[ModelEntry]:
        return list(self._entries.values())

    # -- persistence into SQLite `models` table --------------------------
    def sync_to_db(self, db) -> None:
        for e in self._entries.values():
            db._conn.execute(
                """INSERT OR REPLACE INTO models(id,name,version,modality,task,model_path,
                    format,quantization,runtime,execution_provider,supported_devices,
                    memory_requirement_mb,expected_latency_ms,license,status,fallback_policy)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.id,
                    e.name,
                    e.version,
                    e.modality,
                    e.task,
                    e.model_path,
                    e.format,
                    e.quantization,
                    e.runtime,
                    e.execution_provider,
                    json.dumps(e.supported_devices),
                    e.memory_requirement_mb,
                    e.expected_latency_ms,
                    e.license,
                    e.status,
                    e.fallback_policy,
                ),
            )
        db._conn.commit()

    @staticmethod
    def from_db(db) -> ModelRegistry:
        rows = db._conn.execute("SELECT * FROM models").fetchall()
        reg = ModelRegistry(entries=[])
        for r in rows:
            reg.register(
                ModelEntry(
                    id=r["id"],
                    name=r["name"],
                    version=r["version"] or "",
                    modality=r["modality"] or "any",
                    task=r["task"],
                    model_path=r["model_path"],
                    format=r["format"] or "onnx",
                    quantization=r["quantization"] or "fp32",
                    runtime=r["runtime"] or "onnxruntime",
                    execution_provider=r["execution_provider"] or "auto",
                    supported_devices=json.loads(r["supported_devices"] or "[]"),
                    memory_requirement_mb=r["memory_requirement_mb"] or 0.0,
                    expected_latency_ms=r["expected_latency_ms"],
                    license=r["license"] or "unknown",
                    status=r["status"] or "not_installed",
                    fallback_policy=r["fallback_policy"] or "cpu",
                )
            )
        return reg
