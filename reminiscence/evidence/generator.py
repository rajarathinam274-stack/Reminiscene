"""Local answer generation with strict grounding.

Pipeline position: Evidence Pack -> Local LLM -> Structured Answer.

Two backends behind one interface:
- OnnxLocalLLM: real local model (requires artifacts; EP verified at load).
- ExtractiveGroundedAnswerer: deterministic extractive synthesizer used when
  no LLM is installed.  It only re-states retrieved evidence and never
  invents citations.

Grounding policy: if retrieval yields no anchored evidence, return
``grounded=False`` with an "insufficient evidence" answer — do not fabricate.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .resolver import Evidence, EvidenceResolver


@dataclass
class Answer:
    answer: str
    grounded: bool
    evidence: list[dict] = field(default_factory=list)
    backend: str = ""            # which generator actually ran
    notes: str = ""
    confidence: float = 0.0      # 0..1 — how well the answer is supported
    support: str = "unavailable"  # directly_supported | inferred | uncertain | unavailable
    model_id: str = ""
    execution_provider: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"answer": self.answer, "grounded": self.grounded,
             "evidence": self.evidence, "backend": self.backend, "notes": self.notes,
             "confidence": round(self.confidence, 3), "support": self.support,
             "model_id": self.model_id,
             "execution_provider": self.execution_provider},
            ensure_ascii=False, indent=2,
        )


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, evidences: list[Evidence]) -> Answer: ...


# ---------------------------------------------------------------------------
class ExtractiveGroundedAnswerer(AnswerGenerator):
    """Deterministic, honest fallback synthesizer.

    Produces a structured summary that quotes retrieved evidence snippets and
    attaches their source references.  No claims beyond the evidence.
    """

    backend_name = "extractive-grounded-v1"

    def __init__(self, max_items: int = 4, snippet_chars: int = 200):
        self.max_items = max_items
        self.snippet_chars = snippet_chars

    def generate(self, question: str, evidences: list[Evidence]) -> Answer:
        usable = [e for e in evidences if e.page is not None or e.timestamp is not None
                  or e.section or e.bbox or e.source]
        if not usable:
            return Answer(
                answer="I could not find sufficiently grounded memories for that.",
                grounded=False,
                evidence=[],
                backend=self.backend_name,
                notes="No anchored evidence retrieved; refusing to fabricate citations.",
            )
        lines = [f"Based on {len(usable[:self.max_items])} local memor"
                 f"{'y' if len(usable[:self.max_items]) == 1 else 'ies'}:"]
        payload: list[dict] = []
        for i, e in enumerate(usable[: self.max_items], start=1):
            snip = e.snippet[: self.snippet_chars]
            lines.append(f"{i}. “{snip}” — {e.locator()}")
            payload.append({"memory_id": e.memory_id, **e.to_dict()})
        return Answer(
            answer="\n".join(lines),
            grounded=True,
            evidence=payload,
            backend=self.backend_name,
            notes="Extractive synthesis; install a local LLM for generative answers.",
        )


# ---------------------------------------------------------------------------
class OnnxLocalLLM(AnswerGenerator):
    """Local LLM over ONNX Runtime with verified execution provider.

    The prompt receives ONLY the retrieved evidence pack, never the whole
    database.  Output must be JSON conforming to the structured-answer schema.
    """

    backend_name = "local-onnx-llm"

    SYSTEM = (
        "You are REMINISCENCE, a private local memory assistant. Answer ONLY "
        "from the provided evidence. Cite evidence by its index like [E1]. "
        "If the evidence is insufficient, say so; never invent sources. "
        "Respond in JSON: {\"answer\": str, \"grounded\": bool}."
    )

    def __init__(self, adapter, tokenizer, max_new_tokens: int = 256):
        self._adapter = adapter      # OnnxRuntimeAdapter (already resolvable)
        self._tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @property
    def runtime_info(self):
        return self._adapter.info

    def build_prompt(self, question: str, evidences: list[Evidence]) -> str:
        ev_block = "\n".join(
            f"[E{i}] {e.locator()}: {e.snippet}" for i, e in enumerate(evidences, start=1)
        )
        return f"{self.SYSTEM}\n\nEvidence:\n{ev_block}\n\nQuestion: {question}\nAnswer:"

    def generate(self, question: str, evidences: list[Evidence]) -> Answer:
        if not evidences:
            return ExtractiveGroundedAnswerer().generate(question, evidences)
        prompt = self.build_prompt(question, evidences)
        try:
            raw = self._adapter.run(self._tokenizer(prompt))
            text = self._tokenizer.decode(raw[0]).strip()
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
            answer = parsed.get("answer", text)
            grounded = bool(parsed.get("grounded", True))
        except Exception as e:
            # Fail gracefully — degrade to extractive, label honestly.
            fb = ExtractiveGroundedAnswerer().generate(question, evidences)
            fb.notes = f"LLM path failed ({e}); used extractive fallback."
            fb.backend = f"{self.backend_name}->fallback"
            return fb
        if not grounded:
            return Answer(answer="The retrieved evidence was not sufficient for a confident answer.",
                          grounded=False, evidence=[], backend=self.backend_name)
        return Answer(
            answer=answer,
            grounded=True,
            evidence=[{"memory_id": e.memory_id, **e.to_dict()} for e in evidences],
            backend=self.backend_name,
            notes=f"EP={self._adapter.info.selected_provider}",
        )
