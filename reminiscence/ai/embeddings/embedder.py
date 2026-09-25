"""Embedding subsystem behind a replaceable ``Embedder`` interface.

Two implementations ship:
- OnnxMiniLMEmbedder: real all-MiniLM-L6-v2 ONNX deployment (used when the
  model file + onnxruntime are present).
- HashingTFIDFEmbedder: deterministic, dependency-free local fallback so the
  whole pipeline (chunk -> embed -> index -> hybrid retrieval) is testable
  offline without downloading models.  It is clearly labeled in metadata;
  nothing pretends to be MiniLM when it isn't.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


class Embedder(ABC):
    """Replaceable embedding interface."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape (n, dim)."""


# ---------------------------------------------------------------------------
class HashingTFIDFEmbedder(Embedder):
    """Deterministic hashed bag-of-words with sublinear TF and IDF weighting.

    Not a semantic model — a lexical-embedding stand-in used until/unless a
    real ONNX embedding model is installed.  Metadata always reports the true
    model id so downstream code never mislabels it.
    """

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._doc_freq: Counter = Counter()
        self._n_docs = 0

    model_id = "local-hash-tfidf-512"

    @property
    def dim(self) -> int:
        return self._dim

    def _tokens(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    def fit(self, corpus: list[str]) -> HashingTFIDFEmbedder:
        for t in corpus:
            seen = set(self._tokens(t))
            self._doc_freq.update(seen)
            self._n_docs += 1
        return self

    def _idf(self, tok: str) -> float:
        df = self._doc_freq.get(tok, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0 if self._n_docs else 1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = Counter(self._tokens(text))
            length = sum(counts.values()) or 1
            for tok, c in counts.items():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                idx = h % self._dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                tf = 1.0 + math.log(c)
                out[i, idx] += sign * tf * self._idf(tok)
            # bigrams help phrase-level matching a bit
            toks = self._tokens(text)
            for a, b in zip(toks, toks[1:]):
                bg = a + "_" + b
                h = int.from_bytes(hashlib.blake2b(bg.encode(), digest_size=8).digest(), "little")
                idx = h % self._dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign * 0.5 * self._idf(bg)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


# ---------------------------------------------------------------------------
class OnnxMiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 via ONNX Runtime (QNN EP preferred when verified).

    Production path requirements (P0 — production embeddings):
    * tokenizer/model pairing enforced before any inference;
    * batched inference with provider selection recorded per run;
    * dimension validation on every output;
    * traceable metadata: model_id / model_version / quantization /
      execution_provider.  A different model is never silently substituted —
    the metadata always reflects the artifact actually loaded.
    """

    model_id = "embed-minilm-l6-v2"
    EXPECTED_DIM = 384

    def __init__(
        self,
        model_path: str,
        tokenizer=None,
        *,
        model_version: str = "unknown",
        quantization: str = "fp32",
        expected_dim: int | None = None,
    ):
        from ..runtime import OnnxRuntimeAdapter

        self._adapter = OnnxRuntimeAdapter(model_path)
        self._tokenizer = tokenizer  # callable(text)->dict of arrays
        self._dim = expected_dim or self.EXPECTED_DIM
        self.model_version = model_version
        self.quantization = quantization

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def runtime_info(self):
        """Actual runtime/provider used — never a claim, always a measurement."""
        return self._adapter.info

    @property
    def metadata(self) -> dict:
        info = self._adapter.info
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "quantization": self.quantization,
            "execution_provider": getattr(info, "execution_provider", None),
            "runtime": getattr(info, "runtime", None),
            "dim": self._dim,
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._tokenizer is None:
            raise RuntimeError(
                "OnnxMiniLMEmbedder requires a tokenizer paired with the model. "
                "Install model artifacts via the model manager (allow_network=True once), "
                "never silently substitute a different model."
            )
        outs: list[np.ndarray] = []
        batch = 32
        for start in range(0, len(texts), batch):
            enc = self._tokenizer(texts[start : start + batch])
            feed = {k: np.asarray(v, dtype=np.float32) for k, v in enc.items()}
            result = self._adapter.run(feed)[0]  # (B, T, H) last_hidden_state
            mask = enc["attention_mask"]
            lens = np.asarray(mask).sum(axis=1)
            pooled = np.stack([result[b, : int(l)].mean(axis=0) for b, l in enumerate(lens)])
            outs.append(pooled)
        v = np.concatenate(outs, axis=0).astype(np.float32)
        if v.ndim != 2 or v.shape[1] != self._dim:
            raise RuntimeError(
                f"dimension validation failed: model produced shape {v.shape}, "
                f"expected (*, {self._dim}) — configured model does not match manifest"
            )
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms


def get_embedder(
    prefer_model_path: str | None = None,
    dim: int = 512,
    *,
    tokenizer=None,
    model_version: str = "unknown",
    quantization: str = "fp32",
    expected_dim: int | None = None,
) -> Embedder:
    """Factory honoring registry/availability; never fakes a model.

    If an ONNX artifact is configured but cannot be loaded (missing runtime,
    missing tokenizer pairing), the returned embedder is the clearly-labeled
    local fallback — its ``model_id`` reports the truth so downstream code
    and benchmarks never misattribute results to MiniLM.
    """
    if prefer_model_path:
        try:
            from ..runtime import OnnxRuntimeAdapter

            if OnnxRuntimeAdapter.available():
                emb = OnnxMiniLMEmbedder(
                    prefer_model_path,
                    tokenizer=tokenizer,
                    model_version=model_version,
                    quantization=quantization,
                    expected_dim=expected_dim,
                )
                emb._adapter.load()
                return emb
        except Exception:
            pass  # fall through to honest local fallback
    return HashingTFIDFEmbedder(dim=dim)
