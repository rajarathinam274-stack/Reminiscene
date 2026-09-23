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
from typing import Optional

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

    def fit(self, corpus: list[str]) -> "HashingTFIDFEmbedder":
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
    """all-MiniLM-L6-v2 via ONNX Runtime (QNN EP preferred when verified)."""

    model_id = "embed-minilm-l6-v2"

    def __init__(self, model_path: str, tokenizer=None):
        from ..runtime import OnnxRuntimeAdapter

        self._adapter = OnnxRuntimeAdapter(model_path)
        self._tokenizer = tokenizer  # callable(text)->dict of arrays
        self._dim = 384

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def runtime_info(self):
        return self._adapter.info

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._tokenizer is None:
            raise RuntimeError(
                "OnnxMiniLMEmbedder requires a tokenizer (WordPiece). "
                "Install model artifacts via scripts/setup_models.py."
            )
        outs: list[np.ndarray] = []
        batch = 32
        for start in range(0, len(texts), batch):
            enc = self._tokenizer(texts[start:start + batch])
            feed = {k: np.asarray(v, dtype=np.float32) for k, v in enc.items()}
            result = self._adapter.run(feed)[0]      # (B, T, 384) last_hidden_state
            mask = enc["attention_mask"]
            lens = np.asarray(mask).sum(axis=1)
            pooled = np.stack([result[b, : int(l)] .mean(axis=0) for b, l in enumerate(lens)])
            outs.append(pooled)
        v = np.concatenate(outs, axis=0).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms


def get_embedder(prefer_model_path: Optional[str] = None, dim: int = 512) -> Embedder:
    """Factory honoring registry/availability; never fakes a model."""
    if prefer_model_path:
        try:
            from ..runtime import OnnxRuntimeAdapter
            if OnnxRuntimeAdapter.available():
                emb = OnnxMiniLMEmbedder(prefer_model_path)
                emb._adapter.load()
                return emb
        except Exception:
            pass  # fall through to honest local fallback
    return HashingTFIDFEmbedder(dim=dim)
