"""Vector retrieval behind a replaceable ``VectorIndex`` interface.

Ships with an exact NumPy brute-force implementation (fast enough for
tens of thousands of events on Snapdragon CPU/NPU-friendly BLAS).  FAISS or
hnswlib can be dropped in later by implementing the same interface — no
application logic changes required.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    """Abstract ANN/vector index keyed by memory event id."""

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self, query: np.ndarray, k: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def remove(self, ids: list[str]) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class NumpyVectorIndex(VectorIndex):
    """Exact cosine-similarity index stored as a contiguous matrix."""

    def __init__(self, dim: int | None = None):
        self._dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # normalized rows
        self._lock = threading.RLock()

    @property
    def dim(self) -> int | None:
        return self._dim

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids/vectors length mismatch")
        with self._lock:
            if self._dim is None:
                self._dim = vectors.shape[1]
            elif vectors.shape[1] != self._dim:
                raise ValueError(f"expected dim {self._dim}, got {vectors.shape[1]}")
            vecs = l2_normalize(vectors)
            # Replace existing ids in place
            existing = {i: idx for idx, i in enumerate(self._ids)}
            new_mask = np.zeros((len(ids),), dtype=bool)
            for j, _ in enumerate(ids):
                new_mask[j] = ids[j] not in existing
            updates = [(existing[i], j) for j, i in enumerate(ids) if i in existing]
            if self._matrix is None:
                self._matrix = vecs.copy()
            elif (~new_mask).all() and len(ids) > 0:
                self._matrix = self._matrix.copy()
            else:
                parts = [self._matrix] if self._matrix is not None else []
                if new_mask.any():
                    parts.append(vecs[new_mask])
                self._matrix = np.vstack(parts) if parts else self._matrix
            for row, j in updates:
                assert self._matrix is not None
                self._matrix[row] = vecs[j]
            for j in np.nonzero(new_mask)[0]:
                self._ids.append(ids[int(j)])

    def search(
        self, query: np.ndarray, k: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query must be 1-D")
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return []
            if q.shape[0] != self._dim:
                raise ValueError(f"query dim {q.shape[0]} != index dim {self._dim}")
            qn = l2_normalize(q)
            sims = self._matrix @ qn
            ids = np.asarray(self._ids)
            if allowed_ids is not None:
                mask = np.isin(ids, list(allowed_ids))
                sims = np.where(mask, sims, -np.inf)
            k = min(k, int(np.sum(np.isfinite(sims))))
            if k <= 0:
                return []
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            return [(str(ids[i]), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def remove(self, ids: list[str]) -> None:
        drop = set(ids)
        with self._lock:
            keep = [i for i, mid in enumerate(self._ids) if mid not in drop]
            if len(keep) == len(self._ids):
                return
            self._ids = [self._ids[i] for i in keep]
            if self._matrix is not None:
                self._matrix = self._matrix[keep] if keep else self._matrix[:0]

    def state(self) -> tuple[list[str], np.ndarray | None]:
        with self._lock:
            return list(self._ids), (None if self._matrix is None else self._matrix.copy())

    def load_state(self, ids: list[str], matrix: np.ndarray | None) -> None:
        with self._lock:
            self._ids = list(ids)
            self._matrix = None if matrix is None else np.asarray(matrix, dtype=np.float32)
            if self._matrix is not None and self._matrix.size:
                self._dim = int(self._matrix.shape[1])

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)
