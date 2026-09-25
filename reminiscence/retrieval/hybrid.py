"""Hybrid retrieval: lexical (FTS5) + semantic (vector) + metadata signals,
fused, reranked, and returned as scored candidates.

Score = alpha*semantic + beta*lexical + gamma*temporal + delta*modality
      + epsilon*source

Weights are configuration (tunable/evaluable), NOT claimed-optimal constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ai.embeddings.embedder import Embedder
from ..memory.events import MemoryEvent, Modality
from ..storage.database import Database
from ..storage.vector_index import VectorIndex
from .query_classifier import ClassifiedQuery, QueryCategory


@dataclass
class RetrievalWeights:
    alpha: float = 0.45  # semantic similarity
    beta: float = 0.35  # lexical relevance (BM25 normalized)
    gamma: float = 0.10  # temporal relevance
    delta: float = 0.05  # modality relevance
    epsilon: float = 0.05  # source relevance
    zeta: float = 0.08  # graph/entity relevance

    @staticmethod
    def from_db(db: Database) -> RetrievalWeights:
        w = db.get_setting("retrieval_weights")
        if isinstance(w, dict):
            return RetrievalWeights(
                **{
                    k: float(v)
                    for k, v in w.items()
                    if k in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
                }
            )
        return RetrievalWeights()

    def to_db(self, db: Database) -> None:
        db.set_setting("retrieval_weights", self.__dict__)


@dataclass
class Candidate:
    event: MemoryEvent
    score: float
    semantic: float = 0.0
    lexical: float = 0.0
    temporal: float = 0.0
    modality_rel: float = 0.0
    source_rel: float = 0.0
    graph_rel: float = 0.0
    channels: list[str] = field(default_factory=list)


def _normalize_bm25(ranks: list[float]) -> list[float]:
    """Map negative-is-better bm25 values into [0,1] higher-is-better."""
    if not ranks:
        return []
    vals = [-r for r in ranks]  # now higher is better
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


class HybridRetriever:
    def __init__(
        self,
        db: Database,
        index: VectorIndex,
        embedder: Embedder,
        weights: RetrievalWeights | None = None,
    ):
        self.db = db
        self.index = index
        self.embedder = embedder
        self.weights = weights or RetrievalWeights.from_db(db)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        cq: ClassifiedQuery,
        top_k: int = 10,
        candidate_pool: int = 60,
    ) -> list[Candidate]:
        w = self.weights
        pool: dict[str, Candidate] = {}

        # --- Stage 1: lexical (FTS5) ------------------------------------
        lex_query = " ".join(cq.terms) or cq.text
        lex_hits = self.db.fts_search(lex_query, limit=candidate_pool)
        norm_lex = _normalize_bm25([r for _, r in lex_hits])
        for (ev, _rank), ls in zip(lex_hits, norm_lex):
            c = pool.setdefault(ev.id, Candidate(event=ev, score=0.0))
            c.lexical = max(c.lexical, ls)
            c.channels.append("lexical")

        # --- Stage 2: semantic (vector) ----------------------------------
        try:
            qvec = self.embedder.embed([cq.text])[0]
            sem_hits = self.index.search(qvec, k=candidate_pool)
        except ValueError:
            sem_hits = []
        for eid, sim in sem_hits:
            ev = self.db.get_event(eid)
            if ev is None:
                continue
            c = pool.setdefault(eid, Candidate(event=ev, score=0.0))
            c.semantic = max(c.semantic, max(0.0, sim))
            c.channels.append("semantic")

        # --- Stage 3: metadata filters / boosts ---------------------------
        for c in pool.values():
            ev = c.event
            # temporal relevance — measured against MEMORY time
            # (event_time_start/captured_at), never indexing time.
            if cq.time_range:
                start_iso = cq.time_range[0].isoformat()
                end_iso = cq.time_range[1].isoformat()
                mem_time = ev.effective_event_time()
                in_window = bool(mem_time and start_iso <= mem_time <= end_iso)
                c.temporal = 1.0 if in_window else 0.0
            else:
                c.temporal = 0.5  # neutral when no temporal constraint
            # modality relevance
            if cq.modality_hint:
                c.modality_rel = 1.0 if ev.modality.value == cq.modality_hint else 0.0
            elif QueryCategory.VISUAL in cq.categories:
                c.modality_rel = (
                    1.0 if ev.modality in (Modality.IMAGE, Modality.VIDEO, Modality.PDF) else 0.2
                )
            else:
                c.modality_rel = 0.5
            # source relevance
            if cq.source_hint:
                src = self.db.get_source(ev.source_id)
                name = src["name"].lower() if src else ""
                c.source_rel = 1.0 if cq.source_hint.lower().strip() in name else 0.0
            else:
                c.source_rel = 0.5

        # --- Stage 2b: graph/entity candidates ------------------------------
        # Memories linked (via evidence-backed entity_links) to entities the
        # query mentions enter the pool directly.  This is how "meeting with
        # Alex last month" can surface memories that never contain "Alex".
        for etype, eids in getattr(cq, "entities", {}).items():
            for eid in eids:
                for ev in self.db.events_by_entity(etype, eid):
                    c = pool.setdefault(ev.id, Candidate(event=ev, score=0.0))
                    c.graph_rel = max(c.graph_rel, 1.0)
                    if "graph" not in c.channels:
                        c.channels.append("graph")
        # second-degree expansion from strong lexical/semantic hits
        seeds = [c.event.id for c in pool.values() if c.lexical > 0.7 or c.semantic > 0.7][:5]
        for sid in seeds:
            for _s, tgt, conf in self.db.related(sid)[:8]:
                tev = self.db.get_event(tgt)
                if tev is None:
                    continue
                c = pool.setdefault(tev.id, Candidate(event=tev, score=0.0))
                c.graph_rel = max(c.graph_rel, min(1.0, conf * 0.6))
                if "graph" not in c.channels:
                    c.channels.append("graph")

        # hard filter: temporal lookup must respect the window
        if cq.time_range and QueryCategory.TEMPORAL_LOOKUP in cq.categories:
            for eid in [e for e, c in pool.items() if c.temporal == 0.0]:
                del pool[eid]

        # --- Stage 4: fusion ------------------------------------------------
        for c in pool.values():
            c.score = (
                w.alpha * c.semantic
                + w.beta * c.lexical
                + w.gamma * c.temporal
                + w.delta * c.modality_rel
                + w.epsilon * c.source_rel
                + w.zeta * c.graph_rel
            )

        ranked = sorted(pool.values(), key=lambda c: c.score, reverse=True)

        # --- Stage 5: rerank + stage 6: evidence selection ------------------
        ranked = self._rerank(cq, ranked)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def _rerank(self, cq: ClassifiedQuery, ranked: list[Candidate]) -> list[Candidate]:
        """Lightweight learning-to-rank heuristics driven by query class.

        Cross-source queries penalize single-source domination; visual
        queries boost events with spatial anchors; explanation queries boost
        longer, structured chunks.
        """
        if not ranked:
            return ranked
        cats = set(cq.categories)

        if QueryCategory.CROSS_SOURCE in cats:
            per_source: dict[str, int] = {}
            out: list[Candidate] = []
            rest: list[Candidate] = []
            for c in ranked:
                n = per_source.get(c.event.source_id, 0)
                if n < 3:
                    out.append(c)
                    per_source[c.event.source_id] = n + 1
                else:
                    rest.append(c)
            ranked = out + rest

        if QueryCategory.VISUAL in cats:
            for c in ranked:
                ev = c.event
                if ev.location is not None or ev.metadata.get("keyframe"):
                    c.score += 0.08
                if ev.modality == Modality.IMAGE:
                    c.score += 0.05

        if QueryCategory.EXPLANATION in cats:
            for c in ranked:
                words = len(c.event.content.split())
                c.score += min(0.05, words / 4000.0)
                if c.event.section:
                    c.score += 0.02

        if QueryCategory.SOURCE_LOOKUP in cats and cq.source_hint:
            for c in ranked:
                src = self.db.get_source(c.event.source_id)
                if src and cq.source_hint.lower().strip() in src["name"].lower():
                    c.score += 0.15

        return sorted(ranked, key=lambda c: c.score, reverse=True)
