"""Deterministic SQLite-backed memory graph service; no generative model required."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..memory.events import MemoryEvent
from ..storage.database import Database

@dataclass(frozen=True)
class GraphNode:
    node_type: str
    node_id: str
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.node_type}:{self.node_id}"

@dataclass(frozen=True)
class GraphEdge:
    source: GraphNode
    target: GraphNode
    relationship: str
    confidence: float = 1.0
    method: str = "unknown"
    evidence_id: Optional[str] = None

@dataclass
class GraphPath:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        score = 1.0
        for edge in self.edges:
            score *= max(0.0, min(1.0, edge.confidence))
        return score

class MemoryGraph:
    """Expose entity_links and memory relationships as one bounded graph."""

    def __init__(self, db: Database):
        self.db = db

    def resolve_entity(self, entity_type: str, name: str, min_confidence: float = 0.0) -> list[GraphNode]:
        """Resolve exact normalized names/aliases without an LLM."""
        kind = entity_type.strip().lower()
        needle = " ".join(name.casefold().split())
        if not needle:
            return []
        if kind == "person":
            result = []
            for row in self.db.list_persons():
                if float(row["confidence"] or 0.0) < min_confidence:
                    continue
                aliases = self._json_list(row["aliases"])
                names = [row["display_name"], *aliases]
                if any(" ".join(str(v).casefold().split()) == needle for v in names):
                    result.append(GraphNode("person", str(row["id"]), str(row["display_name"])))
            return result
        if kind == "place":
            result = []
            for row in self.db.list_places():
                names = [row["name"], row["city"], row["region"], row["country"]]
                if any(v and " ".join(str(v).casefold().split()) == needle for v in names):
                    result.append(GraphNode("place", str(row["id"]), str(row["name"] or row["city"] or row["id"])))
            return result
        return []

    def neighbors(self, node: GraphNode, min_confidence: float = 0.0) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        if node.node_type == "memory":
            for row in self.db.entity_links_for(node.node_id):
                target = GraphNode(str(row["entity_type"]), str(row["entity_id"]), str(row["entity_id"]))
                edges.append(GraphEdge(node, target, str(row["relationship_type"]), float(row["confidence"] or 0.0), str(row["method"] or "unknown"), row["evidence_id"]))
            for src, dst, confidence in self.db.related(node.node_id):
                other = dst if src == node.node_id else src
                edges.append(GraphEdge(node, GraphNode("memory", other, other), "RELATED_TO", float(confidence or 0.0), "memory_relationship"))
        else:
            with self.db._lock:
                rows = self.db._conn.execute(
                    "SELECT memory_id, relationship_type, confidence, method, evidence_id FROM entity_links WHERE entity_type=? AND entity_id=?",
                    (node.node_type, node.node_id),
                ).fetchall()
            for row in rows:
                mid = str(row["memory_id"])
                edges.append(GraphEdge(node, GraphNode("memory", mid, mid), str(row["relationship_type"]), float(row["confidence"] or 0.0), str(row["method"] or "unknown"), row["evidence_id"]))
        return [e for e in edges if e.confidence >= min_confidence]

    def walk(self, start: GraphNode, max_depth: int = 2, min_confidence: float = 0.0) -> list[GraphNode]:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        seen = {start.key}
        queue = deque([(start, 0)])
        result = []
        while queue:
            node, depth = queue.popleft()
            result.append(node)
            if depth >= max_depth:
                continue
            for edge in self.neighbors(node, min_confidence):
                if edge.target.key not in seen:
                    seen.add(edge.target.key)
                    queue.append((edge.target, depth + 1))
        return result

    def find_paths(self, start: GraphNode, target: GraphNode, max_depth: int = 3, min_confidence: float = 0.0, max_paths: int = 10) -> list[GraphPath]:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        queue = deque([(start, [start], [])])
        found: list[GraphPath] = []
        while queue and len(found) < max_paths:
            node, nodes, edges = queue.popleft()
            if node.key == target.key:
                found.append(GraphPath(nodes, edges))
                continue
            if len(edges) >= max_depth:
                continue
            for edge in self.neighbors(node, min_confidence):
                if any(n.key == edge.target.key for n in nodes):
                    continue
                queue.append((edge.target, nodes + [edge.target], edges + [edge]))
        return sorted(found, key=lambda p: (-p.confidence, len(p.edges)))

    def memories_for_entity(self, entity_type: str, entity_id: str, max_depth: int = 1, min_confidence: float = 0.0) -> list[MemoryEvent]:
        nodes = self.walk(GraphNode(entity_type.lower(), entity_id, entity_id), max_depth, min_confidence)
        result = []
        seen = set()
        for node in nodes:
            if node.node_type == "memory" and node.node_id not in seen:
                event = self.db.get_event(node.node_id)
                if event is not None:
                    seen.add(node.node_id)
                    result.append(event)
        return result

    def related_entities(self, memory_id: str, max_depth: int = 1, min_confidence: float = 0.0) -> list[GraphNode]:
        return [node for node in self.walk(GraphNode("memory", memory_id, memory_id), max_depth, min_confidence) if node.node_type != "memory"]

    @staticmethod
    def _json_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if not value:
            return []
        try:
            import json
            parsed = json.loads(str(value))
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []

__all__ = ["GraphNode", "GraphEdge", "GraphPath", "MemoryGraph"]