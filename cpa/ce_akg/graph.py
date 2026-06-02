"""CE-AKG over NetworkX (Section 2.7).

Nodes: targets, technologies, CVEs, published fake-CTI variants, observed victim entities.
Edges: relevance / mention / exploitation links. Provides:
  - e_kg embedding for the RL state (mean-pool or GAT),
  - an anomaly score feeding DetectionRisk in the reward.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np


class CEAKG:
    def __init__(self, embed_dim: int = 8) -> None:
        self.g = nx.MultiDiGraph()
        self.embed_dim = embed_dim

    def add_target(self, target_id: str, technologies: Iterable[str]) -> None:
        self.g.add_node(target_id, kind="target")
        for tech in technologies:
            self.g.add_node(tech, kind="tech")
            self.g.add_edge(target_id, tech, rel="uses")

    def add_poison(self, variant_id: int, entities: Iterable[str], relevance: float) -> None:
        node = f"poison:{variant_id}"
        self.g.add_node(node, kind="poison", relevance=relevance)
        for e in entities:
            self.g.add_node(e, kind="entity")
            self.g.add_edge(node, e, rel="mentions")

    def link_observation(self, variant_id: int, observed_entities: Iterable[str]) -> None:
        """Record which poisoned entities surfaced in victim output (feedback loop)."""
        node = f"poison:{variant_id}"
        for e in observed_entities:
            if self.g.has_node(node):
                self.g.add_edge(node, e, rel="surfaced")

    def embedding(self) -> np.ndarray:
        """e_kg via simple structural mean-pool. TODO(RQ2): replace with a GAT."""
        if self.g.number_of_nodes() == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        deg = np.array([d for _, d in self.g.degree()], dtype=np.float32)
        feats = np.zeros(self.embed_dim, dtype=np.float32)
        feats[0] = self.g.number_of_nodes()
        feats[1] = self.g.number_of_edges()
        feats[2] = float(deg.mean())
        feats[3] = float(deg.max())
        return feats

    def anomaly_score(self) -> float:
        """Higher when poison nodes are over-connected vs. organic CTI (detection risk).

        TODO(RQ2): calibrate against the seed-real subgraph distribution.
        """
        poison = [n for n, a in self.g.nodes(data=True) if a.get("kind") == "poison"]
        if not poison:
            return 0.0
        avg_poison_deg = np.mean([self.g.degree(n) for n in poison])
        all_deg = np.mean([d for _, d in self.g.degree()]) or 1.0
        return float(min(avg_poison_deg / all_deg / 3.0, 1.0))
