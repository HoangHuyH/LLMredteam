"""CE-AKG over NetworkX (Section 2.7).

Annotation key
--------------
[DREAM-CONCEPT] — design adapts a concept from DREAM (Lu et al., 2026); no DREAM code imported.
[CPA-ORIGINAL]  — specific to this work.

DREAM's CE-AKG is NOT an explicit class in DREAM's codebase. It is realised implicitly via:
  - VectorRetriever (mcp_retriever.py): ChromaDB + SentenceTransformer tracking cross-env
    attack primitive embeddings.
  - InteractiveAgent session state: tracks which environments have been seeded/exploited.

[DREAM-CONCEPT] This class provides a named graph equivalent: tracks targets, technologies,
CVEs, published fake-CTI variants, and observed victim entities with explicit edges for
relevance / mention / surfaced relationships. The contextual_search method adapts DREAM's
VectorRetriever.search to select the most relevant poison variants.

[CPA-ORIGINAL] NetworkX storage, 8-dim structural embedding, and anomaly score are CPA-specific.

Provides:
  - e_kg embedding for the RL state (mean-pool; GAT is a future improvement),
  - anomaly_score feeding DetectionRisk in the reward,
  - contextual_search for C-GPS-style variant retrieval.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

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
        """e_kg via structural mean-pool.

        [DREAM-CONCEPT: CE-AKG] DREAM's CE-AKG fuses cross-environment attack intelligence
        into a unified world model tracked by VectorRetriever embeddings. Here we encode
        graph structure as a fixed-dim vector for the RL state.
        [CPA-ORIGINAL] GAT-based embedding noted as TODO; structural mean-pool used for now.
        """
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
        """[DREAM-CONCEPT: CE-AKG anomaly] Poison connectivity ratio as detection risk signal.

        [CPA-ORIGINAL] Poison node degree vs. graph mean degree, capped at 1.0.
        TODO(RQ2): calibrate against the seed-real subgraph distribution.
        """
        poison = [n for n, a in self.g.nodes(data=True) if a.get("kind") == "poison"]
        if not poison:
            return 0.0
        avg_poison_deg = np.mean([self.g.degree(n) for n in poison])
        all_deg = np.mean([d for _, d in self.g.degree()]) or 1.0
        return float(min(avg_poison_deg / all_deg / 3.0, 1.0))

    def contextual_search(
        self,
        query: str,
        variant_id_to_text: Dict[int, str],
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k poison variants most relevant to query, with CE-AKG structural boost.

        [DREAM-CONCEPT: VectorRetriever + CE-AKG]
        DREAM's VectorRetriever.search(query_text, k) (mcp_retriever.py) ranks atomic attack
        primitives by SentenceTransformer cosine similarity to the target context. This method
        adapts that concept for the CTI poisoning graph:
          1. Base score: keyword-overlap cosine similarity between query and variant text
             (proxy for SentenceTransformer dense embeddings).
          2. Structural bonus: variants whose entities were already observed (surfaced) in
             victim output receive a boost, mirroring DREAM's CE-AKG tracking of successful
             cross-environment entity propagation.

        Parameters
        ----------
        query:
            Target profile or context string (e.g. technology stack description).
        variant_id_to_text:
            Mapping from variant_id (int) to its text description.
        k:
            Number of top results to return.

        Returns
        -------
        List of (variant_id, score) sorted by descending relevance.
        """
        if not variant_id_to_text:
            return []

        q_tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
        scores: Dict[int, float] = {}
        for vid, text in variant_id_to_text.items():
            t_tokens = set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
            if q_tokens and t_tokens:
                sim = len(q_tokens & t_tokens) / math.sqrt(len(q_tokens) * len(t_tokens))
            else:
                sim = 0.0

            # [DREAM-CONCEPT: CE-AKG structural bonus] Variants with confirmed entity surfacing
            # get a boost — the CE-AKG tracks successful cross-environment propagation.
            node = f"poison:{vid}"
            surfaced_edges = sum(
                1 for _, _, d in self.g.out_edges(node, data=True)
                if d.get("rel") == "surfaced"
            ) if self.g.has_node(node) else 0
            structural_bonus = min(surfaced_edges * 0.05, 0.3)

            scores[vid] = sim + structural_bonus

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
