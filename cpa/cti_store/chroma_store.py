"""Faithful CTI retriever — ChromaDB + MiniLM, ported from DREAM's retrieval.

Mirrors references/dream/{vector_db_builder,mcp_retriever}.py:
  - same embedding model family (all-MiniLM-L6-v2); DREAM uses sentence-transformers, we use
    ChromaDB's DefaultEmbeddingFunction (same model via ONNX, no torch). Pass `embed_fn` to
    force exact-DREAM parity with sentence-transformers.
  - a per-record document string (`_generate_doc`, cf. DREAM `_generate_input_doc`),
  - cosine search returning ranked records (similarity = 1 - distance),
  - an id -> CTIRecord lookup table (cf. DREAM's mcp_lookup_table.pkl).

Drop-in for MockCTIStore: same add/seed_real/query/all/poison_records API, so PoisoningEnv
and run_episode work unchanged. The victim's RAG now retrieves by real semantic similarity,
which is what makes context-aware poison (Group B) measurably more effective (RQ1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from data.schema import CTIRecord, CTIEntities, Relevance

COLLECTION_NAME = "cti_collection"


class ChromaCTIStore:
    def __init__(
        self,
        path: str,
        embed_fn: Optional[Callable[[List[str]], list]] = None,
        collection_name: str = COLLECTION_NAME,
        in_memory: bool = False,
    ) -> None:
        import chromadb

        # in_memory (EphemeralClient) gives each episode a clean, isolated store — required for
        # RQ sweeps so poison never leaks across episodes. Persistent for single interactive runs.
        if in_memory:
            self._client = chromadb.EphemeralClient()
        else:
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
        self._embed_fn = embed_fn  # if set: we pass precomputed embeddings (DREAM-exact)

        if embed_fn is None:
            # ChromaDB default = all-MiniLM-L6-v2 (ONNX). Same model family as DREAM.
            from chromadb.utils import embedding_functions
            ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=collection_name, embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = self._client.get_or_create_collection(
                name=collection_name, embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )

        self._lookup: dict[str, CTIRecord] = {}  # id -> CTIRecord (like DREAM's pickle)
        self._counter = 0

    # --- document construction (cf. DREAM _generate_input_doc) -----------
    @staticmethod
    def _generate_doc(r: CTIRecord) -> str:
        ents = r.entities
        ent_str = ", ".join(filter(None, [
            "CVEs: " + "; ".join(ents.cves) if ents.cves else "",
            "vulns: " + "; ".join(ents.vulnerabilities) if ents.vulnerabilities else "",
        ]))
        base = f"topic: {r.topic}. {r.text()}"
        return f"{base} {ent_str}".strip()

    @staticmethod
    def _metadata(r: CTIRecord) -> dict:
        # ChromaDB metadata must be scalar — flatten lists to comma-joined strings.
        return {
            "topic": r.topic,
            "source_channel": r.source_channel,
            "is_poison": bool(r.is_poison),
            "target_relevance": r.target_relevance.value,
            "variant_id": -1 if r.variant_id is None else int(r.variant_id),
            "cves": ",".join(r.entities.cves),
        }

    # --- ingestion -------------------------------------------------------
    def add(self, record: CTIRecord) -> None:
        rid = f"cti-{self._counter}"
        self._counter += 1
        doc = self._generate_doc(record)
        kwargs = dict(ids=[rid], documents=[doc], metadatas=[self._metadata(record)])
        if self._embed_fn is not None:
            kwargs["embeddings"] = self._embed_fn([doc])
        self._collection.add(**kwargs)
        self._lookup[rid] = record

    def add_many(self, records: List[CTIRecord]) -> None:
        for r in records:
            self.add(r)

    def seed_real(self, jsonl_path: str, sample_size: int | None = None, seed: int = 0) -> int:
        """Load genuine CTI records. If sample_size is set, take a seeded random subset
        (keeps per-episode embedding cost bounded when the corpus is large, e.g. CASIE+CyEnts)."""
        p = Path(jsonl_path)
        if not p.exists():
            return 0
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if sample_size is not None and sample_size < len(lines):
            import random
            lines = random.Random(seed).sample(lines, sample_size)
        n = 0
        for line in lines:
            d = json.loads(line)
            self.add(CTIRecord(
                topic=d.get("topic", "unknown"),
                real_cti=d.get("real_cti") or d.get("text"),
                target_relevance=Relevance(d.get("target_relevance", "low")),
                entities=CTIEntities(**d.get("entities", {})),
                source_channel="seed-real",
                is_poison=False,
            ))
            n += 1
        return n

    # --- retrieval (cf. DREAM VectorRetriever.search) --------------------
    def query(self, query_text: str, k: int = 5) -> List[CTIRecord]:
        if not query_text:
            return []
        n = min(k, max(self._collection.count(), 1))
        qkwargs = dict(n_results=n)
        if self._embed_fn is not None:
            qkwargs["query_embeddings"] = self._embed_fn([query_text])
        else:
            qkwargs["query_texts"] = [query_text]
        res = self._collection.query(**qkwargs)
        ids = res.get("ids", [[]])[0]
        return [self._lookup[i] for i in ids if i in self._lookup]

    def query_scored(self, query_text: str, k: int = 5) -> List[tuple[CTIRecord, float]]:
        """Like query() but also returns cosine similarity (1 - distance), as DREAM does."""
        if not query_text:
            return []
        n = min(k, max(self._collection.count(), 1))
        qkwargs = dict(n_results=n)
        if self._embed_fn is not None:
            qkwargs["query_embeddings"] = self._embed_fn([query_text])
        else:
            qkwargs["query_texts"] = [query_text]
        res = self._collection.query(**qkwargs)
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for i, d in zip(ids, dists):
            if i in self._lookup:
                out.append((self._lookup[i], 1.0 - float(d)))
        return out

    # --- inventory -------------------------------------------------------
    def all(self) -> List[CTIRecord]:
        return list(self._lookup.values())

    def poison_records(self) -> List[CTIRecord]:
        return [r for r in self._lookup.values() if r.is_poison]

    # --- CE-AKG integration ---------------------------------------------
    def populate_ceakg(self, ceakg, target_id: str, technologies: List[str]) -> None:
        """Build the CE-AKG from current records (cf. proposal §2.7 world model)."""
        ceakg.add_target(target_id, technologies)
        for r in self.poison_records():
            relevance = {"low": 0.3, "medium": 0.6, "high": 0.9}[r.target_relevance.value]
            ents = r.entities.cves + r.entities.vulnerabilities
            ceakg.add_poison(r.variant_id if r.variant_id is not None else 0, ents, relevance)
