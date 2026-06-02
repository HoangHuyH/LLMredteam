"""Local mock CTI feed standing in for OTX/GitHub/blogs (sandbox only).

The victim queries this store via RAG exactly as it would query a real feed, so RQ1-3
are answerable without touching any real platform. Records persist to a local path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from data.schema import CTIRecord, CTIEntities, Relevance


class MockCTIStore:
    def __init__(self, path: str, backend: str = "jsonl") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self._records: List[CTIRecord] = []
        self._index = None  # TODO(RQ1): FAISS index when backend == "faiss"

    # --- ingestion -------------------------------------------------------
    def add(self, record: CTIRecord) -> None:
        """Add a CTI record to the LOCAL store. No network involved by design."""
        self._records.append(record)

    def seed_real(self, jsonl_path: str, sample_size: int | None = None, seed: int = 0) -> int:
        """Load genuine CTI used as background corpus the poison must blend into.
        If sample_size is set, take a seeded random subset."""
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

    # --- retrieval (what the victim sees) --------------------------------
    def query(self, query_text: str, k: int = 5) -> List[CTIRecord]:
        """Return top-k CTI records for a victim query.

        TODO(RQ1): replace the naive overlap ranking with FAISS + Sentence-BERT
        to faithfully mimic the victim's RAG retrieval.
        """
        terms = {t.lower() for t in query_text.split()}

        def score(r: CTIRecord) -> int:
            return len(terms & set(r.text().lower().split()))

        return sorted(self._records, key=score, reverse=True)[:k]

    def all(self) -> List[CTIRecord]:
        return list(self._records)

    def poison_records(self) -> List[CTIRecord]:
        return [r for r in self._records if r.is_poison]
