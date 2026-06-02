"""CTI stores — the ONLY place fake CTI is ever written (sandbox).

- MockCTIStore: dependency-free, naive overlap ranking (fast scaffolding / CI).
- ChromaCTIStore: faithful ChromaDB + MiniLM retrieval, ported from DREAM (default for RQ1).
"""
from .mock_store import MockCTIStore

__all__ = ["MockCTIStore", "make_store"]


def make_store(cfg: dict):
    """Factory from config.cti_store.backend: 'chroma' | 'faiss'->chroma | 'jsonl'/'mock'."""
    backend = cfg.get("backend", "chroma")
    path = cfg.get("path", "./.sandbox/cti_store")
    if backend in ("chroma", "faiss"):   # 'faiss' alias kept for config compatibility
        from .chroma_store import ChromaCTIStore
        return ChromaCTIStore(
            path,
            collection_name=cfg.get("collection_name", "cti_collection"),
            in_memory=cfg.get("in_memory", False),
        )
    return MockCTIStore(path, backend)
