"""Pluggable embed_fn / detector_fn with zero-dependency fallbacks.

Lets the full pipeline run end-to-end before heavy models are installed. Swap in
Sentence-BERT / GLTR / RoBERTa for real experiments.
"""
from __future__ import annotations

import hashlib

import numpy as np


def make_embed_fn(cfg: dict):
    """Return text -> np.ndarray. Tries Sentence-BERT, falls back to hashing embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        name = cfg.get("sbert_model", "all-MiniLM-L6-v2")
        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(name, device=device)
        try:
            model.encode("_probe_")                      # CUDA-kernel-mismatch guard
        except Exception as e:
            print(f"[embed] {device} encode failed ({type(e).__name__}); falling back to CPU")
            model = SentenceTransformer(name, device="cpu")
        return lambda text: np.asarray(model.encode(text or ""), dtype=np.float32)
    except Exception:
        dim = cfg.get("fallback_dim", 64)

        def hashing_embed(text: str) -> np.ndarray:
            vec = np.zeros(dim, dtype=np.float32)
            for tok in (text or "").lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % dim] += 1.0
            n = np.linalg.norm(vec)
            return vec / n if n else vec

        return hashing_embed


def make_store_embed_fn(cfg: dict):
    """Return a BATCH embedder (List[str] -> list[list[float]]) for ChromaCTIStore, on GPU if
    available. This replaces Chroma's CPU ONNX MiniLM — the per-episode bottleneck — so re-embedding
    the corpus each episode is far faster on a Kaggle GPU. Returns None if sentence-transformers
    is unavailable (store then falls back to its built-in ONNX encoder)."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        name = cfg.get("sbert_model", "all-MiniLM-L6-v2")
        device = cfg.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(name, device=device)
        # Sanity-probe the device: a mismatched torch/GPU raises CUDA "no kernel image"; if so,
        # transparently fall back to CPU so the run still completes (just slower).
        try:
            model.encode(["_probe_"], convert_to_numpy=True)
        except Exception as e:
            print(f"[store_embed] {device} encode failed ({type(e).__name__}); falling back to CPU")
            model = SentenceTransformer(name, device="cpu")

        def embed(texts):
            arr = model.encode(list(texts), batch_size=cfg.get("embed_batch", 256),
                               convert_to_numpy=True, normalize_embeddings=False)
            return arr.tolist()
        return embed
    except Exception:
        return None


def make_detector_fn(cfg: dict):
    """Return text -> list[float] detector probs. Fallback is a length/keyword heuristic.

    TODO: replace with real GLTR + RoBERTa fake-CTI detectors.
    """
    def heuristic(text: str):
        t = (text or "")
        # crude "AI-ish" proxy: boilerplate phrasing raises suspicion
        score = 0.0
        for marker in ("immediate triage", "previously undisclosed", "active campaign"):
            if marker in t.lower():
                score += 0.2
        return [min(score, 1.0)]

    return heuristic
