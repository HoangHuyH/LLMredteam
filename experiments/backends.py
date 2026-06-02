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
        model = SentenceTransformer(cfg.get("sbert_model", "all-MiniLM-L6-v2"))
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
