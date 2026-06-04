"""StealthEvaluator node — scores a fake CTI variant's detectability in-sandbox.

Aggregates detector probabilities (GLTR, RoBERTa) and cosine similarity to real CTI.
Per SAFETY.md this is for MEASUREMENT (UR/DS metrics), not for tuning live-feed survival.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


class StealthEvaluator:
    def __init__(self, detector_fns: List[Callable[[str], float]], real_corpus_emb=None,
                 embed_fn=None) -> None:
        self.detectors = detector_fns          # each: text -> P(AI-generated)
        self.real_corpus_emb = real_corpus_emb  # np.ndarray [N, d] of real CTI embeddings
        self.embed_fn = embed_fn

    def detector_probs(self, text: str) -> List[float]:
        return [float(d(text)) for d in self.detectors]

    def stealth_score(self, text: str) -> float:
        """Mean inverse detection probability, optionally blended with real-CTI similarity."""
        probs = self.detector_probs(text)
        inv = 1.0 - float(np.mean(probs)) if probs else 1.0
        if self.real_corpus_emb is not None and self.embed_fn is not None:
            e = self.embed_fn(text)
            sims = self.real_corpus_emb @ e / (
                np.linalg.norm(self.real_corpus_emb, axis=1) * np.linalg.norm(e) + 1e-9)
            inv = 0.5 * inv + 0.5 * float(np.max(sims))
        return inv

    def detection_risk(self, text: str, channel_prior: float) -> float:
        """Stealth as env.step sees it: max(content-detector risk, channel base prior).

        This MIRRORS PoisoningEnv.step exactly (max of the strongest detector prob and the
        channel's CHANNEL_DETECTION_RISK), so the graph path and the RL env stay numerically
        consistent. Used only when a turn actually publishes; a no-publish turn exposes nothing.
        """
        probs = self.detector_probs(text)
        detector_risk = float(np.max(probs)) if probs else 0.0
        return max(detector_risk, channel_prior)
