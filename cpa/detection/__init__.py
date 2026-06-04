"""AI-/machine-generated CTI text detectors (sandbox red-team metrics).

Real replacements for the keyword heuristic: a RoBERTa fake/AI-text classifier and a
GLTR-style GPT-2 log-rank feature, plus an ensemble. All heavy deps (torch/transformers)
are lazy-imported and optional; callers fall back to the keyword heuristic if unavailable.
"""
from .detector import make_real_detector

__all__ = ["make_real_detector"]
