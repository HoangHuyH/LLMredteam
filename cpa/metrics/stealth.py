"""Stealth metrics (Section 3.3): Undetected Rate, Detection Score.

NOTE (SAFETY.md): these quantify detectability *inside the sandbox* to study the
attack surface. They must not be used to tune content for survival on live feeds.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def detection_score(detector_probs: Sequence[float]) -> float:
    """DS = mean over K detectors of P_detect,k (probability content is AI-generated)."""
    probs = list(detector_probs)
    if not probs:
        return 0.0
    return float(np.mean(probs))


def undetected_rate(
    per_item_detector_probs: Sequence[Sequence[float]],
    *,
    flag_threshold: float = 0.5,
) -> float:
    """UR = #fake CTI flagged by no detector / #total fake CTI * 100 (percent).

    `per_item_detector_probs[i]` = list of detector probabilities for fake-CTI item i.
    An item is "detected" if any detector's prob >= flag_threshold.
    """
    items = list(per_item_detector_probs)
    if not items:
        return 0.0
    undetected = sum(1 for probs in items if max(probs, default=0.0) < flag_threshold)
    return undetected / len(items) * 100.0
