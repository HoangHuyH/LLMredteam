"""Efficiency metrics (Section 3.3): ATMI, Publish Cost."""
from __future__ import annotations

from typing import Optional, Sequence


def first_major_impact_turn(
    pds_per_turn: Sequence[float],
    fpr_per_turn: Sequence[float],
    *,
    pds_thr: float = 0.25,
    fpr_thr: float = 0.30,
) -> Optional[int]:
    """Earliest turn index (1-based) where PDS >= 0.25 OR FPR >= 0.30, else None."""
    for t, (pds, fpr) in enumerate(zip(pds_per_turn, fpr_per_turn), start=1):
        if pds >= pds_thr or fpr >= fpr_thr:
            return t
    return None


def avg_turns_to_major_impact(per_episode_first_impact: Sequence[Optional[int]]) -> Optional[float]:
    """ATMI = mean of first-major-impact turns over episodes that reached impact."""
    hits = [t for t in per_episode_first_impact if t is not None]
    if not hits:
        return None
    return sum(hits) / len(hits)


def publish_cost(num_published: int, content_complexity: float = 0.0) -> float:
    """Total fake CTI published in an episode (+ optional complexity term)."""
    return float(num_published) + content_complexity
