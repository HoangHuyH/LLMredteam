"""Reward function R (Eq. 1, Section 2.8).

R = 0.40*Stealth + 0.35*Impact - 0.15*PublishCost - 0.10*DetectionRisk
Weights are configurable via config.mdp.reward_weights.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardComponents:
    stealth: float          # mean inverse detection prob (GLTR/RoBERTa/cos-sim)
    impact: float           # f(PDS, FPR) inferred from victim output
    publish_cost: float     # #published + content complexity, normalized
    detection_risk: float   # P(human/detector flag) from CE-AKG anomaly score


DEFAULT_WEIGHTS = {
    "stealth": 0.40,
    "impact": 0.35,
    "publish_cost": -0.15,
    "detection_risk": -0.10,
}


def compute_reward(c: RewardComponents, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return (
        w["stealth"] * c.stealth
        + w["impact"] * c.impact
        + w["publish_cost"] * c.publish_cost
        + w["detection_risk"] * c.detection_risk
    )
