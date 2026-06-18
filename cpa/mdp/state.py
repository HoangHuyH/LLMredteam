"""Partial observation state o_t = [e_kg ; v_obs ; h_hist ; pressure]  (Section 2.8).

Dimension: 8 + 7 + 7 + 1 = 23.
  e_kg(8)    — CE-AKG embedding (mean-pool / GAT)
  v_obs(7)   — parsed victim-output features
  h_hist(7)  — mean of the last history_window v_obs vectors
  pressure(1) — reactive detection-heat accumulated by the attacker (Section 2.8 reactive env)
"""
from __future__ import annotations

from typing import List

import numpy as np


def build_state(
    ekg: np.ndarray,
    vobs: np.ndarray,
    history: List[np.ndarray],
    history_window: int = 5,
    pressure: float = 0.0,
) -> np.ndarray:
    """Concatenate the four components into the policy's observation (23-dim)."""
    if history:
        recent = history[-history_window:]
        hhist = np.mean(np.stack(recent, axis=0), axis=0)
    else:
        hhist = np.zeros_like(vobs)
    return np.concatenate([
        np.asarray(ekg, dtype=np.float32),
        np.asarray(vobs, dtype=np.float32),
        np.asarray(hhist, dtype=np.float32),
        np.array([pressure], dtype=np.float32),
    ])
