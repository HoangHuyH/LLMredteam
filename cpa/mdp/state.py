"""Partial observation state o_t = [e_kg ; v_obs ; h_hist]  (Section 2.8)."""
from __future__ import annotations

from typing import List

import numpy as np


def build_state(
    ekg: np.ndarray,        # CE-AKG embedding (mean-pool / GAT)
    vobs: np.ndarray,       # parsed victim-output features
    history: List[np.ndarray],  # previous v_obs vectors
    history_window: int = 5,
) -> np.ndarray:
    """Concatenate the three components into the policy's observation."""
    if history:
        recent = history[-history_window:]
        hhist = np.mean(np.stack(recent, axis=0), axis=0)
    else:
        hhist = np.zeros_like(vobs)
    return np.concatenate([np.asarray(ekg, dtype=np.float32),
                           np.asarray(vobs, dtype=np.float32),
                           np.asarray(hhist, dtype=np.float32)])
