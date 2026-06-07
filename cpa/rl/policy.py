"""Policies mapping state -> CPAAction.

- RandomPolicy        : random everything (Group-A stealth baseline).
- DreamBaselinePolicy : the DREAM C-GPS / AutoADAPTer baseline — greedy, non-adaptive:
                        publishes every turn on a high-reach but noisy channel. No learning.
- CPAPolicy           : our hybrid-RL policy (PPO+LoRA). Uses a trained SB3 model when
                        provided; otherwise a stealth-aware heuristic (burst-then-throttle on a
                        low-detection channel) so the pipeline runs before training.

RQ2 compares DreamBaselinePolicy vs CPAPolicy on ASR + stealth.
"""
from __future__ import annotations

import numpy as np

from cpa.mdp.action import CPAAction, ACTION_SPACE, Channel


class RandomPolicy:
    def __init__(self, pool_size: int, rng_seed: int = 0) -> None:
        self.pool_size = pool_size
        self._rng = np.random.default_rng(rng_seed)

    def __call__(self, state: np.ndarray) -> CPAAction:
        return CPAAction(
            variant_id=int(self._rng.integers(self.pool_size)),
            channel_id=int(self._rng.integers(ACTION_SPACE["n_channels"])),
            frequency=float(self._rng.random()),
            timing=float(self._rng.random()),
        )


class DreamBaselinePolicy:
    """Greedy C-GPS baseline: always publish, maximize reach, ignore stealth/cost."""

    def __init__(self, pool_size: int, **_) -> None:
        self.pool_size = pool_size
        self._i = 0

    def __call__(self, state: np.ndarray) -> CPAAction:
        v = self._i % self.pool_size
        self._i += 1
        return CPAAction(
            variant_id=v,
            channel_id=int(Channel.PASTEBIN),  # high reach, high detectability
            frequency=1.0,                      # publish every single turn
            timing=0.0,
        )


class CPAPolicy:
    """Hybrid-RL policy. Trained SB3 model if given, else a stealth-aware heuristic."""

    def __init__(self, pool_size: int, model=None, burst_turns: int = 4, **_) -> None:
        self.pool_size = pool_size
        self.model = model
        self.burst_turns = burst_turns
        self._t = 0

    def __call__(self, state: np.ndarray) -> CPAAction:
        if self.model is not None:
            action, _ = self.model.predict(state, deterministic=True)
            return decode_action(action, self.pool_size)
        # Heuristic: burst high-relevance poison early to derail, then throttle on a stealthy
        # channel to preserve impact while minimizing detection (the behavior PPO should learn).
        self._t += 1
        if self._t <= self.burst_turns:
            return CPAAction(variant_id=0, channel_id=int(Channel.SECURITY_BLOG),
                             frequency=1.0, timing=0.0)
        return CPAAction(variant_id=0, channel_id=int(Channel.SECURITY_BLOG),
                         frequency=0.0, timing=1.0)  # stop publishing; victim stays derailed


class NoPoisonPolicy:
    """Control baseline (Section 3.2 'No Poisoning'): never publishes.

    Frequency is held below any publish threshold so no fake CTI ever enters the feed; the victim
    plans against the genuine corpus only. This is the ground-truth arm that PDS/FPR are normalized
    against — any deviation it shows is baseline drift, not attack impact.
    """

    def __init__(self, pool_size: int, **_) -> None:
        self.pool_size = pool_size

    def __call__(self, state: np.ndarray) -> CPAAction:
        return CPAAction(variant_id=0, channel_id=0, frequency=0.0, timing=0.0)


def decode_action(action, pool_size: int) -> CPAAction:
    """Decode a flat SB3 MultiDiscrete/Box action into a CPAAction."""
    a = np.atleast_1d(action)
    return CPAAction(
        variant_id=int(a[0]) % pool_size,
        channel_id=int(a[1]) % ACTION_SPACE["n_channels"] if a.size > 1 else 0,
        frequency=float(a[2]) / max(ACTION_SPACE["frequency_bins"] - 1, 1) if a.size > 2 else 0.5,
        timing=float(a[3]) / max(ACTION_SPACE["timing_bins"] - 1, 1) if a.size > 3 else 0.5,
    )


def make_policy(name: str, pool_size: int, seed: int = 0, model=None):
    if name == "random":
        return RandomPolicy(pool_size, rng_seed=seed)
    if name == "dream":
        return DreamBaselinePolicy(pool_size)
    if name == "cpa":
        return CPAPolicy(pool_size, model=model)
    if name in ("none", "nopoison"):
        return NoPoisonPolicy(pool_size)
    if name in ("mcts", "mcts_only"):
        # DREAM planning baseline (C-GPS+MCTS). Imported lazily so the lightweight policies above
        # never pull the MCTS module unless asked for.
        from cpa.rl.mcts_policy import MCTSPolicy, MCTSOnlyPolicy
        return (MCTSPolicy if name == "mcts" else MCTSOnlyPolicy)(pool_size, seed=seed)
    raise ValueError(f"Unknown policy: {name!r}")
