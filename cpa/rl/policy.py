"""Policies mapping state -> CPAAction.

Annotation key
--------------
[DREAM-CONCEPT] — design adapts a concept from DREAM (Lu et al., 2026); no DREAM code imported.
[CPA-ORIGINAL]  — specific to this work.

- RandomPolicy        : [CPA-ORIGINAL] uniform random baseline.
- DreamBaselinePolicy : [DREAM-CONCEPT] greedy heuristic inspired by DREAM's non-adaptive
                        C-GPS behaviour (always publish, max reach, no look-ahead, no learning).
                        NOT used in the 5 RQ2 ablation arms — superseded by MCTSPolicy.
- CPAPolicy           : [CPA-ORIGINAL] hybrid-RL policy (PPO). Trained SB3 model when provided;
                        otherwise burst-then-throttle heuristic.

RQ2 arms: cpa (CPAPolicy+PPO) vs mcts (MCTSPolicy in mcts_policy.py, DREAM-inspired C-GPS+UCT)
vs random vs nopoison. MCTSPolicy uses relevance-ranked C-GPS variant selection (adapting
DREAM's VectorRetriever.search) combined with a CPA-original UCT surrogate planner.
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
    """Greedy heuristic baseline inspired by DREAM's non-adaptive C-GPS behaviour.

    [DREAM-CONCEPT] Models what DREAM does without MCTS or relevance ranking: always publishes
    every turn on the highest-reach channel, cycling through variants sequentially. This is the
    degenerate case of C-GPS with no context guidance or look-ahead.
    [CPA-ORIGINAL] Channel choice (PASTEBIN) and always-publish rule are CPA-specific.

    NOT used in the 5 RQ2 ablation arms (superseded by MCTSPolicy as the DREAM baseline).
    Kept for reference and quick smoke tests.
    """

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


class ConstantPolicy:
    """Open-loop 'best fixed' baseline: publish once on a low-risk channel, then stop.

    RQ2 baseline: if the RL policy cannot beat this, it has not learned anything useful.
    publish_once=True means the first call fires frequency=1.0; all subsequent calls are
    frequency=0.0 (still-stealthy throttle). The victim stays derailed because it reads the
    same poisoned store every turn after publication.
    """

    def __init__(self, pool_size: int, **_) -> None:
        self.pool_size = pool_size
        self._t = 0

    def __call__(self, state: np.ndarray) -> CPAAction:
        first = self._t == 0
        self._t += 1
        return CPAAction(
            variant_id=0,
            channel_id=int(Channel.SECURITY_BLOG),  # lowest-risk channel
            frequency=1.0 if first else 0.0,
            timing=0.0,
        )


def decode_action(action, pool_size: int) -> CPAAction:
    """Decode a flat SB3 MultiDiscrete/Box action into a CPAAction."""
    a = np.atleast_1d(action)
    return CPAAction(
        variant_id=int(a[0]) % pool_size,
        channel_id=int(a[1]) % ACTION_SPACE["n_channels"] if a.size > 1 else 0,
        frequency=float(a[2]) / max(ACTION_SPACE["frequency_bins"] - 1, 1) if a.size > 2 else 0.5,
        timing=float(a[3]) / max(ACTION_SPACE["timing_bins"] - 1, 1) if a.size > 3 else 0.5,
    )


def make_policy(name: str, pool_size: int, seed: int = 0, model=None, **kwargs):
    if name == "random":
        return RandomPolicy(pool_size, rng_seed=seed)
    if name == "dream":
        return DreamBaselinePolicy(pool_size)
    if name == "cpa":
        return CPAPolicy(pool_size, model=model)
    if name in ("none", "nopoison"):
        return NoPoisonPolicy(pool_size)
    if name == "constant":
        return ConstantPolicy(pool_size)
    if name == "heuristic":
        return CPAPolicy(pool_size, model=None)  # burst-then-throttle heuristic, no PPO model
    if name in ("mcts", "mcts_only"):
        # [DREAM-CONCEPT: C-GPS] Pass target_profile + variant_profiles so MCTSPolicy can rank
        # variants by semantic relevance, adapting DREAM's VectorRetriever.search(query, k).
        # Imported lazily so lightweight policies above never pull the MCTS module.
        from cpa.rl.mcts_policy import MCTSPolicy, MCTSOnlyPolicy
        cls = MCTSPolicy if name == "mcts" else MCTSOnlyPolicy
        return cls(
            pool_size,
            seed=seed,
            target_profile=kwargs.get("target_profile", ""),
            variant_profiles=kwargs.get("variant_profiles"),
            embed_fn=kwargs.get("embed_fn"),
        )
    raise ValueError(f"Unknown policy: {name!r}")
