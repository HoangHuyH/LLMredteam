"""LangGraph wiring of the CPA nodes (Section 2.7).

  Generator ──► StealthEvaluator ──► LocalPublisher ──► (victim turn) ──► Observer
       ▲                                                                     │
       └─────────────────────── RL policy / reward ◄────────────────────────┘

Kept thin: the per-turn mechanics live in cpa/mdp/env.py. This module assembles the
graph for a full episode and is where curriculum / policy hooks attach.
TODO(RQ2): replace the linear scaffold with an actual langgraph.StateGraph.
"""
from __future__ import annotations

from typing import Callable

from cpa.orchestrator.egress_guard import install_guard, assert_no_real_publish


def build_pipeline(cfg: dict, env, policy: Callable):
    """Return an episode-runner closure. `policy(state) -> CPAAction`."""
    install_guard()
    assert_no_real_publish(cfg.get("sandbox", {}).get("allow_real_publish", False))

    def run_episode(max_turns: int):
        state = env.reset()
        trajectory = []
        for _ in range(max_turns):
            action = policy(state)
            state, reward, info = env.step(action)
            trajectory.append({"reward": reward, **{k: info[k] for k in ("pds", "fpr")}})
        return trajectory

    return run_episode
