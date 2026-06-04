"""LangGraph wiring of the CPA nodes (Section 2.7).

  Generator ──► StealthEvaluator ──► LocalPublisher ──► (victim turn) ──► Observer
       ▲                                                                     │
       └─────────────────────── RL policy / reward ◄────────────────────────┘

Two episode runners, SAME return contract (a `run_episode(max_turns)` closure returning a
list of per-turn dicts with at least reward/pds/fpr):

- build_pipeline           : the original thin LINEAR scaffold (drives env.step directly).
- build_langgraph_pipeline : a genuine langgraph.StateGraph over the four explicit CPA nodes
                             (cpa/orchestrator/nodes.py), each reusing the PoisoningEnv's
                             already-configured components. Used for curriculum / demonstration
                             / RQ trajectory generation — NOT the PPO inner loop (that stays in
                             cpa/rl). langgraph is an optional dep: if it is not installed we
                             print a one-line notice and fall back to the linear runner, so
                             nothing ever breaks locally.
"""
from __future__ import annotations

from typing import Callable

from cpa.orchestrator.egress_guard import install_guard, assert_no_real_publish
from cpa.orchestrator.stealth_evaluator import StealthEvaluator
from cpa.orchestrator.nodes import (
    generator_node, stealth_evaluator_node, publisher_node, observer_node,
)


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


def _init_state(env) -> dict:
    """Fresh per-episode graph state (env.reset() also resets env's internal counters)."""
    return {
        "obs": env.reset(),
        "action": None, "variant": None, "do_publish": False,
        "detection_risk": 0.0, "stealth": 1.0,
        "reward": 0.0, "pds": 0.0, "fpr": 0.0,
        "turn": 0, "trajectory": [],
    }


def build_langgraph_pipeline(cfg: dict, env, policy: Callable):
    """Return an episode runner driven by a real langgraph.StateGraph over the four CPA nodes.

    Falls back to build_pipeline (linear) if langgraph is not installed — never raises.
    """
    install_guard()
    assert_no_real_publish(cfg.get("sandbox", {}).get("allow_real_publish", False))

    # Lazy import: langgraph is an optional dependency (present on Kaggle, not in the local venv).
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Any, List
    except Exception:
        print("[orchestrator] langgraph not installed; "
              "build_langgraph_pipeline falling back to linear build_pipeline runner.")
        return build_pipeline(cfg, env, policy)

    # StealthEvaluator over the env's already-configured detector — mirrors env.step exactly.
    evaluator = StealthEvaluator([env.detect])

    class GraphState(TypedDict):
        obs: Any
        action: Any
        variant: Any
        do_publish: bool
        detection_risk: float
        stealth: float
        reward: float
        pds: float
        fpr: float
        turn: int
        trajectory: List[dict]
        max_turns: int

    # Each node closes over (env, policy, evaluator) and delegates to the shared node functions.
    def _generator(s):       return generator_node(env, policy, s)
    def _stealth(s):         return stealth_evaluator_node(env, evaluator, s)
    def _publisher(s):       return publisher_node(env, s)
    def _observer(s):        return observer_node(env, s)

    def _route(s):
        # Loop generator -> ... -> observer -> generator until max_turns, then END.
        return "generator" if s["turn"] < s["max_turns"] else END

    g = StateGraph(GraphState)
    g.add_node("generator", _generator)
    g.add_node("stealth", _stealth)
    g.add_node("publisher", _publisher)
    g.add_node("observer", _observer)
    g.set_entry_point("generator")
    g.add_edge("generator", "stealth")
    g.add_edge("stealth", "publisher")
    g.add_edge("publisher", "observer")
    g.add_conditional_edges("observer", _route, {"generator": "generator", END: END})
    app = g.compile()

    def run_episode(max_turns: int):
        state = _init_state(env)
        state["max_turns"] = max_turns
        # No recursion cap surprises: allow the loop to take all its turns through the graph.
        final = app.invoke(state, config={"recursion_limit": 4 * max_turns + 10})
        return final["trajectory"]

    return run_episode
