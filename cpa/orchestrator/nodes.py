"""The four CPA orchestrator nodes (Section 2.7), as plain callables over a shared state.

Each node does REAL work by reusing the components already wired into a PoisoningEnv
(env.victim / env.store / env.publisher / env.pool / env.embed / env.detect / env.gt /
env.target / env.cfg / env._baseline_emb). They are deliberately env-state-mutating in the
same order as PoisoningEnv.step so the explicit-graph episode runner stays numerically
consistent with the gym/RL path. The langgraph StateGraph in graph.py wires these as nodes;
the linear fallback can drive the exact same functions.

State carried turn-to-turn (a TypedDict at the graph level, a plain dict here):
  obs           : np.ndarray   PO-MDP state o_t fed to the policy
  action        : CPAAction    policy's decision for this turn
  variant       : CTIRecord    pool[action.variant_id % len]
  do_publish    : bool         frequency gate result
  detection_risk: float        exposure this turn (0.0 if no publish)
  stealth       : float        1 - detection_risk
  reward/pds/fpr: float        per-turn metrics
  turn          : int          turn index
  trajectory    : list[dict]   accumulated per-turn rows
"""
from __future__ import annotations

import numpy as np

from cpa.mdp.env import CHANNEL_DETECTION_RISK
from cpa.mdp.state import build_state
from cpa.mdp.reward import RewardComponents, compute_reward
from cpa.observer.parser import parse_turn, observation_vector
from cpa.metrics.effectiveness import planning_deviation_score, false_positive_rate
from cpa.orchestrator.stealth_evaluator import StealthEvaluator


def generator_node(env, policy, state: dict) -> dict:
    """Generator: pick the next action and resolve the fake-CTI variant + publish gate.

    Mirrors the head of PoisoningEnv.step: variant = pool[id % len], do_publish from the
    frequency threshold gate. No store mutation here — publishing is the publisher_node's job.
    """
    action = policy(state["obs"])
    variant = env.pool[action.variant_id % len(env.pool)]
    do_publish = action.frequency >= env.cfg.get("publish_threshold", 0.5)
    state.update(action=action, variant=variant, do_publish=do_publish)
    return state


def stealth_evaluator_node(env, evaluator: StealthEvaluator, state: dict) -> dict:
    """StealthEvaluator: detection exposure for this turn, identical to env.step's formula.

    Exposure happens ONLY when we publish (content detectors blended with the channel's base
    prior via max(...)); a no-publish turn exposes nothing new -> full stealth.
    """
    action, variant = state["action"], state["variant"]
    if state["do_publish"]:
        channel_prior = CHANNEL_DETECTION_RISK.get(action.channel_id, 0.3)
        detection_risk = evaluator.detection_risk(variant.text(), channel_prior)
    else:
        detection_risk = 0.0
    state.update(detection_risk=detection_risk, stealth=1.0 - detection_risk)
    return state


def publisher_node(env, state: dict) -> dict:
    """LocalPublisher: insert the variant into the mock store ONLY (sandbox; egress-guarded).

    Faithfully reproduces PoisoningEnv.step's publish side effects, including the CE-AKG
    poison link and the env._published counter that feeds publish_cost.
    """
    if state["do_publish"]:
        action, variant = state["action"], state["variant"]
        env.publisher.publish(variant, channel_id=action.channel_id)
        env._published += 1
        if env.ceakg is not None:
            relevance = {"low": 0.3, "medium": 0.6, "high": 0.9}[variant.target_relevance.value]
            env.ceakg.add_poison(variant.variant_id or 0,
                                 variant.entities.cves + variant.entities.vulnerabilities, relevance)
    return state


def observer_node(env, state: dict) -> dict:
    """Observer: retrieve poisoned context, run the victim turn, parse, score, advance state.

    This block mirrors PoisoningEnv.step bottom half (retrieval with retrieval_score attached,
    victim.step, parse_turn -> observation_vector, PDS/FPR -> impact, reward, build_state).
    We intentionally route through env's own components so it stays numerically consistent.
    Note: the verifier/anti-starvation defense path is exclusive to the RL env (env.step) and is
    NOT replayed here — this explicit graph is the attacker-only curriculum/trajectory runner.
    """
    variant, do_publish = state["variant"], state["do_publish"]

    # Retrieve the (now possibly poisoned) feed, attaching the real cosine retrieval score.
    profile = env.target.get("profile", "")
    if hasattr(env.store, "query_scored"):
        scored = env.store.query_scored(profile, k=5)
    else:
        scored = [(r, None) for r in env.store.query(profile, k=5)]
    cti_context = []
    for r, s in scored:
        try:
            r.retrieval_score = s
        except Exception:
            pass
        cti_context.append(r)

    turn = env.victim.step(cti_context)

    parsed = parse_turn(turn, env.gt)
    vobs = observation_vector(parsed)
    env._history.append(vobs)

    pds = planning_deviation_score(env._baseline_emb, env.embed(turn.plan_text))
    fpr = false_positive_rate(turn.reported_vulns, env.gt)
    impact = 0.6 * pds + 0.4 * fpr
    publish_cost = env._published / max(env.cfg.get("max_turns", 25), 1)

    reward = compute_reward(
        RewardComponents(stealth=state["stealth"], impact=impact,
                         publish_cost=publish_cost, detection_risk=state["detection_risk"]),
        env.cfg.get("reward_weights"),
    )

    if env.ceakg is not None:
        env.ceakg.link_observation(variant.variant_id or 0, parsed.get("cves", []))

    # Advance o_t for the next generator step.
    state["obs"] = build_state(env._ekg(), vobs, env._history)
    state["turn"] += 1
    state["reward"], state["pds"], state["fpr"] = reward, pds, fpr
    state["trajectory"].append({
        "reward": reward, "pds": pds, "fpr": fpr, "stealth": state["stealth"],
    })
    return state
