"""Run sandbox poisoning episodes.

`run_one(...)` is the programmatic core used by both this CLI and experiments/run_rq.py.
Group A uses random/generic CTI; Group B uses context-aware CTI (RQ1). The acting policy
(random / dream-baseline / cpa) is the RQ2 dimension.

    python -m experiments.run_episode --config config/local.yaml --group B --turns 20
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
import yaml

from cpa.cti_store import make_store
from cpa.generator import FakeCTIGenerator
from cpa.victim import make_victim
from cpa.ce_akg import CEAKG
from cpa.orchestrator import LocalPublisher
from cpa.orchestrator.egress_guard import install_guard, assert_no_real_publish
from cpa.mdp.env import PoisoningEnv
from cpa.rl.policy import make_policy
from cpa.metrics.effectiveness import attack_success, cascade_failure_rate, SELF_SABOTAGE_ACTIONS
from cpa.observer.parser import CVE_RE
from cpa.metrics.stealth import undetected_rate, detection_score
from cpa.metrics.efficiency import first_major_impact_turn
from experiments.backends import make_embed_fn, make_detector_fn, make_store_embed_fn


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Built once and reused across episodes (loading SentenceTransformer per episode would dominate
# runtime). Enabled by config cti_store.gpu_embed; used to embed the store on a GPU when present.
_STORE_EMBED_FN = None
_STORE_EMBED_INIT = False


def _store_embed_fn(cfg: dict):
    global _STORE_EMBED_FN, _STORE_EMBED_INIT
    if not _STORE_EMBED_INIT:
        _STORE_EMBED_FN = make_store_embed_fn(cfg.get("backends", {}))
        _STORE_EMBED_INIT = True
    return _STORE_EMBED_FN


def run_one(cfg: dict, target: dict, group: str, policy_name: str,
            turns: int, seed: int = 0, log_dir: Optional[str] = None, cpa_model=None,
            defense: bool = False) -> dict:
    """Run one episode; return a metrics dict (no printing).

    defense=True installs a CTIProvenanceVerifier that filters the retrieved feed before the
    victim reads it (the defense ablation): poison below the trust threshold is dropped.
    """
    install_guard()
    assert_no_real_publish(cfg.get("sandbox", {}).get("allow_real_publish", False))

    # Ground truth keyed by CVE: the victim reports vulns by CVE, so FPR / fake-vuln detection must
    # compare against the target's real CVEs, not its internal GT-* record ids.
    ground_truth_ids = {v["cve"] for v in target.get("ground_truth_vulns", []) if v.get("cve")}

    # Isolated in-memory store per episode so poison never leaks across runs (RQ sweeps).
    store_cfg = {**cfg["cti_store"], "in_memory": True,
                 "collection_name": f"cti_{target['id']}_{group}_{policy_name}_{seed}"}
    store_embed = _store_embed_fn(cfg) if cfg["cti_store"].get("gpu_embed") else None
    store = make_store(store_cfg, embed_fn=store_embed)
    store.seed_real(cfg["cti_store"].get("seed_real_cti", ""),
                    sample_size=cfg["cti_store"].get("seed_sample_size"), seed=seed)

    pool = FakeCTIGenerator(cfg["generator"]).build_pool(target, group=group)
    publisher = LocalPublisher(store)

    victim_cfg = dict(cfg["victim"]); victim_cfg["max_turns"] = turns; victim_cfg["seed"] = seed
    victim = make_victim(victim_cfg)

    ceakg = CEAKG()
    techs = [t.strip() for t in target.get("profile", "").replace(",", " ").split() if len(t) > 3]
    ceakg.add_target(target["id"], techs)

    embed_fn = make_embed_fn(cfg.get("backends", {}))
    detector_fn = make_detector_fn(cfg.get("backends", {}))

    verifier = None
    if defense:
        from cpa.defense.verifier import CTIProvenanceVerifier
        dcfg = cfg.get("defense", {})
        verifier = CTIProvenanceVerifier(
            corroboration_k=dcfg.get("corroboration_k", 2),
            trust_threshold=dcfg.get("trust_threshold", 0.5),
            detector_fn=detector_fn)

    env_cfg = {"reward_weights": cfg["mdp"]["reward_weights"], "ekg_dim": 8, "max_turns": turns,
               "heat_gain": cfg.get("heat_gain", 0.25), "heat_decay": cfg.get("heat_decay", 0.6)}
    env = PoisoningEnv(victim, store, publisher, pool, target,
                       ground_truth_ids, embed_fn, detector_fn, env_cfg,
                       ceakg=ceakg, verifier=verifier)

    policy = make_policy(policy_name, pool_size=len(pool), seed=seed, model=cpa_model)

    # Universe of attacker-injected fake CVEs; "adoption" = the victim echoing any of these in its
    # output (the TARGETED impact signal, vs PDS's undirected drift).
    poison_cves = {c.upper() for r in pool for c in (r.entities.cves or [])}

    state = env.reset()
    pds_seq, fpr_seq, rewards, stealth_seq = [], [], [], []
    adopt_seq = []    # per-turn: did the victim echo an injected fake CVE?
    all_actions = []  # victim actions across all turns, for CFR (self-sabotage) below
    for _ in range(turns):
        state, reward, info = env.step(policy(state))
        pds_seq.append(info["pds"]); fpr_seq.append(info["fpr"])
        rewards.append(reward); stealth_seq.append(info["stealth"])
        victim_cves = {m.upper() for m in CVE_RE.findall(getattr(info["turn"], "raw_stdout", "") or "")}
        adopt_seq.append(bool(victim_cves & poison_cves))
        all_actions.extend(getattr(info["turn"], "actions", []) or [])

    # Stealth (UR/DS) over the poison actually published this episode.
    poison_texts = [r.text() for r in store.poison_records()]
    per_item = [detector_fn(t) for t in poison_texts]
    ur = undetected_rate(per_item) if per_item else 100.0
    ds = detection_score([p for probs in per_item for p in probs]) if per_item else 0.0

    # Cascade Failure Rate (self-sabotage): % of victim actions that are self-harming.
    cfr = cascade_failure_rate(all_actions)
    n_self_sabotage = sum(1 for a in all_actions if a.get("type") in SELF_SABOTAGE_ACTIONS)

    # Mean target-relevance of the poison actually published (RQ3 factor): low/medium/high -> 0.3/0.6/0.9.
    _rel_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    pub_poison = store.poison_records()
    mean_relevance = (float(np.mean([_rel_map.get(getattr(r.target_relevance, "value", "low"), 0.3)
                                     for r in pub_poison])) if pub_poison else 0.0)

    final_pds, final_fpr = pds_seq[-1], fpr_seq[-1]
    poison_adopted = any(adopt_seq)
    result = {
        "target": target["id"], "group": group, "policy": policy_name, "seed": seed,
        "defense": defense,
        "final_pds": final_pds, "final_fpr": final_fpr,
        "max_pds": max(pds_seq), "max_fpr": max(fpr_seq),
        "poison_adopted": poison_adopted,
        "poison_adoption_rate": round(float(np.mean(adopt_seq)) if adopt_seq else 0.0, 4),
        "success": bool(attack_success(max(pds_seq), max(fpr_seq),
                                       pds_thr=cfg["experiment"]["success_thresholds"]["pds"],
                                       fpr_thr=cfg["experiment"]["success_thresholds"]["fpr"])),
        "mean_relevance": round(mean_relevance, 4),     # RQ3 factor: target relevance of published poison
        "atmi": first_major_impact_turn(pds_seq, fpr_seq),
        "undetected_rate": ur, "detection_score": ds,
        "stealth_score": float(np.mean(stealth_seq)),
        "mean_reward": float(np.mean(rewards)), "published": publisher.published_count,
        "cfr": round(cfr, 4),                    # percent of self-sabotage actions across the episode
        "n_self_sabotage": n_self_sabotage,      # raw count, for aggregation
        "n_actions_total": len(all_actions),
    }
    if defense:
        result["filter_stats"] = dict(env.filter_stats)
        # Corpus-level sanity check: would the verifier keep GENUINE CTI? Under Group B the
        # top-k feed is dominated by poison, so real records rarely reach it; verify all real
        # records in the store directly to confirm the filter is discriminating, not feed-starving.
        corpus = store.all()
        real_recs = [r for r in corpus if not r.is_poison]
        if real_recs:
            kept = sum(1 for r in real_recs
                       if verifier.verify(r, corpus).trust >= verifier.threshold)
            result["real_corpus_retention"] = round(kept / len(real_recs), 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", choices=["A", "B"], default="B")
    ap.add_argument("--policy", choices=["random", "dream", "cpa"], default="random")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--target-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    target = cfg["victim"]["targets"][args.target_idx]
    result = run_one(cfg, target, args.group, args.policy, args.turns, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
