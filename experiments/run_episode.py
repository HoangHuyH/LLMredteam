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
from cpa.metrics.effectiveness import attack_success
from cpa.metrics.stealth import undetected_rate, detection_score
from cpa.metrics.efficiency import first_major_impact_turn
from experiments.backends import make_embed_fn, make_detector_fn


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(cfg: dict, target: dict, group: str, policy_name: str,
            turns: int, seed: int = 0, log_dir: Optional[str] = None, cpa_model=None,
            defense: bool = False) -> dict:
    """Run one episode; return a metrics dict (no printing).

    defense=True installs a CTIProvenanceVerifier that filters the retrieved feed before the
    victim reads it (the defense ablation): poison below the trust threshold is dropped.
    """
    install_guard()
    assert_no_real_publish(cfg.get("sandbox", {}).get("allow_real_publish", False))

    ground_truth_ids = {v["id"] for v in target.get("ground_truth_vulns", [])}

    # Isolated in-memory store per episode so poison never leaks across runs (RQ sweeps).
    store_cfg = {**cfg["cti_store"], "in_memory": True,
                 "collection_name": f"cti_{target['id']}_{group}_{policy_name}_{seed}"}
    store = make_store(store_cfg)
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

    env_cfg = {"reward_weights": cfg["mdp"]["reward_weights"], "ekg_dim": 8, "max_turns": turns}
    env = PoisoningEnv(victim, store, publisher, pool, target,
                       ground_truth_ids, embed_fn, detector_fn, env_cfg,
                       ceakg=ceakg, verifier=verifier)

    policy = make_policy(policy_name, pool_size=len(pool), seed=seed, model=cpa_model)

    state = env.reset()
    pds_seq, fpr_seq, rewards, stealth_seq = [], [], [], []
    for _ in range(turns):
        state, reward, info = env.step(policy(state))
        pds_seq.append(info["pds"]); fpr_seq.append(info["fpr"])
        rewards.append(reward); stealth_seq.append(info["stealth"])

    # Stealth (UR/DS) over the poison actually published this episode.
    poison_texts = [r.text() for r in store.poison_records()]
    per_item = [detector_fn(t) for t in poison_texts]
    ur = undetected_rate(per_item) if per_item else 100.0
    ds = detection_score([p for probs in per_item for p in probs]) if per_item else 0.0

    final_pds, final_fpr = pds_seq[-1], fpr_seq[-1]
    result = {
        "target": target["id"], "group": group, "policy": policy_name, "seed": seed,
        "defense": defense,
        "final_pds": final_pds, "final_fpr": final_fpr,
        "max_pds": max(pds_seq), "max_fpr": max(fpr_seq),
        "success": bool(attack_success(max(pds_seq), max(fpr_seq),
                                       pds_thr=cfg["experiment"]["success_thresholds"]["pds"],
                                       fpr_thr=cfg["experiment"]["success_thresholds"]["fpr"])),
        "atmi": first_major_impact_turn(pds_seq, fpr_seq),
        "undetected_rate": ur, "detection_score": ds,
        "stealth_score": float(np.mean(stealth_seq)),
        "mean_reward": float(np.mean(rewards)), "published": publisher.published_count,
    }
    if defense:
        result["filter_stats"] = dict(env.filter_stats)
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
