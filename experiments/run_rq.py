"""RQ runners (Section 3.1) — run the episode design and report statistics.

RQ1: context-aware (Group B) vs random (Group A) CTI -> PDS, FPR, ASR. Paired t-test (B-A),
     pairing each (target, seed) so the two groups see the same conditions.
RQ2: CPA (hybrid RL) vs DREAM baseline on Group B -> ASR + stealth. Paired t-test on stealth,
     ASR delta. Pass --cpa-model to use a trained PPO policy; else the CPA heuristic is used.

    python -m experiments.run_rq --rq 1 --config config/default.yaml --episodes 15
    python -m experiments.run_rq --rq 2 --config config/default.yaml --episodes 15
    python -m experiments.run_rq --rq all --episodes 5 --max-targets 4   # quick smoke

Uses the rule_based victim by default (offline, reproducible). Override with --provider.
"""
from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from typing import List

from experiments.run_episode import load_cfg, run_one

try:
    from scipy import stats as _stats
except Exception:
    _stats = None


def _summ(xs: List[float]) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": round(mean(xs), 4) if xs else 0.0,
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0, "n": len(xs)}


def _paired_t(b: List[float], a: List[float]) -> dict:
    """Paired t-test of (b - a). alpha = 0.05."""
    if _stats is None:
        return {"note": "scipy not installed — install for t-test"}
    if len(b) != len(a) or len(b) < 2:
        return {"note": "need >=2 paired samples"}
    t, p = _stats.ttest_rel(b, a)
    return {"t": round(float(t), 4), "p": round(float(p), 6), "significant_0.05": bool(p < 0.05)}


def _two_way_anova(rows: List[dict], fa: str, fb: str, dv: str) -> dict:
    """Balanced two-way ANOVA with replication, computed by hand (no statsmodels).

    fa, fb = factor keys in each row (e.g. 'group', 'policy'); dv = dependent-variable key.
    Reports F, p (via scipy.stats.f) and eta^2 = SS_factor / SS_total (share of variance
    explained) for each factor and their interaction. eta^2 answers RQ3: which lever matters most.
    """
    la = sorted({r[fa] for r in rows}); lb = sorted({r[fb] for r in rows})
    a, b = len(la), len(lb)
    cells = {(x, y): [float(r[dv]) for r in rows if r[fa] == x and r[fb] == y]
             for x in la for y in lb}
    ns = {len(v) for v in cells.values()}
    if len(ns) != 1 or 0 in ns:
        return {"note": f"unbalanced/empty design: cell sizes={ns}"}
    n = ns.pop()
    allv = [v for cell in cells.values() for v in cell]
    grand = mean(allv)
    mean_a = {x: mean([v for y in lb for v in cells[(x, y)]]) for x in la}
    mean_b = {y: mean([v for x in la for v in cells[(x, y)]]) for y in lb}
    cell_mean = {k: mean(v) for k, v in cells.items()}

    ss_a = b * n * sum((mean_a[x] - grand) ** 2 for x in la)
    ss_b = a * n * sum((mean_b[y] - grand) ** 2 for y in lb)
    ss_ab = n * sum((cell_mean[(x, y)] - mean_a[x] - mean_b[y] + grand) ** 2
                    for x in la for y in lb)
    ss_tot = sum((v - grand) ** 2 for v in allv)
    ss_err = ss_tot - ss_a - ss_b - ss_ab
    df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_err = a * b * (n - 1)

    def _F(ss, df):
        if _stats is None or df_err <= 0 or ss_err <= 0:
            return {"note": "scipy missing or zero error variance"}
        ms, ms_err = ss / df, ss_err / df_err
        F = ms / ms_err
        p = float(_stats.f.sf(F, df, df_err))
        return {"F": round(F, 4), "p": round(p, 6), "eta_sq": round(ss / ss_tot, 4),
                "significant_0.05": bool(p < 0.05)}

    return {
        "design": f"{a}x{b}, n={n} per cell, df_error={df_err}",
        "levels": {fa: la, fb: lb},
        f"factor_{fa}": _F(ss_a, df_a),
        f"factor_{fb}": _F(ss_b, df_b),
        "interaction": _F(ss_ab, df_ab),
        "cell_means": {f"{x}|{y}": round(cell_mean[(x, y)], 4) for x in la for y in lb},
    }


def _targets(cfg: dict, limit: int) -> list:
    ts = cfg["victim"]["targets"]
    return ts if limit <= 0 else ts[:limit]


def run_rq1(cfg: dict, episodes: int, turns: int, max_targets: int) -> dict:
    rows = {"A": [], "B": []}
    for tgt in _targets(cfg, max_targets):
        for seed in range(episodes):
            for grp in ("A", "B"):
                rows[grp].append(run_one(cfg, tgt, grp, "dream", turns, seed=seed))
    out = {}
    for metric in ("final_pds", "final_fpr", "poison_adoption_rate"):
        out[metric] = {
            "A": _summ([r[metric] for r in rows["A"]]),
            "B": _summ([r[metric] for r in rows["B"]]),
            "paired_t_B_vs_A": _paired_t([r[metric] for r in rows["B"]],
                                         [r[metric] for r in rows["A"]]),
        }
    out["ASR_%"] = {g: round(100 * mean([r["success"] for r in rows[g]]), 1) for g in ("A", "B")}
    out["n_episodes_per_group"] = len(rows["A"])
    return out


def run_rq2(cfg: dict, episodes: int, turns: int, max_targets: int, cpa_model=None) -> dict:
    rows = {"dream": [], "cpa": []}
    for tgt in _targets(cfg, max_targets):
        for seed in range(episodes):
            rows["dream"].append(run_one(cfg, tgt, "B", "dream", turns, seed=seed))
            rows["cpa"].append(run_one(cfg, tgt, "B", "cpa", turns, seed=seed, cpa_model=cpa_model))
    out = {}
    for metric in ("stealth_score", "max_pds", "max_fpr", "mean_reward", "published"):
        out[metric] = {
            "dream": _summ([r[metric] for r in rows["dream"]]),
            "cpa": _summ([r[metric] for r in rows["cpa"]]),
            "paired_t_cpa_vs_dream": _paired_t([r[metric] for r in rows["cpa"]],
                                               [r[metric] for r in rows["dream"]]),
        }
    out["ASR_%"] = {p: round(100 * mean([r["success"] for r in rows[p]]), 1) for p in ("dream", "cpa")}
    out["n_episodes_per_policy"] = len(rows["dream"])
    return out


def run_rq3(cfg: dict, episodes: int, turns: int, max_targets: int, cpa_model=None) -> dict:
    """RQ3 — factor analysis. Two-way ANOVA of CTI-context (group A/B) x policy (dream/cpa)
    on attack impact (max_pds) and stealth, with (target, seed) as replicates. Tells us which
    lever drives the outcome and whether the two interact."""
    rows = []
    for tgt in _targets(cfg, max_targets):
        for seed in range(episodes):
            for grp in ("A", "B"):
                for pol in ("dream", "cpa"):
                    r = run_one(cfg, tgt, grp, pol, turns, seed=seed,
                                cpa_model=cpa_model if pol == "cpa" else None)
                    rows.append(r)
    out = {"n_episodes": len(rows)}
    for dv in ("max_pds", "stealth_score"):
        out[f"anova_{dv}"] = _two_way_anova(rows, "group", "policy", dv)
    return out


def run_defense_ablation(cfg: dict, episodes: int, turns: int, max_targets: int) -> dict:
    """Defense ablation — re-run the strongest attack (Group B, DREAM always-publish) with the
    CTIProvenanceVerifier OFF vs ON, paired by (target, seed). Reports ASR reduction, the impact
    drop, and the filter's precision/recall on poison vs real CTI (does it block poison without
    starving the victim of genuine feed?)."""
    rows = {"off": [], "on": []}
    for tgt in _targets(cfg, max_targets):
        for seed in range(episodes):
            rows["off"].append(run_one(cfg, tgt, "B", "dream", turns, seed=seed, defense=False))
            rows["on"].append(run_one(cfg, tgt, "B", "dream", turns, seed=seed, defense=True))
    out = {}
    for metric in ("max_pds", "max_fpr", "stealth_score"):
        out[metric] = {
            "defense_off": _summ([r[metric] for r in rows["off"]]),
            "defense_on": _summ([r[metric] for r in rows["on"]]),
            "paired_t_off_vs_on": _paired_t([r[metric] for r in rows["off"]],
                                            [r[metric] for r in rows["on"]]),
        }
    out["ASR_%"] = {"defense_off": round(100 * mean([r["success"] for r in rows["off"]]), 1),
                    "defense_on": round(100 * mean([r["success"] for r in rows["on"]]), 1)}
    # Aggregate the feed-filter confusion matrix across all defense-ON episodes.
    agg = {"poison_blocked": 0, "poison_passed": 0, "real_blocked": 0, "real_passed": 0}
    for r in rows["on"]:
        for k, v in r.get("filter_stats", {}).items():
            agg[k] += v
    tp, fn = agg["poison_blocked"], agg["poison_passed"]      # poison = positive class to block
    fp, tn = agg["real_blocked"], agg["real_passed"]
    rc = [r["real_corpus_retention"] for r in rows["on"] if "real_corpus_retention" in r]
    out["feed_filter"] = {
        "confusion": agg,
        "poison_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,     # % poison caught
        "poison_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,  # of blocked, % poison
        "real_retention_in_feed": round(tn / (tn + fp), 4) if (tn + fp) else None,
        # Anti-starvation check: real records guaranteed into the feed now reach the verifier,
        # so this is > 0 (was always 0 before, which made real_retention_in_feed null).
        "real_in_feed_total": tn + fp,

        # Corpus-level: avg fraction of genuine CTI the verifier keeps (feed-starvation check).
        "real_corpus_retention": round(mean(rc), 4) if rc else None,
    }
    out["n_episodes_per_arm"] = len(rows["off"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", choices=["1", "2", "3", "defense", "all"], required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--episodes", type=int, default=15, help="episodes (seeds) per target")
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--max-targets", type=int, default=0, help="0 = all targets")
    ap.add_argument("--provider", default="rule_based")
    ap.add_argument("--cpa-model", default=None, help="path to a trained SB3 PPO model (RQ2)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["victim"]["provider"] = args.provider

    model = None
    if args.cpa_model:
        from stable_baselines3 import PPO
        model = PPO.load(args.cpa_model)

    report = {}
    if args.rq in ("1", "all"):
        report["RQ1"] = run_rq1(cfg, args.episodes, args.turns, args.max_targets)
    if args.rq in ("2", "all"):
        report["RQ2"] = run_rq2(cfg, args.episodes, args.turns, args.max_targets, cpa_model=model)
    if args.rq in ("3", "all"):
        report["RQ3"] = run_rq3(cfg, args.episodes, args.turns, args.max_targets, cpa_model=model)
    if args.rq in ("defense", "all"):
        report["defense_ablation"] = run_defense_ablation(cfg, args.episodes, args.turns,
                                                          args.max_targets)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
