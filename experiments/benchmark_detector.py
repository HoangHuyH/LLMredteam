"""Benchmark the AI-text detector: ROC AUC of REAL CTI vs FAKE CTI (sandbox red-team metric).

Loads genuine CTI from data/real_cti_corpus.jsonl (label 0 = real) and generates fake CTI with the
template FakeCTIGenerator across the config's targets (label 1 = fake), scores every record with
the selected detector backend, and reports ROC AUC + accuracy@0.5. A near-0.5 AUC means the
detector cannot tell poison from genuine CTI -- which is exactly what the keyword heuristic does,
and why the stealth / undetected-rate / detection-score metrics need a real detector.

AUC is computed MANUALLY via the rank / Mann-Whitney U identity (sklearn is not installed).

    python -m experiments.benchmark_detector --backend ensemble --n 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import yaml

from cpa.generator import FakeCTIGenerator
from experiments.backends import make_detector_fn

OUT = Path("experiments/detector_benchmark.json")


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_real_texts(jsonl_path: str, n: int, seed: int) -> List[str]:
    """Sample up to n genuine CTI texts from the corpus (seeded)."""
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"real CTI corpus not found: {jsonl_path}")
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if n < len(lines):
        lines = random.Random(seed).sample(lines, n)
    texts: List[str] = []
    for line in lines:
        d = json.loads(line)
        text = d.get("real_cti") or d.get("text") or ""
        if text:
            texts.append(text)
    return texts


def build_fake_texts(cfg: dict, n: int, generator: str | None = None) -> List[str]:
    """Generate fake CTI across all configured targets (Group B = context-aware), capped at n.

    `generator` overrides cfg["generator"]["backend"] ("template" | "llm_local") so the same
    benchmark can score either the deterministic template poison or the LLM-generated poison.
    For the slow llm_local backend we shrink the per-target pool to exactly the remaining need
    so we never generate more LLM samples than the cap requires.
    """
    gen_cfg = dict(cfg["generator"])
    if generator:
        gen_cfg["backend"] = generator
    gen = FakeCTIGenerator(gen_cfg)
    targets = cfg["victim"]["targets"]
    texts: List[str] = []
    for target in targets:
        gen.pool_size = min(gen.pool_size, n - len(texts))   # don't over-generate (matters for LLM)
        for rec in gen.build_pool(target, group="B"):
            texts.append(rec.text())
            if len(texts) >= n:
                return texts
    return texts


def detector_prob(detector_fn, text: str) -> float:
    """Mirror the caller convention: max(probs) is the AI-generated/detection probability."""
    probs = detector_fn(text)
    return max(probs) if probs else 0.0


def roc_auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC via the Mann-Whitney U / rank-sum identity (label 1 = positive = fake).

    AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg), with tied scores
    receiving averaged ranks. Returns 0.5 if either class is empty.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0          # 1-based, averaged over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def accuracy_at(scores: List[float], labels: List[int], thr: float = 0.5) -> float:
    correct = sum(1 for s, y in zip(scores, labels) if int(s >= thr) == y)
    return correct / len(labels) if labels else 0.0


def run(cfg: dict, backend: str, n: int, seed: int, generator: str | None = None) -> dict:
    det_cfg = dict(cfg.get("backends", {}))
    det_cfg["detector_backend"] = backend
    detector_fn = make_detector_fn(det_cfg)

    real_texts = load_real_texts(cfg["cti_store"].get("seed_real_cti", "./data/real_cti_corpus.jsonl"),
                                 n, seed)
    fake_texts = build_fake_texts(cfg, n, generator)

    scores: List[float] = []
    labels: List[int] = []
    for t in real_texts:
        scores.append(detector_prob(detector_fn, t)); labels.append(0)
    for t in fake_texts:
        scores.append(detector_prob(detector_fn, t)); labels.append(1)

    auc = roc_auc(scores, labels)
    acc = accuracy_at(scores, labels, 0.5)
    real_scores = [s for s, y in zip(scores, labels) if y == 0]
    fake_scores = [s for s, y in zip(scores, labels) if y == 1]
    return {
        "backend": backend,
        "generator": generator or cfg["generator"].get("backend", "template"),
        "n_real": len(real_texts),
        "n_fake": len(fake_texts),
        "auc": round(auc, 4),
        "accuracy_at_0.5": round(acc, 4),
        "mean_prob_real": round(sum(real_scores) / len(real_scores), 4) if real_scores else None,
        "mean_prob_fake": round(sum(fake_scores) / len(fake_scores), 4) if fake_scores else None,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--backend", default="heuristic",
                    choices=["heuristic", "roberta", "gltr", "ensemble"])
    ap.add_argument("--n", type=int, default=200, help="max real records sampled (and fake cap)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generator", default=None, choices=["template", "llm_local"],
                    help="override generator backend for the fake CTI (default: config value)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    result = run(cfg, args.backend, args.n, args.seed, args.generator)

    # Keep template vs llm_local results side by side when --generator is given (avoids overwrite).
    out = OUT if not args.generator else OUT.with_name(f"detector_benchmark_{args.generator}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\n[benchmark] gen={result['generator']} AUC={result['auc']} "
          f"acc@0.5={result['accuracy_at_0.5']} "
          f"(real={result['n_real']}, fake={result['n_fake']}) -> {out}")


if __name__ == "__main__":
    main()
