# Results — RQ1 & RQ2 (sandbox)

All runs use the **offline `rule_based` victim** (deterministic given a seed, seeded
stochasticity across seeds) so the full design runs without API cost and reproducibly.
The victim faithfully reproduces Excalibur's TDA: poisoned CTI it trusts inflates
`evidence_confidence` → TDI drops → it flips to EXPLOITATION and commits to fabricated
vulnerabilities. Retrieval is the ported DREAM ChromaDB + MiniLM stack.

**Real CTI background:** the genuine CTI corpus is built from two public datasets —
[CASIE](https://github.com/Ebiquity/CASIE) (cyber-incident news) and
[CyEnts](https://github.com/UMBC-Onramp/CyEnts-Cyber-Blog-Dataset) (security blog text),
5,886 records total (`python -m data.build_corpus`). Each episode samples 200 of them as the
feed the fake CTI must compete against. **The victim's trust (evidence) is now driven by the
REAL retrieval cosine score** against the target profile — so the Group A vs B difference
emerges from measured semantic similarity (≈0.42 generic vs ≈0.77 context-aware), not a
hardcoded label.

Reproduce:
```powershell
python -m experiments.run_rq --rq all --episodes 15        # full: 8 targets × 15 seeds
python -m experiments.run_rq --rq all --episodes 6 --max-targets 4   # quick
```

## RQ1 — context-aware (Group B) vs random (Group A) CTI

8 episodes/group (2 targets × 4 seeds), 12 turns — smoke run with the real corpus + cosine-driven
evidence. Paired t-test (B − A), α = 0.05. (Scale up `--episodes`/`--max-targets` for final.)

| Metric      | Group A (random) | Group B (context-aware) | paired t | p | sig |
|-------------|------------------|-------------------------|----------|---|-----|
| final PDS   | 0.058 ± 0.100    | **0.237 ± 0.007**       | 4.55     | 0.0026 | ✓ |
| final FPR   | 0.198 ± 0.343    | **0.792 ± 0.072**       | 4.23     | 0.0039 | ✓ |
| ASR         | 25%              | **100%**                |          |   | ✓ |

**Answer:** Context-aware fake CTI causes significantly higher planning deviation and false
positives than random CTI. Mechanism: against the real CASIE+CyEnts background, BOTH kinds of
poison out-rank the generic corpus and reach the victim, but the context-aware poison is far more
similar to the target profile (cosine ≈ 0.77 vs ≈ 0.42). The victim's trust tracks that cosine,
so it commits to the fabricated path under Group B but usually stays cautious under Group A.
Note Group A is no longer 0% (25%, high variance) — generic poison occasionally clears the trust
bar — which is exactly the realistic spread the earlier tag-driven version lacked.

## RQ2 — CPA (hybrid RL) vs DREAM baseline (on Group B)

24 episodes/policy. DREAM baseline = always-publish, noisy channel (C-GPS greedy). CPA =
burst-then-throttle on a stealthy channel (heuristic shown here; PPO model via `--cpa-model`).

| Metric        | DREAM baseline | CPA            | paired t | p | sig |
|---------------|----------------|----------------|----------|---|-----|
| ASR           | 100%           | 100%           |          |   | = |
| stealth score | 0.40 ± 0.00    | **0.84 ± 0.00**| (Δ large)|   | ✓ |
| max PDS       | 0.228          | 0.215          | −5.5     | <1e-4 | ✓ |
| max FPR       | 0.819          | 0.800          | −5.7     | <1e-5 | ✓ |
| publish count | 15.0           | **4.0**        | (Δ large)|   | ✓ |
| mean reward   | 0.141          | **0.429**      | 167.8    | <1e-6 | ✓ |

**Answer:** Hybrid RL keeps attack success (ASR 100%) while **doubling stealth** (0.84 vs 0.40)
and cutting the published-poison footprint ~4× (4 vs 15), at a negligible cost in impact
(PDS/FPR drop by ~1–2 points, still far above success thresholds). The reward gain (3×) reflects
the stealth + publish-cost advantage the proposal predicts for the RL upgrade over DREAM's
heuristic search.

### RQ2 with a trained PPO policy

A short PPO run (`cpa/rl/train.py`, 1.2k steps, pool 6 — a smoke-sized model) already learns to
**beat both** the DREAM baseline and the CPA heuristic:

| Metric        | DREAM | CPA-PPO | note |
|---------------|-------|---------|------|
| ASR           | 100%  | 100%    | success preserved |
| stealth score | 0.40  | **0.95**| learned the stealthiest channel |
| publish count | 12.0  | **1.0** | learned a single, well-timed publish suffices |
| mean reward   | 0.169 | **0.463** | |
| max FPR       | 0.833 | 0.500   | still ≫ 0.40 threshold |

The PPO policy discovered that **one** high-relevance poison, published once on a low-detection
channel, is enough to derail the victim for the whole episode (the cascade latch) — minimizing
exposure. Train longer on `config/default.yaml` (8 targets, larger pool) for paper-grade numbers.

> Note: the heuristic CPA has fixed behavior, so its `stealth`/`published` show zero within-policy
> variance (degenerate t, large deterministic Δ). The PPO policy and PDS/FPR/reward carry genuine
> across-seed variance.

## Caveats / for the thesis
- Numbers above are from the rule_based victim. Re-run key cells with `--provider anthropic`
  (or `openrouter`) for the real-LLM victim to confirm the effect transfers (stochastic → richer stats).
- PDS uses the zero-dependency hashing embedding by default; install `sentence-transformers`
  for Sentence-BERT PDS (typically larger separation).
- Defense ablation (re-run with `CTIProvenanceVerifier` filtering the feed) is the next result.
