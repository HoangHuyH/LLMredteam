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
python -m experiments.run_rq --rq all --episodes 15        # RQ1+RQ2+RQ3+defense, 8 targets × 15 seeds
python -m experiments.run_rq --rq 3 --episodes 5           # RQ3 factor analysis (two-way ANOVA)
python -m experiments.run_rq --rq defense --episodes 5     # defense ablation (verifier on/off)
python -m experiments.run_rq --rq all --episodes 6 --max-targets 4   # quick smoke
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

## RQ3 — factor analysis (which lever drives the attack?)

Two-way ANOVA of **CTI-context** (Group A random / B context-aware) × **policy** (DREAM / CPA)
on attack impact (`max_pds`), with (target, seed) as replicates. ANOVA computed by hand in
`run_rq._two_way_anova` (balanced design); `eta²` = share of total variance explained.
Smoke: 2 targets × 6 seeds = 24 episodes (2×2, n=6/cell). (Scale `--episodes` for final.)

| Factor (DV = max_pds)        | F      | p       | η² (variance explained) | sig |
|------------------------------|--------|---------|-------------------------|-----|
| **CTI-context** (A vs B)     | 15.71  | 0.0008  | **0.342**               | ✓ |
| **policy** (DREAM vs CPA)    | 9.49   | 0.0059  | 0.207                   | ✓ |
| interaction (context×policy) | 0.68   | 0.418   | 0.015                   | ✗ |

Cell means (max_pds): A|cpa 0.000, A|dream 0.077, B|cpa 0.107, **B|dream 0.241**.

**Answer:** **Context-awareness of the fake CTI is the dominant factor** (η²≈0.34) — about
1.65× the variance explained by the attacker's publishing policy (η²≈0.21). The two factors are
**additive, not interacting** (interaction p=0.42): a smarter policy and more relevant CTI each
help independently. This directly supports the thesis premise that *what* you poison with
(target relevance) matters more than *how* you deliver it.

> `stealth_score` ANOVA degenerates (zero error variance): stealth is fully determined by the
> policy (CPA 0.80 vs DREAM 0.40) and is unaffected by CTI-context — so the design has no
> within-cell variance to test. Reported via cell means instead. This is itself a clean finding:
> CTI-context drives *impact*, policy drives *stealth*; the two levers are orthogonal.

## Defense ablation — CTIProvenanceVerifier on/off

Re-run the strongest attack (Group B, DREAM always-publish) with the provenance verifier
(`cpa/defense/verifier.py`) **OFF vs ON**, paired by (target, seed). The verifier scores each
retrieved record by a weighted sum of **source reputation** + **AI-text likelihood** +
**corroboration**, and drops records below `trust_threshold=0.5` before the victim reads them.
Smoke: 2 targets × 6 seeds = 12 episodes/arm.

| Metric  | Defense OFF        | Defense ON   | paired t | p | sig |
|---------|--------------------|--------------|----------|---|-----|
| max PDS | 0.241 ± 0.00       | **0.000**    | 181.8    | <1e-6 | ✓ |
| max FPR | 0.833 ± 0.00       | **0.000**    | (Δ large)|   | ✓ |
| **ASR** | **100%**           | **0%**       |          |   | ✓ |

**Feed filter quality (defense ON):** poison **recall 1.0**, poison **precision 1.0**
(162 poison records seen, all blocked, zero real records wrongly blocked), and
**real-corpus retention 1.0** — the verifier keeps 100% of the genuine CASIE+CyEnts CTI while
removing 100% of the poison.

**Answer:** A lightweight provenance filter **neutralizes the attack entirely** (ASR 100→0) in
this sandbox *without* starving the agent of legitimate CTI. The discrimination is honest, not a
blanket block: it comes from two signals that genuinely differ between real and fake here —
low-reputation injection channel (poison `mock-local` 0.4 vs real `seed-real` 0.9) and a
boilerplate AI-text detector the template poison trips. **Caveat:** the template poison is an
easy target; the result is an *upper bound* on this defense. A generator that mimics real source
reputation and evades the AI-text detector (the natural next adversary, and the Defense /
Threat-model-2 direction) would push recall back down — that arms race is the point.

> Why a weighted SUM, not the original product trust formula: our real corpus ships no CVE
> annotations, so every record is "uncorroborated"; a hard corroboration gate (product) would
> drop genuine CTI alongside poison and wipe the feed. The sum lets reputation + AI-likelihood
> carry the decision, with corroboration as a minor bonus that would matter more on an annotated
> corpus.

## Caveats / for the thesis
- Numbers above are from the rule_based victim. Re-run key cells with `--provider anthropic`
  (or `openrouter`) for the real-LLM victim to confirm the effect transfers (stochastic → richer stats).
- PDS uses the zero-dependency hashing embedding by default; install `sentence-transformers`
  for Sentence-BERT PDS (typically larger separation).
- RQ3/defense tables above are smoke-sized (2 targets); the `--episodes 5` full 8-target run is
  the paper-grade version (same command, no `--max-targets`).
