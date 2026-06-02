# Implementation Plan

Scaffold is in place. Order of work to reach publishable RQ1–RQ3 results, all in-sandbox.

## Phase 0 — run the skeleton (done by scaffold)
- [x] Metrics (PDS/FPR/CFR/ASR/UR/DS/ATMI) with formulas — `cpa/metrics/`
- [x] PO-MDP (state/action/reward/env) — `cpa/mdp/`
- [x] Mock CTI store + LocalPublisher + egress guard — sandbox boundary enforced
- [x] LLM victim (Anthropic provider) — `cpa/victim/llm_victim.py`
- [x] Zero-dep fallbacks so `run_episode` runs before heavy models — `experiments/backends.py`
- [x] Tests for metrics + guard — `tests/test_metrics.py`

## Phase 1 — make the victim loop faithful (RQ1)  ✅
- [x] `RuleBasedVictim` for fast/free offline runs (config `provider: rule_based`).
- [x] Per-target `ground_truth_vulns` in config (8 financial targets) so FPR is meaningful.
- [x] ChromaDB + MiniLM retrieval (`ChromaCTIStore`, ported from DREAM) — replaces overlap.
- [x] Group-aware generator (A=random/low-relevance, B=context-aware/high).
- [x] RQ1 runner: Group A vs B + paired t-test (`experiments/run_rq.py --rq 1`). See RESULTS.md.
- [ ] `llm_local` generator backend: Llama-3-8B QLoRA on GFCTI-Finance (optional upgrade).

## Phase 2 — RL (RQ2)  ✅
- [x] `PoisoningEnv` wrapped as `gymnasium.Env` (`cpa/rl/gym_env.py`), passes SB3 check_env.
- [x] CE-AKG embedding feeds `e_kg`; anomaly tracked (diagnostic).
- [x] `DreamBaselinePolicy` (C-GPS greedy) + `CPAPolicy` (heuristic + PPO model slot).
- [x] PPO trainer (`cpa/rl/train.py`); RQ2 runner compares CPA vs DREAM on ASR + stealth.
- [ ] Offline pretrain on DREAM-baseline trajectories → online curriculum + LoRA on Llama-3-8B
      (the full proposal pipeline layered on the working PPO core).

## Phase 3 — defense (Section 2.6, the constructive contribution)
- [ ] Real GLTR + RoBERTa detectors in `experiments/backends.make_detector_fn`.
- [ ] Defense ablation: re-run episodes with `CTIProvenanceVerifier` filtering the feed;
      report ASR reduction. This is the headline defensive result.

## Phase 4 — reporting (RQ3)
- [ ] `run_rq.py`: aggregate 120 episodes, ANOVA over (relevance, stealth, impact, PDS).
- [ ] Mean ± std tables, significance at α=0.05.

## Guardrails to keep (do not remove)
- Egress guard installed at every entry point.
- `allow_real_publish` stays false; no real-platform connectors.
- No offensive/exploitation modules (Scenario 2 out of scope).
