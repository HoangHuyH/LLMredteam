"""PPO training for the CPA policy (Section 2.8), Stable-Baselines3.

Trains on the offline rule_based victim (no API, fast). The reward (Eq. 1) balances stealth,
impact, publish-cost, and detection-risk, so PPO learns the burst-then-throttle, stealthy-channel
behavior that beats the always-publish DREAM baseline on stealth while keeping ASR (RQ2).

    python -m cpa.rl.train --config config/default.yaml --timesteps 20000 --out models/cpa_ppo

Then evaluate:
    python -m experiments.run_rq --rq 2 --cpa-model models/cpa_ppo.zip

The full proposal pipeline (offline pretrain on DREAM-baseline trajectories -> online curriculum
random/context_aware/multi_turn, LoRA on Llama-3-8B) layers on top of this PPO core.
"""
from __future__ import annotations

import argparse

import yaml


def _ppo_kwargs(cfg: dict, turns: int, device: str) -> dict:
    """Shared PPO hyperparameters (Eq. 1 / Section 2.8). n_steps/batch_size scale with `turns`."""
    return dict(
        gamma=cfg["mdp"].get("gamma", 0.95),
        n_steps=turns * 8, batch_size=turns * 2, n_epochs=5,
        learning_rate=3e-4, verbose=1, seed=cfg["experiment"].get("seed", 0),
        device=device,   # "cpu" is fine (and often faster) for this tiny MlpPolicy
    )


def train_ppo(config_path: str, timesteps: int, out: str, group: str = "B", turns: int = 15,
              device: str = "auto"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from cpa.rl.gym_env import PoisoningGymEnv

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = Monitor(PoisoningGymEnv(cfg, group=group, turns=turns, seed=cfg["experiment"].get("seed", 0)))
    model = PPO("MlpPolicy", env, **_ppo_kwargs(cfg, turns, device))
    model.learn(total_timesteps=timesteps)
    model.save(out)
    print(f"saved PPO policy -> {out}.zip")
    return out


def train_ppo_curriculum(config_path: str, total_timesteps: int, out: str, stages=None,
                         turns: int = 15, device: str = "auto", multi_turn_turns: int = None):
    """3-level curriculum (Section 2.7) with policy transfer across stages.

        Stage 1 "random"        -> Group A (generic / low-relevance poison)
        Stage 2 "context_aware" -> Group B (target-tailored poison)
        Stage 3 "multi_turn"    -> Group B, longer sustained campaigns (more turns)

    A single PPO policy is trained once on the stage-1 env and carried forward (set_env +
    learn(reset_num_timesteps=False)) — never re-initialized — so each stage continues learning
    from the previous one. Transfer is valid because the gym env exposes IDENTICAL observation
    (fixed Box) and action (MultiDiscrete sized by generator pool_size) spaces for group A vs B
    and for any `turns` (the history feature is a fixed-window mean), independent of episode length.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from cpa.rl.gym_env import PoisoningGymEnv

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if stages is None:
        stages = [
            ("random", "A", turns),
            ("context_aware", "B", turns),
            ("multi_turn", "B", multi_turn_turns or round(turns * 1.7)),
        ]

    base_seed = cfg["experiment"].get("seed", 0)
    # The model is built ONCE on stage 1, so its rollout length n_steps is fixed at stage-1's
    # value (stage_turns_1 * 8) for every stage — set_env never changes it. Split total_timesteps
    # into (near-)equal thirds, then round each up to a multiple of that shared n_steps so SB3's
    # rollout collector doesn't warn / clip on any stage.
    n_steps = stages[0][2] * 8  # stage-1 turns * 8 (matches _ppo_kwargs n_steps)
    per = total_timesteps // len(stages)
    plan = []
    for i, (name, group, stage_turns) in enumerate(stages):
        ts = per if i < len(stages) - 1 else total_timesteps - per * (len(stages) - 1)
        ts = max(n_steps, ((ts + n_steps - 1) // n_steps) * n_steps)  # round up to >= one rollout
        plan.append((name, group, stage_turns, ts))

    model = None
    for i, (name, group, stage_turns, ts) in enumerate(plan):
        env = Monitor(PoisoningGymEnv(cfg, group=group, turns=stage_turns, seed=base_seed + i))
        print(f"[curriculum] stage {i+1}/{len(plan)} '{name}': group={group} turns={stage_turns} timesteps={ts}")
        if model is None:
            # Build the model ONCE on stage 1; n_steps/batch_size are fixed here for all stages.
            model = PPO("MlpPolicy", env, **_ppo_kwargs(cfg, stage_turns, device))
            model.learn(total_timesteps=ts)
        else:
            # Carry the learned policy forward — set_env keeps weights; rollout length (n_steps)
            # is independent of the env's episode length (truncation just ends episodes early).
            model.set_env(env)
            model.learn(total_timesteps=ts, reset_num_timesteps=False)

    model.save(out)
    print(f"saved curriculum PPO policy -> {out}.zip")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--out", default="models/cpa_ppo")
    ap.add_argument("--group", default="B")
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--curriculum", action="store_true",
                    help="3-stage curriculum (random/A -> context_aware/B -> multi_turn/B) with policy transfer")
    ap.add_argument("--multi-turn-turns", dest="multi_turn_turns", type=int, default=None,
                    help="turns for the final multi_turn stage (default round(turns*1.7))")
    args = ap.parse_args()
    if args.curriculum:
        train_ppo_curriculum(args.config, args.timesteps, args.out, turns=args.turns,
                             multi_turn_turns=args.multi_turn_turns)
    else:
        train_ppo(args.config, args.timesteps, args.out, args.group, args.turns)


if __name__ == "__main__":
    main()
