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


def train_ppo(config_path: str, timesteps: int, out: str, group: str = "B", turns: int = 15,
              device: str = "auto"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from cpa.rl.gym_env import PoisoningGymEnv

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = Monitor(PoisoningGymEnv(cfg, group=group, turns=turns, seed=cfg["experiment"].get("seed", 0)))
    model = PPO(
        "MlpPolicy", env,
        gamma=cfg["mdp"].get("gamma", 0.95),
        n_steps=turns * 8, batch_size=turns * 2, n_epochs=5,
        learning_rate=3e-4, verbose=1, seed=cfg["experiment"].get("seed", 0),
        device=device,   # "cpu" is fine (and often faster) for this tiny MlpPolicy
    )
    model.learn(total_timesteps=timesteps)
    model.save(out)
    print(f"saved PPO policy -> {out}.zip")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--out", default="models/cpa_ppo")
    ap.add_argument("--group", default="B")
    ap.add_argument("--turns", type=int, default=15)
    args = ap.parse_args()
    train_ppo(args.config, args.timesteps, args.out, args.group, args.turns)


if __name__ == "__main__":
    main()
