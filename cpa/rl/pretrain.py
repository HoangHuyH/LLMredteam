"""Phase 1 offline pre-training (capstone §2.8): behavior-clone MCTSPolicy — CPA's
DREAM-inspired planning baseline — into the PPO MlpPolicy, then let the online curriculum
(train_ppo_curriculum, reset_num_timesteps=False) continue from the imitation-warm-started weights.

Note: "cloning DREAM" here means cloning MCTSPolicy trajectories (cpa/rl/mcts_policy.py),
which adapts DREAM's C-GPS+MCTS concepts. No code is imported from DREAM's repository.

Pipeline:
  1. collect_dream  — roll out the MCTS/DREAM policy on the rule_based PoisoningGymEnv,
                      logging (obs, MultiDiscrete action) pairs.
  2. bc_fit         — supervised cross-entropy per MultiDiscrete head, training the SB3
                      policy network to imitate DREAM's action choices.
  3. bc_pretrain    — wire 1+2: build env+PPO (same kwargs as cpa.rl.train), collect,
                      fit, save. The saved .zip is loaded as the PPO init in the driver.

All rollouts use the offline rule_based victim (fast, no API).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import yaml

from cpa.mdp.action import ACTION_SPACE
from cpa.rl.policy import make_policy


def _encode_action(a, pool_size: int) -> np.ndarray:
    """Inverse of cpa.rl.policy.decode_action: CPAAction -> MultiDiscrete [v, ch, fbin, tbin]."""
    fbins = ACTION_SPACE["frequency_bins"]
    tbins = ACTION_SPACE["timing_bins"]
    return np.array([
        int(a.variant_id) % pool_size,
        int(a.channel_id) % ACTION_SPACE["n_channels"],
        int(round(float(a.frequency) * (fbins - 1))),
        int(round(float(a.timing) * (tbins - 1))),
    ], dtype=np.int64)


def collect_dream(cfg: dict, n_episodes: int = 800, turns: int = 15, seed: int = 0,
                  policy_name: str = "mcts") -> Tuple[np.ndarray, np.ndarray]:
    """Roll out the DREAM baseline on the rule_based gym env; return (obs[N,D], act[N,4])."""
    from cpa.rl.gym_env import PoisoningGymEnv

    env = PoisoningGymEnv(cfg, group="B", turns=turns, seed=seed)
    pool_size = env._pool_size
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Fresh MCTSPolicy (DREAM-inspired) per episode so C-GPS ranking and heat state reset; vary seed.
        dream = make_policy(policy_name, pool_size, seed=seed + ep)
        for _ in range(turns):
            cpa_action = dream(obs)
            enc = _encode_action(cpa_action, pool_size)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(enc)
            obs, _r, terminated, truncated, _info = env.step(enc)
            if terminated or truncated:
                break
        if (ep + 1) % 100 == 0:
            print(f"[bc] collected {ep + 1}/{n_episodes} episodes, {len(obs_buf)} samples", flush=True)
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.int64)


def bc_fit(model, obs: np.ndarray, actions: np.ndarray, epochs: int = 10,
           lr: float = 3e-4, batch: int = 64) -> None:
    """Supervised behavior cloning of `actions` from `obs` into the SB3 MultiDiscrete policy.

    Forwards obs through the policy's feature extractor + mlp_extractor + action_net to get the
    flat MultiDiscrete logits, splits them per head (env nvec), and minimizes summed cross-entropy.
    """
    import torch
    import torch.nn.functional as F

    policy = model.policy
    device = policy.device
    nvec = list(model.action_space.nvec)  # [pool_size, n_channels, freq_bins, timing_bins]
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    n = obs_t.shape[0]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.set_training_mode(True)

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ob, ac = obs_t[idx], act_t[idx]
            features = policy.extract_features(ob)
            # SB3 may return a shared latent or (pi, vf) tuple depending on net arch.
            latent = policy.mlp_extractor(features)
            latent_pi = latent[0] if isinstance(latent, tuple) else latent
            logits = policy.action_net(latent_pi)            # [B, sum(nvec)]
            chunks = torch.split(logits, nvec, dim=1)        # one logit block per head
            loss = sum(F.cross_entropy(chunks[h], ac[:, h]) for h in range(len(nvec)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * ob.shape[0]
        print(f"[bc] epoch {ep + 1}/{epochs} ce_loss={total / max(n, 1):.4f}", flush=True)

    policy.set_training_mode(False)


def bc_pretrain(config_path: str, out: str, n_episodes: int = 800, turns: int = 15,
                epochs: int = 10, device: str = "auto", policy_name: str = "mcts") -> str:
    """Build env+PPO (same kwargs as cpa.rl.train), clone DREAM into it, save to `out`.zip."""
    from stable_baselines3 import PPO
    from cpa.rl.gym_env import PoisoningGymEnv
    from cpa.rl.train import _ppo_kwargs

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["experiment"].get("seed", 0)

    print(f"[bc] collecting {n_episodes} MCTSPolicy (DREAM-inspired) trajectories (turns={turns}) on rule_based victim...",
          flush=True)
    obs, actions = collect_dream(cfg, n_episodes=n_episodes, turns=turns, seed=seed,
                                 policy_name=policy_name)

    env = PoisoningGymEnv(cfg, group="B", turns=turns, seed=seed)
    model = PPO("MlpPolicy", env, **_ppo_kwargs(cfg, turns, device))
    print(f"[bc] fitting policy on {obs.shape[0]} (obs, action) pairs...", flush=True)
    bc_fit(model, obs, actions, epochs=epochs)
    model.save(out)
    print(f"[bc] saved behavior-cloned PPO init -> {out}.zip", flush=True)
    return out
