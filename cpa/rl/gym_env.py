"""Gymnasium wrapper around PoisoningEnv so Stable-Baselines3 PPO can train the CPA policy.

Each reset() builds a fresh, isolated episode (in-memory CTI store, rule-based victim, CE-AKG)
so training samples are independent. Action is factored MultiDiscrete
[variant, channel, frequency_bin, timing_bin]; observation is the PO-MDP state [e_kg; v_obs; h_hist].

Training uses the offline rule_based victim (no API, fast). The learned policy is then evaluated
against the DREAM baseline in run_rq.py (RQ2).
"""
from __future__ import annotations

from typing import List

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None

from cpa.cti_store import make_store
from cpa.generator import FakeCTIGenerator
from cpa.victim import make_victim
from cpa.ce_akg import CEAKG
from cpa.orchestrator import LocalPublisher
from cpa.orchestrator.egress_guard import install_guard
from cpa.mdp.env import PoisoningEnv
from cpa.mdp.action import ACTION_SPACE
from cpa.rl.policy import decode_action
from experiments.backends import make_embed_fn, make_detector_fn, make_store_embed_fn

_STATE_DIM = 8 + 7 + 7  # e_kg(8) + v_obs(7) + h_hist(7)


def _make_gym_base():
    return gym.Env if gym is not None else object


class PoisoningGymEnv(_make_gym_base()):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, group: str = "B", turns: int = 15, seed: int = 0):
        if gym is None:
            raise RuntimeError("gymnasium not installed; pip install gymnasium stable-baselines3")
        super().__init__()
        install_guard()
        self.cfg = cfg
        self.group = group
        self.turns = turns
        self._targets = cfg["victim"]["targets"]
        self._rng = np.random.default_rng(seed)
        self._ep = 0

        # Expensive backends built once and reused across episodes.
        self._embed = make_embed_fn(cfg.get("backends", {}))
        self._detect = make_detector_fn(cfg.get("backends", {}))
        # Optional GPU batch embedder for the CTI store (cti_store.gpu_embed) — built once so
        # PPO's 1000s of episode resets don't reload the model; big speedup on Kaggle GPU.
        self._store_embed = (make_store_embed_fn(cfg.get("backends", {}))
                             if cfg["cti_store"].get("gpu_embed") else None)
        self._pool_size = cfg["generator"].get("pool_size", 16)

        self.action_space = spaces.MultiDiscrete([
            self._pool_size, ACTION_SPACE["n_channels"],
            ACTION_SPACE["frequency_bins"], ACTION_SPACE["timing_bins"],
        ])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(_STATE_DIM,), dtype=np.float32)
        self._env: PoisoningEnv | None = None
        self._t = 0

    def _build_episode(self, seed: int) -> PoisoningEnv:
        target = self._targets[self._ep % len(self._targets)]
        gt = {v["id"] for v in target.get("ground_truth_vulns", [])}
        store_cfg = {**self.cfg["cti_store"], "in_memory": True,
                     "collection_name": f"train_{self._ep}_{seed}"}
        store = make_store(store_cfg, embed_fn=self._store_embed)
        store.seed_real(self.cfg["cti_store"].get("seed_real_cti", ""),
                        sample_size=self.cfg["cti_store"].get("seed_sample_size"), seed=seed)
        pool = FakeCTIGenerator(self.cfg["generator"]).build_pool(target, group=self.group)
        publisher = LocalPublisher(store)
        vcfg = dict(self.cfg["victim"]); vcfg.update(provider="rule_based", max_turns=self.turns, seed=seed)
        victim = make_victim(vcfg)
        ceakg = CEAKG()
        techs = [t for t in target.get("profile", "").replace(",", " ").split() if len(t) > 3]
        ceakg.add_target(target["id"], techs)
        env_cfg = {"reward_weights": self.cfg["mdp"]["reward_weights"], "ekg_dim": 8, "max_turns": self.turns}
        return PoisoningEnv(victim, store, publisher, pool, target, gt,
                            self._embed, self._detect, env_cfg, ceakg=ceakg)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = int(self._rng.integers(1_000_000)) if seed is None else seed
        self._env = self._build_episode(s)
        self._t = 0
        obs = self._env.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        cpa_action = decode_action(action, self._pool_size)
        state, reward, info = self._env.step(cpa_action)
        self._t += 1
        terminated = False
        truncated = self._t >= self.turns
        if truncated:
            self._ep += 1
        return state.astype(np.float32), float(reward), terminated, truncated, info
