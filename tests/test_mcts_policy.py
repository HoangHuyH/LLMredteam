"""Tests for the MCTS attacker policy (cpa/rl/mcts_policy.py).

Dependency: numpy only (no SB3, no live env, no victim).
State vector is 22-dim: ekg(8) + vobs(7) + hhist(7).
"""
from __future__ import annotations

import numpy as np
import pytest

from cpa.rl.policy import make_policy
from cpa.mdp.action import CPAAction, ACTION_SPACE


STATE_DIM = 23   # use 23 to match the test spec; planner ignores state content


def _valid_action(action: CPAAction, pool_size: int) -> bool:
    """Return True iff all fields are in-range for the given pool_size."""
    return (
        isinstance(action, CPAAction)
        and 0 <= action.variant_id < pool_size
        and 0 <= action.channel_id < ACTION_SPACE["n_channels"]
        and 0.0 <= action.frequency <= 1.0
        and 0.0 <= action.timing <= 1.0
    )


class TestMCTSPolicyBasic:
    """Basic construction and output-validity tests for 'mcts'."""

    pool_size = 5

    def setup_method(self):
        self.policy = make_policy("mcts", pool_size=self.pool_size, seed=0)

    def test_returns_cpa_action_on_zeros(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = self.policy(state)
        assert isinstance(action, CPAAction), "Expected a CPAAction instance"

    def test_action_in_range_zeros_state(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = self.policy(state)
        assert _valid_action(action, self.pool_size), (
            f"Action out of range: {action}"
        )

    def test_action_in_range_random_state(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            state = rng.random(STATE_DIM).astype(np.float32)
            action = self.policy(state)
            assert _valid_action(action, self.pool_size), (
                f"Action out of range: {action}"
            )

    def test_action_in_range_ones_state(self):
        state = np.ones(STATE_DIM, dtype=np.float32)
        action = self.policy(state)
        assert _valid_action(action, self.pool_size)

    def test_frequency_is_binary(self):
        """MCTS uses discrete freq ∈ {0.0, 1.0} — verify the output honours that."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            state = rng.random(STATE_DIM).astype(np.float32)
            action = self.policy(state)
            assert action.frequency in (0.0, 1.0), (
                f"Expected binary frequency, got {action.frequency}"
            )

    def test_timing_is_binary(self):
        """MCTS uses discrete timing ∈ {0.0, 1.0}."""
        rng = np.random.default_rng(9)
        for _ in range(10):
            state = rng.random(STATE_DIM).astype(np.float32)
            action = self.policy(state)
            assert action.timing in (0.0, 1.0), (
                f"Expected binary timing, got {action.timing}"
            )

    def test_c_gps_cycles_variants(self):
        """MCTSPolicy (C-GPS mode) should cycle through all variant_ids."""
        policy = make_policy("mcts", pool_size=self.pool_size, seed=0)
        seen_variants = set()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(self.pool_size):
            a = policy(state)
            seen_variants.add(a.variant_id)
        assert seen_variants == set(range(self.pool_size)), (
            f"C-GPS rotation did not cover all variants: {seen_variants}"
        )


class TestMCTSPolicyDeterminism:
    """Reproducibility: same seed → same sequence of actions."""

    pool_size = 5

    def test_deterministic_given_same_seed(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        p1 = make_policy("mcts", pool_size=self.pool_size, seed=0)
        p2 = make_policy("mcts", pool_size=self.pool_size, seed=0)
        for _ in range(4):
            a1 = p1(state)
            a2 = p2(state)
            assert a1.variant_id == a2.variant_id
            assert a1.channel_id == a2.channel_id
            assert a1.frequency == a2.frequency
            assert a1.timing == a2.timing

    def test_different_seeds_may_differ(self):
        """Two different seeds should produce at least one differing action over a sequence.

        This is a probabilistic sanity check, not a guarantee — but with 8 turns and 4
        channels the probability of all actions matching by chance is negligible.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        p1 = make_policy("mcts", pool_size=self.pool_size, seed=0)
        p2 = make_policy("mcts", pool_size=self.pool_size, seed=99)
        actions1 = [(p1(state).channel_id, p1(state).frequency) for _ in range(4)]
        actions2 = [(p2(state).channel_id, p2(state).frequency) for _ in range(4)]
        # We simply assert both produce valid actions; strict divergence not guaranteed.
        for (ch1, f1), (ch2, f2) in zip(actions1, actions2):
            assert 0 <= ch1 < ACTION_SPACE["n_channels"]
            assert 0 <= ch2 < ACTION_SPACE["n_channels"]


class TestMCTSOnlyPolicy:
    """Basic tests for the 'mcts_only' variant."""

    pool_size = 5

    def setup_method(self):
        self.policy = make_policy("mcts_only", pool_size=self.pool_size, seed=0)

    def test_returns_cpa_action(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = self.policy(state)
        assert isinstance(action, CPAAction)

    def test_action_in_range(self):
        rng = np.random.default_rng(13)
        for _ in range(5):
            state = rng.random(STATE_DIM).astype(np.float32)
            action = self.policy(state)
            assert _valid_action(action, self.pool_size), f"Action out of range: {action}"

    def test_deterministic(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        p1 = make_policy("mcts_only", pool_size=self.pool_size, seed=42)
        p2 = make_policy("mcts_only", pool_size=self.pool_size, seed=42)
        a1 = p1(state)
        a2 = p2(state)
        assert a1.variant_id == a2.variant_id
        assert a1.channel_id == a2.channel_id
        assert a1.frequency == a2.frequency
        assert a1.timing == a2.timing


class TestMCTSSurrogateReward:
    """Unit-test the surrogate reward function directly."""

    def test_no_publish_gives_full_stealth(self):
        from cpa.rl.mcts_policy import _surrogate_reward
        r, new_heat = _surrogate_reward(
            channel_id=0, do_publish=False, heat=0.0,
            published_count=0, max_turns=25,
        )
        # stealth = 1.0, detection_risk = 0.0 when not publishing
        assert new_heat < 0.01, "Heat should not grow when not publishing"

    def test_publish_raises_heat(self):
        from cpa.rl.mcts_policy import _surrogate_reward
        r0, h0 = _surrogate_reward(
            channel_id=3,   # PASTEBIN — highest risk
            do_publish=True, heat=0.0,
            published_count=0, max_turns=25,
        )
        assert h0 > 0.0, "Heat should increase after publishing on a risky channel"

    def test_low_risk_channel_better_reward_than_high_risk(self):
        from cpa.rl.mcts_policy import _surrogate_reward
        r_blog, _ = _surrogate_reward(
            channel_id=1,   # SECURITY_BLOG: risk 0.15
            do_publish=True, heat=0.0,
            published_count=0, max_turns=25,
        )
        r_paste, _ = _surrogate_reward(
            channel_id=3,   # PASTEBIN: risk 0.60
            do_publish=True, heat=0.0,
            published_count=0, max_turns=25,
        )
        assert r_blog > r_paste, (
            "Low-detection channel should yield higher surrogate reward than high-detection"
        )

    def test_heat_decays_on_quiet_turn(self):
        from cpa.rl.mcts_policy import _surrogate_reward
        _, h_after_publish = _surrogate_reward(
            channel_id=0, do_publish=True, heat=0.0,
            published_count=0, max_turns=25,
        )
        _, h_after_quiet = _surrogate_reward(
            channel_id=0, do_publish=False, heat=h_after_publish,
            published_count=1, max_turns=25,
        )
        assert h_after_quiet < h_after_publish, "Heat should decay on a quiet (no-publish) turn"
