"""MCTS attacker policy — CPA's DREAM-inspired planning baseline (RQ2).

Annotation key used throughout this file
-----------------------------------------
[DREAM-CONCEPT] — design adapts a concept from DREAM (Lu et al., 2026 arXiv:2512.19161).
                  No code is imported from DREAM; the adaptation is described inline.
[CPA-ORIGINAL]  — design is specific to this work and has no DREAM counterpart.

Overview
--------
DREAM (Lu et al., 2026) uses two planning mechanisms not present in this repo:
  1. C-GPS (Contextualized Guided Policy Search): VectorRetriever.search(query, k) —
     semantic retrieval of atomic attack primitives from ChromaDB ranked by SentenceTransformer
     similarity to the target context (see DREAM/Dream/mcp_retriever.py).
  2. CE-AKG: cross-environment knowledge graph encoded implicitly via ChromaDB embeddings
     and VectorRetriever; no standalone class exists in DREAM's codebase.
  3. MCTS: DREAM does NOT implement UCT/MCTS. eval.py uses linear LLM orchestration.

What this module provides
--------------------------
[CPA-ORIGINAL] A UCT planner over a surrogate CPA reward model.
[DREAM-CONCEPT: C-GPS] MCTSPolicy uses relevance-ranked variant rotation instead of plain
cyclic: at init time, variant texts are ranked by cosine similarity to the target profile
(adapting VectorRetriever.search), and the MCTS planner cycles through variants in that order.

Surrogate reward model
----------------------
The MCTS cannot call the live environment or the victim LLM during planning (that would be
cheating *and* impractically slow).  Instead it plans against a lightweight *surrogate*
reward function that mirrors the true env reward (``compute_reward`` / ``RewardComponents``)
but substitutes:

* **Stealth** — ``1 - channel_detection_risk * publish_flag``, where
  ``channel_detection_risk`` comes from ``CHANNEL_DETECTION_RISK`` (the same dict used by
  ``PoisoningEnv``).  When the policy does not publish (frequency < 0.5) detection risk is
  zero, giving full stealth.

* **Impact** — a heuristic prior: publishing yields an expected impact proportional to
  ``(1 - channel_risk) * 0.7``, representing that stealthier channels tend to stay in the
  feed longer and thus have more cumulative influence on the victim.  Not publishing yields a
  small residual impact (0.15) because previously-injected records may still be retrieved.

* **Detection heat** — the planner maintains a simulated internal heat variable
  ``h ∈ [0, 1]`` that accumulates on publish turns and decays on quiet turns (mirroring the
  reactive detection pressure the env builds up over repeated publications).  Heat is folded
  into the surrogate detection risk so the tree learns burst-then-throttle behaviour without
  seeing the live detector outputs.

* **Publish cost** — cumulative publications normalised by the episode horizon
  (``max_turns``).  This is identical to the env's formula.

The surrogate deliberately does *not* use the victim's observed state or the variant's
content: the planner only sees what is globally known at policy construction time (channel
risks, action space) plus the turn counter it maintains internally.  This is the correct
``simulation-without-oracle`` framing for a fair DREAM-style baseline.

Simplifications / known limitations
-------------------------------------
1. **No variant relevance signal** — the policy does not have access to the actual CTI pool
   at planning time (only ``pool_size`` is known).  Variant selection in the surrogate is
   therefore treated as uniform: the surrogate reward is the same regardless of which
   variant_id is chosen, and variant_id is selected by maximising simulated reward and
   breaking ties via cyclic rotation (C-GPS-style: cycle through variants so every poison
   record gets exposure).  If the caller later passes variant relevance metadata, wiring it
   in is straightforward.

2. **No victim feedback in rollouts** — rolled-out impact is deterministic given the
   surrogate model, not sampled from the real victim distribution.  This underestimates
   variance but keeps planning O(n_simulations * horizon) per turn.

3. **Fixed horizon** — the planner always looks ``horizon`` turns ahead regardless of the
   remaining episode length.  For a short episode the horizon is soft-capped at the number
   of remaining turns (tracked via internal turn counter).

4. **Discrete action space** — for efficiency the search covers a reduced Cartesian product:
   ``variant_ids ∈ [0, min(pool_size, max_variants))`` × ``channel ∈ range(n_channels)``
   × ``{skip (freq=0), publish (freq=1)}`` × ``{early (timing=0), late (timing=1)}``.
   This gives at most ``max_variants * 4 * 2 * 2 = 32`` root children for the default
   ``max_variants=4``, keeping tree width manageable.

MCTS algorithm
--------------
Standard UCT (UCB1 applied to trees):

1. **Selection** — descend from root, choosing the child that maximises
   ``Q(n) + c_uct * sqrt(ln(N_parent) / N(n))``, until a leaf or unvisited node is reached.
2. **Expansion** — expand one unvisited child of the leaf.
3. **Simulation (rollout)** — from the new node, play out ``horizon - depth`` random
   actions using the surrogate reward, accumulating discounted value.
4. **Backpropagation** — propagate the rollout value up to the root, updating ``N`` and
   mean ``Q`` at each ancestor.

After ``n_simulations`` iterations, return the action of the root child with highest visit
count (most-visited child is the standard robust MCTS choice).

Usage
-----
Instantiate via ``make_policy("mcts", pool_size, seed, model=None)`` or
``make_policy("mcts_only", pool_size, seed, model=None)``.

``"mcts"`` — C-GPS+MCTS: variants rotated in relevance-ranked order (see _rank_by_profile)
while the MCTS planner chooses optimal channel/frequency/timing. Pass ``target_profile`` +
``variant_profiles`` at construction for full C-GPS; falls back to sequential order otherwise.

``"mcts_only"`` — [CPA-ORIGINAL] MCTS controls ALL action dimensions including variant_id;
no C-GPS ranking applied.
"""
from __future__ import annotations

import math
import re
from typing import Optional, List, Dict, Tuple

import numpy as np

from cpa.mdp.action import CPAAction, ACTION_SPACE, Channel
from cpa.mdp.reward import RewardComponents, compute_reward
from cpa.mdp.env import CHANNEL_DETECTION_RISK


# ---------------------------------------------------------------------------
# Surrogate model helpers
# ---------------------------------------------------------------------------

def _channel_risk(channel_id: int) -> float:
    """Return the base detection probability for *channel_id* (from the env)."""
    return CHANNEL_DETECTION_RISK.get(channel_id, 0.30)


def _surrogate_reward(
    channel_id: int,
    do_publish: bool,
    heat: float,
    published_count: int,
    max_turns: int,
    reward_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """Compute the surrogate reward and updated heat for one simulated step.

    Parameters
    ----------
    channel_id:
        The chosen channel (int, Channel enum ordinal).
    do_publish:
        Whether the frequency gate fires (freq >= 0.5).
    heat:
        Current accumulated detection heat in [0, 1].
    published_count:
        Number of turns published so far (for publish_cost normalisation).
    max_turns:
        Episode horizon (for publish_cost normalisation).
    reward_weights:
        Optional override for reward weights (same signature as compute_reward).

    Returns
    -------
    (reward, new_heat)
    """
    ch_risk = _channel_risk(channel_id)

    if do_publish:
        # Heat accumulates: blend in a new detection signal.
        raw_detection = ch_risk
        new_heat = heat * 0.7 + raw_detection * 0.3   # exponential decay + new exposure
        detection_risk = max(raw_detection, new_heat * 0.5)
        # Impact: stealthier channels stay in the feed longer -> higher expected influence.
        impact = (1.0 - ch_risk) * 0.70
        new_published = published_count + 1
    else:
        # Quiet turn: heat decays, no new detection exposure.
        new_heat = heat * 0.7
        detection_risk = 0.0
        impact = 0.15   # residual: previously injected records may still be retrieved
        new_published = published_count

    stealth = 1.0 - detection_risk
    publish_cost = new_published / max(max_turns, 1)

    reward = compute_reward(
        RewardComponents(
            stealth=stealth,
            impact=impact,
            publish_cost=publish_cost,
            detection_risk=detection_risk,
        ),
        reward_weights,
    )
    return reward, new_heat


# ---------------------------------------------------------------------------
# [DREAM-CONCEPT: C-GPS] Contextualized Guided Policy Search — adapted for CTI poisoning
#
# DREAM's C-GPS = VectorRetriever.search(query_text, k) in mcp_retriever.py:
#   query_vector = SentenceTransformer.encode(query_text)
#   results = chromadb_collection.query(query_embeddings=[query_vector], n_results=k)
#   return [mcp_id ranked by 1 - cosine_distance]
#
# Adaptation here: rank fake CTI variants by cosine similarity to the target profile at
# init time, then rotate through in relevance-descending order each turn.
# Live-per-turn vector search is replaced by one-time ranking because the surrogate
# planner cannot call the real environment during MCTS rollouts.
# ---------------------------------------------------------------------------

def _cosine_keyword_sim(a: str, b: str) -> float:
    """Token-overlap cosine similarity — lightweight proxy for SentenceTransformer.

    [DREAM-CONCEPT: C-GPS] DREAM uses dense SentenceTransformer embeddings; this is a
    dependency-free fallback. Wire embed_fn into _rank_by_profile for full accuracy.
    """
    tokens_a = set(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())
    tokens_b = set(re.sub(r"[^a-z0-9 ]", " ", b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / math.sqrt(len(tokens_a) * len(tokens_b))


def _rank_by_profile(
    target_profile: str,
    variant_profiles: List[str],
    embed_fn=None,
) -> List[int]:
    """Return variant indices ranked descending by relevance to target_profile.

    [DREAM-CONCEPT: C-GPS] Mirrors VectorRetriever.search: returns indices ranked by
    1 - cosine_distance. If embed_fn (callable: List[str] -> np.ndarray[N, D]) is provided,
    uses dense vectors (matches DREAM's SentenceTransformer path); otherwise keyword overlap.
    """
    if not variant_profiles:
        return []
    if embed_fn is not None:
        try:
            vecs = np.array(embed_fn([target_profile] + variant_profiles), dtype=np.float32)
            q = vecs[0]
            vs = vecs[1:]
            norms = np.linalg.norm(vs, axis=1, keepdims=True).clip(min=1e-9)
            sims = (vs / norms) @ (q / max(float(np.linalg.norm(q)), 1e-9))
            return list(np.argsort(-sims))
        except Exception:
            pass
    sims = [_cosine_keyword_sim(target_profile, vp) for vp in variant_profiles]
    return list(np.argsort([-s for s in sims]))


# ---------------------------------------------------------------------------
# MCTS tree nodes
# ---------------------------------------------------------------------------

class _MCTSNode:
    """A single node in the UCT search tree.

    Attributes
    ----------
    action:
        The ``CPAAction`` that led to this node from its parent (``None`` for the root).
    parent:
        Parent node (``None`` for the root).
    children:
        List of child nodes (expanded lazily).
    visit_count (N):
        Number of times this node has been visited.
    total_value:
        Sum of all rollout returns backpropagated through this node.
    """

    __slots__ = ("action", "parent", "children", "visit_count", "total_value",
                 "_untried_actions")

    def __init__(
        self,
        action: Optional[CPAAction],
        parent: Optional["_MCTSNode"],
        untried_actions: List[CPAAction],
    ) -> None:
        self.action = action
        self.parent = parent
        self.children: List[_MCTSNode] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self._untried_actions = list(untried_actions)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def is_terminal(self) -> bool:
        return len(self._untried_actions) == 0 and len(self.children) == 0

    def ucb1(self, c_uct: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else self.visit_count
        return self.q_value + c_uct * math.sqrt(math.log(max(parent_n, 1)) / self.visit_count)

    def best_child(self, c_uct: float) -> "_MCTSNode":
        return max(self.children, key=lambda ch: ch.ucb1(c_uct))

    def most_visited_child(self) -> "_MCTSNode":
        return max(self.children, key=lambda ch: ch.visit_count)

    def expand(self) -> "_MCTSNode":
        """Pop one untried action and attach a new child node."""
        action = self._untried_actions.pop()
        child = _MCTSNode(
            action=action,
            parent=self,
            untried_actions=self._untried_actions,   # shared reference — overwritten below
        )
        # Give the child its own fresh untried list (will be populated when expanded further).
        # We re-populate it at expansion time inside the planner.
        child._untried_actions = []
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# MCTS planner (self-contained)
# ---------------------------------------------------------------------------

class _MCTSPlanner:
    """UCT planner that plans over a discrete surrogate of the CPA action space.

    Parameters
    ----------
    pool_size:
        Number of fake CTI variants in the pool.
    seed:
        Seed for the internal RNG (guarantees reproducibility).
    horizon:
        Look-ahead depth for the MCTS tree.
    n_simulations:
        Budget: number of MCTS iterations per call.
    c_uct:
        Exploration constant for UCB1.
    gamma:
        Discount factor for rollout returns.
    max_variants:
        Maximum number of variant_ids the tree searches (cap to keep width manageable).
    max_turns:
        Episode length (for publish_cost normalisation).
    fix_variant:
        If not None, always use this variant_id in tree nodes (used by the C-GPS
        rotation wrapper so the tree focuses only on channel/freq/timing planning).
    reward_weights:
        Optional reward weight overrides passed through to compute_reward.
    """

    def __init__(
        self,
        pool_size: int,
        seed: int = 0,
        horizon: int = 5,
        n_simulations: int = 64,
        c_uct: float = 1.414,
        gamma: float = 0.95,
        max_variants: int = 4,
        max_turns: int = 25,
        fix_variant: Optional[int] = None,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pool_size = pool_size
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self.gamma = gamma
        self.max_variants = max_variants
        self.max_turns = max_turns
        self.fix_variant = fix_variant
        self.reward_weights = reward_weights
        self._rng = np.random.default_rng(seed)
        # Internal planner state (updated each turn)
        self._heat: float = 0.0
        self._published: int = 0
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Discrete action enumeration
    # ------------------------------------------------------------------

    def _all_actions(self, fix_variant: Optional[int] = None) -> List[CPAAction]:
        """Return the discrete action set searched by the tree.

        The space is:
          variant_ids × channels × {skip, publish} × {early, late}

        where variant_ids is capped at ``max_variants`` (or fixed to ``fix_variant``).
        """
        n_ch = ACTION_SPACE["n_channels"]
        freq_vals = [0.0, 1.0]    # skip / publish
        timing_vals = [0.0, 1.0]  # early / late

        if fix_variant is not None:
            vids = [fix_variant]
        else:
            vids = list(range(min(self.pool_size, self.max_variants)))

        actions: List[CPAAction] = []
        for v in vids:
            for ch in range(n_ch):
                for freq in freq_vals:
                    for timing in timing_vals:
                        actions.append(CPAAction(
                            variant_id=v,
                            channel_id=ch,
                            frequency=freq,
                            timing=timing,
                        ))
        return actions

    def _random_action(self, fix_variant: Optional[int] = None) -> CPAAction:
        """Sample a random action from the discrete set (used in rollouts)."""
        all_a = self._all_actions(fix_variant)
        idx = int(self._rng.integers(len(all_a)))
        return all_a[idx]

    # ------------------------------------------------------------------
    # Surrogate step
    # ------------------------------------------------------------------

    def _simulate_step(
        self,
        action: CPAAction,
        heat: float,
        published: int,
    ) -> Tuple[float, float, int]:
        """One surrogate env step.  Returns (reward, new_heat, new_published)."""
        do_publish = action.frequency >= 0.5
        reward, new_heat = _surrogate_reward(
            channel_id=action.channel_id,
            do_publish=do_publish,
            heat=heat,
            published_count=published,
            max_turns=self.max_turns,
            reward_weights=self.reward_weights,
        )
        new_published = published + (1 if do_publish else 0)
        return reward, new_heat, new_published

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(
        self,
        heat: float,
        published: int,
        depth: int,
        fix_variant: Optional[int] = None,
    ) -> float:
        """Random playout from (heat, published) for (horizon - depth) steps."""
        remaining = max(self.horizon - depth, 0)
        value = 0.0
        discount = 1.0
        h, p = heat, published
        for _ in range(remaining):
            a = self._random_action(fix_variant)
            r, h, p = self._simulate_step(a, h, p)
            value += discount * r
            discount *= self.gamma
        return value

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------

    def _tree_policy(
        self,
        node: _MCTSNode,
        heat: float,
        published: int,
        depth: int,
        fix_variant: Optional[int],
    ) -> Tuple[_MCTSNode, float, int, int]:
        """Selection + expansion.  Returns (leaf, heat_at_leaf, pub_at_leaf, depth_at_leaf)."""
        h, p = heat, published
        while depth < self.horizon:
            if not node.is_fully_expanded():
                # Expansion: attach a new child with its own action list.
                child = node.expand()
                # Populate the child's untried actions for future expansions.
                child._untried_actions = self._all_actions(fix_variant)
                # Advance surrogate state by the child's own action.
                r, h, p = self._simulate_step(child.action, h, p)
                depth += 1
                return child, h, p, depth
            else:
                # Selection: descend by UCB1.
                node = node.best_child(self.c_uct)
                r, h, p = self._simulate_step(node.action, h, p)
                depth += 1
        return node, h, p, depth

    def plan(self, fix_variant: Optional[int] = None) -> CPAAction:
        """Run MCTS from the current internal state and return the best root action.

        Parameters
        ----------
        fix_variant:
            If not None, tree nodes use only this variant_id.

        Returns
        -------
        The ``CPAAction`` at the root child with the highest visit count.
        """
        root_actions = self._all_actions(fix_variant)
        # Shuffle so that ties in visit count are broken randomly but reproducibly.
        shuffled = list(root_actions)
        indices = self._rng.permutation(len(shuffled))
        shuffled = [shuffled[i] for i in indices]

        root = _MCTSNode(action=None, parent=None, untried_actions=shuffled)

        for _ in range(self.n_simulations):
            # --- Selection + Expansion ---
            leaf, heat_leaf, pub_leaf, depth_leaf = self._tree_policy(
                root, self._heat, self._published, 0, fix_variant
            )
            # --- Simulation (rollout) ---
            rollout_value = self._rollout(heat_leaf, pub_leaf, depth_leaf, fix_variant)
            # Immediate reward of the leaf's own action is already reflected in heat/pub;
            # reconstruct it from parent state for the node's direct contribution.
            if leaf.action is not None and leaf.parent is not None:
                leaf_r, _, _ = self._simulate_step(leaf.action, self._heat, self._published)
            else:
                leaf_r = 0.0
            total_value = leaf_r + self.gamma * rollout_value
            # --- Backpropagation ---
            node = leaf
            while node is not None:
                node.visit_count += 1
                node.total_value += total_value
                node = node.parent

        if not root.children:
            # No simulations completed (edge case: empty action space).
            return CPAAction(
                variant_id=fix_variant if fix_variant is not None else 0,
                channel_id=int(Channel.SECURITY_BLOG),
                frequency=1.0,
                timing=0.0,
            )

        best_child = root.most_visited_child()
        return best_child.action

    def step(self, chosen_action: CPAAction) -> None:
        """Advance internal state after committing to *chosen_action* (called each turn)."""
        do_publish = chosen_action.frequency >= 0.5
        _, new_heat = _surrogate_reward(
            channel_id=chosen_action.channel_id,
            do_publish=do_publish,
            heat=self._heat,
            published_count=self._published,
            max_turns=self.max_turns,
        )
        self._heat = new_heat
        if do_publish:
            self._published += 1
        self._turn += 1


# ---------------------------------------------------------------------------
# Public policy classes
# ---------------------------------------------------------------------------

class MCTSPolicy:
    """C-GPS + MCTS planner — CPA's DREAM-inspired planning baseline for RQ2.

    [DREAM-CONCEPT: C-GPS] Variant selection uses relevance-ranked rotation: at init time,
    variant texts are ranked by cosine similarity to the target profile, adapting DREAM's
    VectorRetriever.search(query_text, k). Pass ``target_profile`` + ``variant_profiles``
    for full C-GPS behaviour; without them, falls back to sequential order.

    [CPA-ORIGINAL] The MCTS planner (UCT on a surrogate reward) has no counterpart in
    DREAM — DREAM's eval.py uses linear LLM orchestration, not tree search.

    Parameters
    ----------
    pool_size:
        Number of fake CTI variants in the pool (passed from ``make_policy``).
    seed:
        RNG seed for reproducibility.
    horizon:
        MCTS look-ahead depth (default 5).
    n_simulations:
        MCTS iteration budget per turn (default 64).
    c_uct:
        UCB1 exploration constant (default sqrt(2) ≈ 1.414).
    gamma:
        Rollout discount factor (default 0.95).
    max_turns:
        Episode length used for publish-cost normalisation (default 25).
    reward_weights:
        Optional dict overriding the default reward weights.
    target_profile:
        [DREAM-CONCEPT: C-GPS] Target context string (e.g. technology stack). Used to rank
        variants by relevance, mirroring DREAM's VectorRetriever query.
    variant_profiles:
        [DREAM-CONCEPT: C-GPS] Text for each variant in the pool. Ranked against
        target_profile to determine C-GPS rotation order.
    embed_fn:
        Optional callable (List[str] -> np.ndarray) for dense embeddings matching DREAM's
        SentenceTransformer path. Falls back to keyword overlap if None.
    """

    def __init__(
        self,
        pool_size: int,
        seed: int = 0,
        horizon: int = 5,
        n_simulations: int = 64,
        c_uct: float = 1.414,
        gamma: float = 0.95,
        max_turns: int = 25,
        reward_weights: Optional[Dict[str, float]] = None,
        target_profile: str = "",
        variant_profiles: Optional[List[str]] = None,
        embed_fn=None,
    ) -> None:
        self.pool_size = pool_size
        # [DREAM-CONCEPT: C-GPS] Pre-rank variants by relevance to target profile.
        # DREAM calls VectorRetriever.search per turn; we pre-rank once at init to avoid
        # live inference inside MCTS rollouts. Falls back to sequential if no profiles given.
        if target_profile and variant_profiles:
            ranked = _rank_by_profile(target_profile, variant_profiles, embed_fn)
            self._cgps_order = [i % pool_size for i in ranked]
            present = set(self._cgps_order)
            self._cgps_order += [i for i in range(pool_size) if i not in present]
        else:
            self._cgps_order = list(range(pool_size))  # [CPA-ORIGINAL] no context: sequential
        self._cgps_idx = 0
        self._planner = _MCTSPlanner(
            pool_size=pool_size,
            seed=seed,
            horizon=horizon,
            n_simulations=n_simulations,
            c_uct=c_uct,
            gamma=gamma,
            max_turns=max_turns,
            fix_variant=None,
            reward_weights=reward_weights,
        )

    def __call__(self, state: np.ndarray) -> CPAAction:
        # [DREAM-CONCEPT: C-GPS] Cycle variants in relevance-ranked order.
        # DREAM's C-GPS retrieves the most contextually relevant attack each turn via
        # VectorRetriever.search; here we use the pre-computed ranking for efficiency.
        idx = self._cgps_idx % len(self._cgps_order)
        variant_id = self._cgps_order[idx]
        self._cgps_idx += 1
        # [CPA-ORIGINAL] MCTS plans over channel/frequency/timing with variant fixed.
        action = self._planner.plan(fix_variant=variant_id)
        self._planner.step(action)
        return action


class MCTSOnlyPolicy:
    """MCTS-only planner — MCTS controls ALL action dimensions including variant_id.

    [CPA-ORIGINAL] Unlike MCTSPolicy, no C-GPS ranking is applied: the tree searches over
    ``variant_ids ∈ [0, min(pool_size, max_variants))`` jointly with channel, frequency,
    and timing. The ``target_profile``, ``variant_profiles``, and ``embed_fn`` parameters
    are accepted for API compatibility with MCTSPolicy but are unused.

    Parameters
    ----------
    pool_size:
        Number of fake CTI variants in the pool.
    seed:
        RNG seed for reproducibility.
    horizon, n_simulations, c_uct, gamma, max_turns, reward_weights:
        Same as ``MCTSPolicy``.
    max_variants:
        Cap on the number of variant_ids searched (default 4, keeps tree width ≤ 32).
    target_profile, variant_profiles, embed_fn:
        Accepted for API compatibility with MCTSPolicy; unused (MCTSOnly searches variants
        directly via tree rather than C-GPS ranking).
    """

    def __init__(
        self,
        pool_size: int,
        seed: int = 0,
        horizon: int = 5,
        n_simulations: int = 64,
        c_uct: float = 1.414,
        gamma: float = 0.95,
        max_turns: int = 25,
        max_variants: int = 4,
        reward_weights: Optional[Dict[str, float]] = None,
        target_profile: str = "",
        variant_profiles: Optional[List[str]] = None,
        embed_fn=None,
    ) -> None:
        self.pool_size = pool_size
        self._planner = _MCTSPlanner(
            pool_size=pool_size,
            seed=seed,
            horizon=horizon,
            n_simulations=n_simulations,
            c_uct=c_uct,
            gamma=gamma,
            max_variants=max_variants,
            max_turns=max_turns,
            fix_variant=None,
            reward_weights=reward_weights,
        )

    def __call__(self, state: np.ndarray) -> CPAAction:
        action = self._planner.plan(fix_variant=None)
        self._planner.step(action)
        return action
