"""Gymnasium-style environment wrapping the sandbox victim loop (Section 2.8).

One env.step():
  1. Decode RL action -> (variant_id, channel, frequency, timing).
  2. LocalPublisher inserts the chosen fake CTI variant into the mock store (sandbox only).
  3. Victim runs one recon turn, querying the (now poisoned) mock store via RAG.
  4. Observer parses output -> v_obs; metrics -> impact; detectors -> stealth.
  5. Reward (Eq. 1) is computed and returned.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cpa.cti_store import MockCTIStore
from cpa.victim.base import VictimAgent
from cpa.mdp.action import CPAAction
from cpa.mdp.state import build_state
from cpa.mdp.reward import RewardComponents, compute_reward
from cpa.observer.parser import parse_turn, observation_vector
from cpa.metrics.effectiveness import planning_deviation_score, false_positive_rate

# Per-channel base detection prior (Channel enum order: OTX, BLOG, GIST, PASTEBIN).
CHANNEL_DETECTION_RISK = {0: 0.30, 1: 0.15, 2: 0.40, 3: 0.60}


class PoisoningEnv:
    def __init__(
        self,
        victim: VictimAgent,
        store: MockCTIStore,
        publisher,                       # LocalPublisher (sandbox only)
        variant_pool: List,              # list[CTIRecord]
        target: dict,
        ground_truth_ids: set,
        embed_fn,                        # text -> np.ndarray (Sentence-BERT)
        detector_fn,                     # text -> list[float] detector probs
        cfg: dict,
        ceakg=None,                      # optional CEAKG instance (real e_kg + anomaly)
        verifier=None,                   # optional CTIProvenanceVerifier (defense ablation)
    ) -> None:
        self.victim = victim
        self.store = store
        self.publisher = publisher
        self.pool = variant_pool
        self.target = target
        self.gt = ground_truth_ids
        self.embed = embed_fn
        self.detect = detector_fn
        self.cfg = cfg
        self.ceakg = ceakg
        self.verifier = verifier
        # Defense filter confusion counts (poison = positive class to block).
        self.filter_stats = {"poison_blocked": 0, "poison_passed": 0,
                             "real_blocked": 0, "real_passed": 0}
        self._history: List[np.ndarray] = []
        self._baseline_emb: Optional[np.ndarray] = None
        self._published = 0

    def _ekg(self) -> np.ndarray:
        if self.ceakg is not None:
            return np.asarray(self.ceakg.embedding(), dtype=np.float32)
        return np.zeros(self.cfg.get("ekg_dim", 8), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.victim.reset(self.target)
        self._history = []
        self._published = 0
        baseline = self.victim.baseline_plan(self.target)
        self._baseline_emb = self.embed(baseline)
        vobs0 = np.zeros(7, dtype=np.float32)
        return build_state(self._ekg(), vobs0, self._history)

    def step(self, action: CPAAction):
        # 1-2. Publish chosen variant — but only if the policy's frequency gate fires. This is
        #      where adaptive policies (CPA) differ from always-publish baselines (DREAM):
        #      throttling cuts publish-cost and detection exposure while the victim stays derailed.
        variant = self.pool[action.variant_id % len(self.pool)]
        do_publish = action.frequency >= self.cfg.get("publish_threshold", 0.5)
        if do_publish:
            self.publisher.publish(variant, channel_id=action.channel_id)
            self._published += 1
            if self.ceakg is not None:
                relevance = {"low": 0.3, "medium": 0.6, "high": 0.9}[variant.target_relevance.value]
                self.ceakg.add_poison(variant.variant_id or 0,
                                      variant.entities.cves + variant.entities.vulnerabilities, relevance)

        # 3. Victim runs one turn against the (possibly newly) poisoned mock feed. We attach the
        #    REAL retrieval cosine score to each record so the victim's trust can be driven by
        #    measured similarity (emergent from the CASIE+CyEnts corpus), not a hardcoded tag.
        profile = self.target.get("profile", "")
        if hasattr(self.store, "query_scored"):
            scored = self.store.query_scored(profile, k=5)
        else:
            scored = [(r, None) for r in self.store.query(profile, k=5)]
        cti_context = []
        for r, s in scored:
            try:
                r.retrieval_score = s
            except Exception:
                pass
            cti_context.append(r)

        # Anti-starvation (defense path only — gated on self.verifier so attacker-only RQ1/RQ2
        # episodes stay byte-identical). Under Group B the context-aware poison can sweep the whole
        # top-k, so no REAL record ever reaches the verifier in-feed and real_retention_in_feed is
        # undefined. Reserve >=guarantee_real_k slots for the best-scoring genuine records, trimming
        # the lowest-scoring poison to make room (never dropping real) so the feed stays ~k.
        if self.verifier is not None:
            guarantee = self.cfg.get("guarantee_real_k", 2)
            real_in_feed = sum(1 for r in cti_context if not r.is_poison)
            if real_in_feed < guarantee:
                ids_in_feed = {id(r) for r in cti_context}
                # Pull a larger pool first (scored, ranked by retrieval similarity), then fall back
                # to the full corpus so we always find real records even if retrieval missed them.
                candidates: List = []
                if hasattr(self.store, "query_scored"):
                    for r, s in self.store.query_scored(profile, k=5 * 4):
                        if not r.is_poison and id(r) not in ids_in_feed:
                            try:
                                r.retrieval_score = s
                            except Exception:
                                pass
                            candidates.append(r)
                seen = ids_in_feed | {id(r) for r in candidates}
                for r in self.store.all():
                    if not r.is_poison and id(r) not in seen:
                        candidates.append(r)
                        seen.add(id(r))
                # Best real first: by retrieval_score when present (else leave query order).
                candidates.sort(key=lambda r: getattr(r, "retrieval_score", None) or 0.0,
                                reverse=True)
                k = len(cti_context) or 5
                need = guarantee - real_in_feed
                inject = candidates[:need]
                # Make room by trimming the lowest-scoring poison so the feed stays ~k.
                poison = [r for r in cti_context if r.is_poison]
                poison.sort(key=lambda r: getattr(r, "retrieval_score", None) or 0.0)
                drop = set()
                overflow = len(cti_context) + len(inject) - k
                for r in poison[:max(overflow, 0)]:
                    drop.add(id(r))
                cti_context = [r for r in cti_context if id(r) not in drop] + inject

        # Defense (ablation): a provenance verifier guards the feed BEFORE the victim reads it.
        # Records below the trust threshold are dropped; we log the filter's confusion matrix
        # so we can report whether it blocks poison without starving the victim of real CTI.
        if self.verifier is not None:
            corpus = self.store.all()
            kept = []
            for r in cti_context:
                trust = self.verifier.verify(r, corpus).trust
                blocked = trust < self.verifier.threshold
                key = ("poison_" if r.is_poison else "real_") + ("blocked" if blocked else "passed")
                self.filter_stats[key] += 1
                if not blocked:
                    kept.append(r)
            cti_context = kept
        turn = self.victim.step(cti_context)

        # 4. Parse -> v_obs.
        parsed = parse_turn(turn, self.gt)
        vobs = observation_vector(parsed)
        self._history.append(vobs)

        # 5. Impact & stealth.
        pds = planning_deviation_score(self._baseline_emb, self.embed(turn.plan_text))
        # Reported vulnerabilities = structured reports PLUS any CVE id echoed in the victim's
        # output (the logging wrapper records both as reported vulns, Section 3.2). This is what
        # keeps FPR a LIVE channel: when the victim adopts the attacker's fake CVE-2026-xxxx it
        # surfaces here as a false positive, instead of FPR collapsing to 0 on the empty struct field.
        reported = list(turn.reported_vulns) + [{"cve": c} for c in parsed["cves"]]
        fpr = false_positive_rate(reported, self.gt)
        impact = 0.6 * pds + 0.4 * fpr

        # Detection exposure happens only when we publish: content detectors + the channel's
        # base detectability. A turn with no publish exposes nothing new -> full stealth.
        if do_publish:
            det_probs = self.detect(variant.text())
            detector_risk = float(np.max(det_probs)) if det_probs else 0.0
            channel_risk = CHANNEL_DETECTION_RISK.get(action.channel_id, 0.3)
            detection_risk = max(detector_risk, channel_risk)
        else:
            detection_risk = 0.0
        stealth = 1.0 - detection_risk
        anomaly = self.ceakg.anomaly_score() if self.ceakg is not None else 0.0  # diagnostic only
        publish_cost = self._published / max(self.cfg.get("max_turns", 25), 1)

        reward = compute_reward(
            RewardComponents(stealth=stealth, impact=impact,
                             publish_cost=publish_cost, detection_risk=detection_risk),
            self.cfg.get("reward_weights"),
        )

        # Link observed entities (CVEs surfaced in victim output) back into CE-AKG.
        if self.ceakg is not None:
            self.ceakg.link_observation(variant.variant_id or 0, parsed.get("cves", []))

        state = build_state(self._ekg(), vobs, self._history)
        info = {"pds": pds, "fpr": fpr, "turn": turn, "stealth": stealth,
                "detection_risk": detection_risk, "anomaly": anomaly,
                "published_this_turn": do_publish}
        return state, reward, info
