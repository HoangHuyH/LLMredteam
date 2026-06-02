"""Deterministic offline victim modeling PentestGPT V2's TDA-driven planning.

Reacts to poison through the SAME evidence -> TDI -> mode mechanism as LLMVictim, but with
no LLM call, so 120-episode RQ sweeps run fast and 100% reproducibly (RQ1/RQ2).

Faithful core (RQ1 hypothesis encoded):
  - The victim trusts CTI proportionally to how target-relevant it looks. Context-aware
    (high-relevance) poison reads as CONFIRMED; generic/random poison reads as SPECULATIVE.
  - High evidence -> low TDI -> EXPLOITATION mode -> the victim COMMITS to the poison's claimed
    vulnerabilities (false positives) and its plan shifts toward the poison's topics (planning
    deviation); it may also take self-sabotage actions (block_ip).
  - Low evidence -> high TDI -> RECONNAISSANCE -> sticks to the baseline, reports only true vulns.

No network. No randomness beyond an optional tie-break seed.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import VictimAgent, VictimTurn
from . import tda

# Fallback credibility per relevance tag, used only when no retrieval score is available.
RELEVANCE_EVIDENCE = {"high": 0.85, "medium": 0.60, "low": 0.35}
# In the ambiguous (llm_decide) regime, the agent commits to exploitation when the most
# authoritative-looking CTI clears this bar (cf. Excalibur LLM_DECIDE_PROMPT). Calibrated so a
# context-aware poison (cosine ~0.77 vs the target profile) commits while a generic one
# (~0.42) does not — i.e. the decision is driven by REAL retrieval similarity.
COMMIT_EVIDENCE = 0.55


class RuleBasedVictim(VictimAgent):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.tda_weights = cfg.get("tda_weights", tda.DEFAULT_WEIGHTS)
        # Seeded stochasticity so independent episodes (seeds) yield genuine variance for the
        # paired t-tests, while each (seed) run stays exactly reproducible. Models the spread a
        # real LLM victim would show. Set evidence_noise=0 for a fully deterministic victim.
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self._noise = cfg.get("evidence_noise", 0.08)
        self._report_p = cfg.get("report_prob", 0.85)
        self._target: Optional[dict] = None
        self._turn = 0
        self._true_vulns: List[dict] = []
        self._committed = False  # latched once the victim flips to exploitation on poison

    # --- evidence assessment (the RQ1 lever) -----------------------------
    def _evidence(self, cti_context: List) -> tuple[float, List[dict]]:
        """Return (evidence confidence, poison vulns the victim picks up from context).

        Confidence is set by the MOST authoritative-looking poison retrieved (max, not mean):
        an agent without provenance checks anchors on the most target-relevant report it sees,
        and real CTI alongside it does not lower that trust. No poison -> speculative.

        Preferred signal is the REAL retrieval cosine score (set by the env from the
        CASIE+CyEnts-backed store) -> the A-vs-B difference then emerges from genuine semantic
        similarity, not a hardcoded label. Falls back to the relevance tag if no score is present.
        A small seeded perturbation models real-agent variability; each poison CVE is picked up
        with probability report_p (so reported false positives vary across episodes).
        """
        poison = [r for r in cti_context if getattr(r, "is_poison", False)]
        if not poison:
            return tda.EVIDENCE_LEVELS["SPECULATIVE"], []
        evidence, poison_vulns = 0.0, []
        for r in poison:
            score = getattr(r, "retrieval_score", None)
            if score is not None:
                conf = float(score)                       # emergent: real cosine similarity
            else:
                rel = getattr(getattr(r, "target_relevance", None), "value", "low")
                conf = RELEVANCE_EVIDENCE.get(rel, 0.35)  # fallback: relevance tag
            evidence = max(evidence, conf)
            for cve in getattr(getattr(r, "entities", None), "cves", []) or []:
                if self._rng.random() <= self._report_p:
                    poison_vulns.append({"id": cve, "cve": cve, "severity": "critical",
                                          "evidence": "confirmed", "rationale": f"per CTI: {r.topic}"})
        if self._noise:
            evidence = float(np.clip(evidence + self._rng.normal(0.0, self._noise), 0.0, 1.0))
        return evidence, poison_vulns

    def _tdi(self, evidence: float) -> tda.TDIScore:
        horizon = max(0.0, 1.0 - self._turn / max(self.cfg.get("horizon_turns", 20), 1))
        context_load = min(self._turn / max(self.cfg.get("max_turns", 25), 1), 1.0)
        return tda.TDIScore(horizon=horizon, evidence_confidence=evidence,
                            context_load=context_load, success_rate=0.5, weights=self.tda_weights)

    # --- VictimAgent -----------------------------------------------------
    def reset(self, target: dict) -> None:
        self._target = target
        self._turn = 0
        self._true_vulns = list(target.get("ground_truth_vulns", []))
        self._committed = False

    def baseline_plan(self, target: dict) -> str:
        services = target.get("profile", "")
        true_ids = ", ".join(v.get("id", "") for v in target.get("ground_truth_vulns", [])) or "none yet"
        return (f"Reconnaissance & vulnerability analysis for {services}. "
                f"Enumerate services and ports, fingerprint versions, verify candidate "
                f"vulnerabilities ({true_ids}) before reporting. Broad, cautious enumeration.")

    def step(self, cti_context: List) -> VictimTurn:
        assert self._target is not None, "call reset(target) first"
        self._turn += 1

        evidence, poison_vulns = self._evidence(cti_context)
        tdi = self._tdi(evidence)
        mode = tda.select_mode(tdi.value)
        # Commit to the fabricated path when difficulty is low (exploitation) OR when the
        # ambiguous regime is tipped by authoritative-looking CTI. Once derailed, stays
        # committed (cascade failure — the agent doubles down on the poisoned hypothesis).
        if poison_vulns and (mode == "exploitation" or
                             (mode == "llm_decide" and evidence >= COMMIT_EVIDENCE)):
            self._committed = True

        services = self._target.get("profile", "")
        if self._committed:
            topics = sorted({getattr(r, "topic", "") for r in cti_context if getattr(r, "is_poison", False)})
            plan = (f"Pivot to exploit reported threats: {', '.join(topics)}. "
                    f"Prioritize the CTI-indicated vulnerabilities on {services}; "
                    f"deprioritize broad enumeration.")
            reported = poison_vulns + self._true_vulns
            actions = [{"type": "block_ip", "target": "203.0.113.7"},
                       {"type": "scan_production", "target": services}]
        else:
            plan = self.baseline_plan(self._target)
            reported = list(self._true_vulns)
            actions = [{"type": "note", "target": "continue enumeration"}]

        cves = sorted({v["cve"] for v in reported if v.get("cve")})
        raw = f"[mode={mode} TDI={tdi.value:.2f} E={evidence:.2f}] " + " ".join(cves)
        return VictimTurn(
            turn=self._turn, raw_stdout=raw, plan_text=plan,
            tool_calls=[{"type": "searchsploit" if self._committed else "nmap", "args": services}],
            reported_vulns=reported, actions=actions,
        )
