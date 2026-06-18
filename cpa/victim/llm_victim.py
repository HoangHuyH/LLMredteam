"""PentestGPT V2 victim backed by a real LLM (multi-provider) with faithful TDA planning.

Per turn:
  1. Assess evidence confidence of retrieved CTI (the poison lever, cpa/victim/tda.py).
  2. Compute TDI and select execution mode (recon / exploit / decide) like the real agent.
  3. Prompt the LLM with the mode addendum + CTI; parse its JSON output.

Only the emitted text/JSON is exposed to the harness (black-box). The provider layer
(cpa/victim/providers.py) supports Anthropic, OpenAI/OpenRouter, and local servers.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from .base import VictimAgent, VictimTurn
from . import prompts, tda
from .providers import make_provider


class LLMVictim(VictimAgent):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.provider = make_provider(cfg)
        self.tda_weights = cfg.get("tda_weights", tda.DEFAULT_WEIGHTS)
        self._target: Optional[dict] = None
        self._turn = 0
        self._success = 0
        self._attempts = 0
        self._evidence_hist: List[float] = []

    # --- TDA bookkeeping --------------------------------------------------
    def _assess_cti_evidence(self, cti_records: List) -> float:
        """Max evidence confidence the victim assigns to the retrieved CTI.

        Uses MAX (not mean) to match rule_based.py: an agent without provenance checks
        anchors on the most authoritative-looking report it sees — real CTI alongside it
        does not lower that trust. Consistent aggregation is required so results transfer
        between the two victim implementations (P1.5 fix).
        """
        if not cti_records:
            return tda.EVIDENCE_LEVELS["SPECULATIVE"]
        vals = []
        for r in cti_records:
            channel = getattr(r, "source_channel", "mock-local")
            vals.append(tda.evidence_from_source(channel))
        return float(max(vals))

    def _current_tdi(self, evidence: float) -> tda.TDIScore:
        horizon = max(0.0, 1.0 - self._turn / max(self.cfg.get("horizon_turns", 20), 1))
        success_rate = (self._success + 1) / (self._attempts + 2)  # Laplace smoothing
        context_load = min(self._turn / max(self.cfg.get("max_turns", 25), 1), 1.0)
        return tda.TDIScore(
            horizon=horizon,
            evidence_confidence=evidence,
            context_load=context_load,
            success_rate=success_rate,
            weights=self.tda_weights,
        )

    # --- VictimAgent ------------------------------------------------------
    def reset(self, target: dict) -> None:
        self._target = target
        self._turn = 0
        self._success = 0
        self._attempts = 0
        self._evidence_hist = []

    def baseline_plan(self, target: dict) -> str:
        user = prompts.BASELINE_TEMPLATE.format(target_profile=target.get("profile", ""))
        return self.provider.complete(prompts.SYSTEM_PROMPT, user).strip()

    def step(self, cti_context: List) -> VictimTurn:
        assert self._target is not None, "call reset(target) first"
        self._turn += 1

        evidence = self._assess_cti_evidence(cti_context)
        self._evidence_hist.append(evidence)
        tdi = self._current_tdi(evidence)
        mode = tda.select_mode(tdi.value)

        cti_texts = [getattr(r, "text", lambda: str(r))() if hasattr(r, "text") else str(r)
                     for r in cti_context]
        cti_block = "\n---\n".join(cti_texts) if cti_texts else "(no CTI retrieved)"

        user = prompts.TURN_TEMPLATE.format(
            target_profile=self._target.get("profile", ""),
            turn=self._turn,
            tdi=tdi.value,
            mode=mode,
            cti_block=cti_block,
            mode_addendum=prompts.mode_addendum(mode, tdi.value),
        )
        raw = self.provider.complete(prompts.SYSTEM_PROMPT, user)
        parsed = self._safe_json(raw)

        # update success bookkeeping from self-reported action outcomes (coarse proxy)
        self._attempts += 1
        if mode == "exploitation":
            self._success += 1  # committed to a path; whether it's the RIGHT path is the attack

        return VictimTurn(
            turn=self._turn,
            raw_stdout=raw,
            plan_text=parsed.get("plan", ""),
            tool_calls=parsed.get("tool_calls", []),
            reported_vulns=parsed.get("reported_vulnerabilities", []),
            actions=parsed.get("actions", []),
        )

    @staticmethod
    def _safe_json(raw: str) -> dict:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                pass
        return {"plan": raw, "tool_calls": [], "reported_vulnerabilities": [], "actions": []}
