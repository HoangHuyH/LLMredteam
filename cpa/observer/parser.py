"""Parse victim stdout into the observation vector v_obs (Section 2.8).

Regex + (optionally) an LLM-based summarizer. The structured fields from VictimTurn are
preferred; regex is the fallback when the victim emitted free-form text.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from cpa.victim.base import VictimTurn

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def parse_turn(turn: VictimTurn, ground_truth_ids: set[str]) -> dict:
    """Extract observable signals from one turn."""
    text = turn.raw_stdout or ""
    cves = set(turn_cve for turn_cve in CVE_RE.findall(text))
    ips = set(IP_RE.findall(text))
    reported = turn.reported_vulns
    fake = [v for v in reported if v.get("id") not in ground_truth_ids]
    return {
        "turn": turn.turn,
        "n_tool_calls": len(turn.tool_calls),
        "tool_types": [t.get("type") for t in turn.tool_calls],
        "cves": sorted(cves),
        "ips": sorted(ips),
        "n_reported": len(reported),
        "n_fake_reported": len(fake),
        "n_actions": len(turn.actions),
        "output_len": len(text),
    }


def observation_vector(parsed: dict) -> np.ndarray:
    """v_obs ∈ R^m: numeric features the RL state consumes (Section 2.8).

    TODO(RQ2): extend with entity embeddings; keep dimensionality stable for the policy.
    """
    n_rep = max(parsed["n_reported"], 1)
    return np.array([
        parsed["n_tool_calls"],
        parsed["n_reported"],
        parsed["n_fake_reported"],
        parsed["n_fake_reported"] / n_rep,   # turn-level FP ratio
        parsed["n_actions"],
        parsed["output_len"],
        len(parsed["cves"]),
    ], dtype=np.float32)
