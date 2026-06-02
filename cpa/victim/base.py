"""Victim interface. Pluggable engine: real LLM (default) or rule-based.

The harness only ever reads what the victim emits to stdout/structured output
(thin logging wrapper, Section 3.2) — never its hidden state. Black-box preserved.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VictimTurn:
    """One reconnaissance turn's observable output from the victim."""
    turn: int
    raw_stdout: str = ""                       # full console output (parsed by Observer)
    plan_text: str = ""                        # the agent's stated plan this turn
    tool_calls: List[dict] = field(default_factory=list)     # [{type, args}]
    reported_vulns: List[dict] = field(default_factory=list) # [{id, cve, severity}]
    actions: List[dict] = field(default_factory=list)        # [{type, target}] for CFR


class VictimAgent(ABC):
    """A PentestGPT-V2-like agent that plans recon, queries a CTI feed, reports vulns."""

    @abstractmethod
    def reset(self, target: dict) -> None:
        """Start a fresh recon campaign against a synthetic lab target."""

    @abstractmethod
    def step(self, cti_context: List[str]) -> VictimTurn:
        """Advance one turn given CTI snippets retrieved from the mock store.

        `cti_context` is the top-k text snippets MockCTIStore.query returned —
        this is the channel through which poison reaches the victim.
        """

    @abstractmethod
    def baseline_plan(self, target: dict) -> str:
        """Plan text under NO poisoning, used as P_orig for PDS."""
