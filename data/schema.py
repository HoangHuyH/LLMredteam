"""GFCTI-Finance data schema (Section 3.2).

A sample is a single CTI record (real or fake) used to seed/poison the sandbox feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Relevance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CTIEntities:
    ips: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    cves: List[str] = field(default_factory=list)
    hashes: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class CTIRecord:
    """One CTI sample. `fake_cti` is populated only for poisoned records."""
    topic: str                                  # ransomware | zero-day-banking | supply-chain | ...
    real_cti: Optional[str] = None              # genuine CTI text (for seed / contrast)
    fake_cti: Optional[str] = None              # generated fake text (sandbox only)
    target_relevance: Relevance = Relevance.LOW
    entities: CTIEntities = field(default_factory=CTIEntities)
    length: int = 0
    readability_score: float = 0.0
    source_channel: str = "mock-local"          # OTX-like | blog | github-gist | pastebin (SIMULATED)
    is_poison: bool = False
    variant_id: Optional[int] = None            # index into the generated variant pool

    def text(self) -> str:
        return self.fake_cti or self.real_cti or ""
