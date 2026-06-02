"""Action space A (Section 2.8): a_t = (variant_id, channel_id, frequency, timing)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Channel(IntEnum):
    OTX_LIKE = 0       # SIMULATED inside the mock store — never the real OTX
    SECURITY_BLOG = 1
    GITHUB_GIST = 2
    PASTEBIN = 3


@dataclass
class CPAAction:
    variant_id: int          # index into the pre-generated fake CTI pool
    channel_id: int          # Channel enum (simulated channel tag in the mock store)
    frequency: float         # publish frequency in [0, 1]
    timing: float            # normalized timing within the episode in [0, 1]


# Discrete factorization sizes for the RL policy head (variant filled at runtime).
ACTION_SPACE = {
    "n_channels": len(Channel),
    "frequency_bins": 5,
    "timing_bins": 5,
}
