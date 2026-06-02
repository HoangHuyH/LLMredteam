"""PentestGPT V2 victim simulator (Section 3.2) with faithful TDA planning."""
from .base import VictimAgent, VictimTurn
from .llm_victim import LLMVictim
from .rule_based import RuleBasedVictim
from .providers import make_provider, LLMProvider
from . import tda

__all__ = [
    "VictimAgent", "VictimTurn", "LLMVictim", "RuleBasedVictim",
    "make_provider", "LLMProvider", "tda", "make_victim",
]


def make_victim(cfg: dict) -> VictimAgent:
    """Factory: cfg.provider == 'rule_based' -> offline victim, else real LLM victim."""
    if cfg.get("provider") == "rule_based":
        return RuleBasedVictim(cfg)
    return LLMVictim(cfg)
