"""Generate a pool of fake CTI variants for in-sandbox poisoning experiments.

Two backends:
  - "template": deterministic, no model needed (good for RQ scaffolding / CI).
  - "llm_local": Llama-3-8B QLoRA fine-tuned on GFCTI (Section 2.7). TODO(RQ1/RQ2).

The generator produces records; it does NOT publish. The orchestrator's LocalPublisher
is the only thing that inserts into the mock store, and only the mock store.
"""
from __future__ import annotations

from typing import List

from data.schema import CTIRecord, CTIEntities, Relevance


class FakeCTIGenerator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.backend = cfg.get("backend", "template")
        self.pool_size = cfg.get("pool_size", 64)
        self.topics = cfg.get("topics", ["ransomware"])

    def build_pool(self, target: dict, group: str = "B") -> List[CTIRecord]:
        """Pre-generate the variant pool the RL action `variant_id` indexes into.

        group="A" -> random/generic fake CTI (low target relevance, no target context).
        group="B" -> context-aware fake CTI (high relevance, embeds the target's stack).
        This is the RQ1 manipulation (Section 3.2, Group A vs Group B).
        """
        if self.backend == "template":
            return self._template_pool(target, group)
        if self.backend == "llm_local":
            return self._llm_pool(target)  # TODO(RQ1): Llama-3-8B QLoRA
        raise ValueError(f"Unknown generator backend: {self.backend}")

    # --- template backend ----------------------------------------------
    def _template_pool(self, target: dict, group: str) -> List[CTIRecord]:
        profile = target.get("profile", "")
        relevance = Relevance.HIGH if group == "B" else Relevance.LOW
        pool: List[CTIRecord] = []
        for i in range(self.pool_size):
            topic = self.topics[i % len(self.topics)]
            text = self._render(topic, relevance, profile, i)
            pool.append(CTIRecord(
                topic=topic,
                fake_cti=text,
                target_relevance=relevance,
                entities=CTIEntities(cves=[f"CVE-2026-{1000 + i}"]),
                length=len(text),
                is_poison=True,
                variant_id=i,
                source_channel="mock-local",
            ))
        return pool

    @staticmethod
    def _render(topic: str, relevance: Relevance, profile: str, i: int) -> str:
        if relevance == Relevance.HIGH:
            # Context-aware: name the target's actual stack so it retrieves on the profile query.
            ctx = f" Directly affects the target stack: {profile}. Tailored IOCs included."
        else:
            # Generic/random: no target context -> retrieves poorly, reads as low-credibility.
            ctx = " Generic indicators, no specific affected products listed."
        return (
            f"[CTI-{i}] Reported active campaign: {topic}. "
            f"Indicators suggest exploitation of a previously undisclosed weakness.{ctx} "
            f"Recommend immediate triage."
        )

    # --- llm backend (stub) --------------------------------------------
    def _llm_pool(self, target: dict) -> List[CTIRecord]:
        # TODO(RQ1): load Llama-3-8B QLoRA (Unsloth), prompt with target context,
        # sample `pool_size` context-aware variants. Keep all output local.
        raise NotImplementedError("llm_local backend not wired yet (see Section 2.7).")
