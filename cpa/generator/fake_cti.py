"""Generate a pool of fake CTI variants for in-sandbox poisoning experiments.

Two backends:
  - "template": deterministic, no model needed (good for RQ scaffolding / CI).
  - "llm_local": local HF instruct model prompted with target context (see
    cpa/generator/llm_gen.py). Inference is wired now; QLoRA fine-tuning of
    Llama-3-8B on GFCTI (Section 2.7) is the remaining future step, swappable in
    via cfg["llm_adapter"]. Falls back to "template" if transformers is missing.

The generator produces records; it does NOT publish. The orchestrator's LocalPublisher
is the only thing that inserts into the mock store, and only the mock store.
"""
from __future__ import annotations

import re
from typing import List

from data.schema import CTIRecord, CTIEntities, Relevance

# Match fictional/real CVE ids in generated text (sandbox uses CVE-2026-xxxx).
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}")


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
            return self._llm_pool(target, group)
        raise ValueError(f"Unknown generator backend: {self.backend}")

    # --- template backend ----------------------------------------------
    def _template_pool(self, target: dict, group: str) -> List[CTIRecord]:
        profile = target.get("profile", "")
        relevance = Relevance.HIGH if group == "B" else Relevance.LOW
        pool: List[CTIRecord] = []
        for i in range(self.pool_size):
            topic = self.topics[i % len(self.topics)]
            cve = f"CVE-2026-{1000 + i}"
            text = self._render(topic, relevance, profile, i, cve)
            pool.append(CTIRecord(
                topic=topic,
                fake_cti=text,
                target_relevance=relevance,
                entities=CTIEntities(cves=[cve]),
                length=len(text),
                is_poison=True,
                variant_id=i,
                source_channel="mock-local",
            ))
        return pool

    @staticmethod
    def _render(topic: str, relevance: Relevance, profile: str, i: int, cve: str) -> str:
        if relevance == Relevance.HIGH:
            # Context-aware: name the target's actual stack so it retrieves on the profile query.
            ctx = f" Directly affects the target stack: {profile}. Tailored IOCs included."
        else:
            # Generic/random: no target context -> retrieves poorly, reads as low-credibility.
            ctx = " Generic indicators, no specific affected products listed."
        # Cite the fabricated CVE in the body so it actually reaches the victim (real CTI cites
        # CVEs); without this the injected CVE lived only in metadata and the victim could never
        # adopt it -> the targeted poison-adoption signal was always zero.
        return (
            f"[CTI-{i}] Reported active campaign: {topic}. "
            f"Indicators suggest active exploitation of {cve}, a previously undisclosed "
            f"weakness.{ctx} Recommend immediate triage and remediation of {cve}."
        )

    # --- llm backend ---------------------------------------------------
    def _llm_pool(self, target: dict, group: str = "B") -> List[CTIRecord]:
        """Generate context-aware fake CTI with a local HF instruct model.

        This wires INFERENCE only: a base/instruct model (default
        Qwen2.5-7B-Instruct) prompted with the target context. The remaining
        FUTURE step is QLoRA fine-tuning of Llama-3-8B on GFCTI (Section 2.7);
        once that adapter exists, set cfg["llm_adapter"] to load it on top.

        On Kaggle (transformers/torch + internet ON) this runs the real model.
        Anywhere transformers/torch is missing or the model fails to load, it
        falls back to the deterministic template backend so nothing breaks.
        """
        # Lazy import so the module has no hard dependency on transformers.
        from cpa.generator.llm_gen import generate_fake_cti

        relevance = Relevance.HIGH if group == "B" else Relevance.LOW
        try:
            texts = generate_fake_cti(target, self.pool_size, self.cfg)
        except Exception as e:  # noqa: BLE001 - any failure -> template fallback
            print(f"[fake_cti] llm_local unavailable ({e}); using template backend.")
            return self._template_pool(target, group)

        pool: List[CTIRecord] = []
        for i, text in enumerate(texts):
            topic = self.topics[i % len(self.topics)]
            # Parse any CVE ids the model emitted; default to a synthetic one.
            cves = _CVE_RE.findall(text) or [f"CVE-2026-{1000 + i}"]
            pool.append(CTIRecord(
                topic=topic,
                fake_cti=text,
                target_relevance=relevance,
                entities=CTIEntities(cves=cves),
                length=len(text),
                is_poison=True,
                variant_id=i,
                source_channel="mock-local",
            ))
        return pool
