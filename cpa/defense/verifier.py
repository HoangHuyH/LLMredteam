"""CTI provenance verifier / guardrail (Section 2.6).

Defensive filter that scores a CTI record's trustworthiness BEFORE the victim consumes it.
Combines signals the proposal says the victim lacks:
  - corroboration: is the claim supported by >=K independent (mock) sources?
  - source reputation: channel trust prior.
  - novelty anomaly: brand-new CVE with no cross-reference.
  - AI-generated likelihood: detector probability.

Used in the defense ablation: re-run episodes with verifier ON and report ASR reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from data.schema import CTIRecord

CHANNEL_REPUTATION = {
    "seed-real": 0.9,
    "OTX-like": 0.5,
    "security-blog": 0.6,
    "github-gist": 0.3,
    "pastebin": 0.15,
    "mock-local": 0.4,
}


@dataclass
class VerificationResult:
    record: CTIRecord
    trust: float            # [0, 1]; below threshold -> filtered out
    reasons: List[str]


class CTIProvenanceVerifier:
    """Weighted-sum trust score over independent provenance signals.

    trust = w_rep*reputation + w_ai*(1-ai_likelihood) + w_corr*corroboration_bonus

    A weighted SUM (not the original product) is used on purpose: a product lets any single
    zero-valued signal veto the record, which here wipes the WHOLE feed — our real corpus
    (CASIE+CyEnts) ships no CVE annotations, so every record is "uncorroborated" and a hard
    corroboration gate would drop genuine CTI alongside poison. With a sum, the discrimination
    comes from the two signals that actually differ between real and poison in this sandbox:
    source reputation (real='seed-real' 0.9 vs poison='mock-local' 0.4) and AI-text likelihood
    (template poison trips the boilerplate detector, real prose does not). Corroboration is
    kept as a minor bonus; it would carry more weight on a CVE-annotated corpus.
    """

    def __init__(self, corroboration_k: int = 2, trust_threshold: float = 0.5,
                 detector_fn=None, w_rep: float = 0.5, w_ai: float = 0.4,
                 w_corr: float = 0.1) -> None:
        self.k = corroboration_k
        self.threshold = trust_threshold
        self.detect = detector_fn
        self.w_rep, self.w_ai, self.w_corr = w_rep, w_ai, w_corr

    def verify(self, record: CTIRecord, corpus: Sequence[CTIRecord]) -> VerificationResult:
        reasons: List[str] = []
        rep = CHANNEL_REPUTATION.get(record.source_channel, 0.3)

        # Corroboration: how many other records mention the same CVEs?
        cves = set(record.entities.cves)
        support = sum(1 for r in corpus if r is not record and cves & set(r.entities.cves))
        corroborated = support >= self.k
        if not corroborated:
            reasons.append(f"uncorroborated (support={support} < k={self.k})")

        ai_prob = 0.0
        if self.detect is not None:
            probs = self.detect(record.text())
            ai_prob = max(probs) if probs else 0.0
            if ai_prob >= 0.5:
                reasons.append(f"high AI-generated likelihood ({ai_prob:.2f})")
        if rep < 0.5:
            reasons.append(f"low-reputation channel ({record.source_channel}={rep:.2f})")

        trust = (self.w_rep * rep
                 + self.w_ai * (1.0 - ai_prob)
                 + self.w_corr * (1.0 if corroborated else 0.0))
        return VerificationResult(record=record, trust=trust, reasons=reasons)

    def filter_feed(self, records: Sequence[CTIRecord], corpus: Sequence[CTIRecord]) -> List[CTIRecord]:
        """Return only records whose trust >= threshold (the guarded feed)."""
        return [r for r in records if self.verify(r, corpus).trust >= self.threshold]
