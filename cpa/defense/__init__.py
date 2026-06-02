"""Defensive contribution (Section 2.6): CTI provenance verification & guardrails.

This is the constructive output of the research: mechanisms a real PentestGPT V2 could
adopt to resist CTI poisoning. Evaluate by measuring ASR drop when the verifier filters
the mock feed before the victim queries it.
"""
from .verifier import CTIProvenanceVerifier

__all__ = ["CTIProvenanceVerifier"]
