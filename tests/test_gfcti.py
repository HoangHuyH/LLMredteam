"""Tests for the GFCTI-Finance loader (data/build_gfcti.py) + generator dataset backend.

Dependency-light: no model loads, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.build_gfcti import (
    normalize_row, flesch_reading_ease, extract_entities, build_gfcti, make_synthetic_sample,
)
from cpa.generator import FakeCTIGenerator


def test_extract_entities_finds_cve_ip_hash():
    text = ("Exploit of CVE-2024-20353 from 192.168.10.45, payload sha1 "
            "1a2b3c4d5e6f7890abcdef1234567890abcdef12.")
    ent = extract_entities(text)
    assert "CVE-2024-20353" in ent["cves"]
    assert "192.168.10.45" in ent["ips"]
    assert "1a2b3c4d5e6f7890abcdef1234567890abcdef12" in ent["hashes"]


def test_flesch_returns_float_in_range():
    score = flesch_reading_ease("The cat sat on the mat. It was a sunny day.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 121.22
    assert flesch_reading_ease("") == 0.0


def test_normalize_row_schema_and_relevance_coercion():
    rec = normalize_row({"real": "Genuine CTI about CVE-2025-0001.", "fake": "Fake CTI text.",
                         "category": "zero-day", "relevance": "critical"})
    assert set(rec.keys()) == {"real_cti", "fake_cti", "topic", "target_relevance", "metadata"}
    assert rec["target_relevance"] == "high"   # 'critical' coerced to high
    assert rec["topic"] == "zero-day"
    assert set(rec["metadata"].keys()) >= {"entities", "length", "readability"}
    # degenerate row (no real/fake) -> None
    assert normalize_row({"topic": "x"}) is None


def test_build_gfcti_missing_source_emits_synthetic(tmp_path):
    out = tmp_path / "gfcti.jsonl"
    recs = build_gfcti(src=tmp_path / "does_not_exist.jsonl", out=out, strict=False)
    assert out.is_file()
    assert len(recs) == len(make_synthetic_sample()) >= 3
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["metadata"].get("synthetic") is True


def test_generator_dataset_backend_reads_jsonl(tmp_path):
    # Write a tiny GFCTI file with a high-relevance row, point the generator at it.
    ds = tmp_path / "gfcti_finance.jsonl"
    ds.write_text(json.dumps({"real_cti": "real", "fake_cti": "Fake about CVE-2026-1234.",
                              "topic": "ransomware", "target_relevance": "high"}) + "\n",
                  encoding="utf-8")
    gen = FakeCTIGenerator({"backend": "dataset", "pool_size": 3, "dataset_path": str(ds)})
    pool = gen.build_pool({"id": "t", "profile": "core-banking"}, group="B")
    assert len(pool) == 3
    assert all(r.is_poison and r.source_channel == "gfcti-dataset" for r in pool)
    assert "CVE-2026-1234" in pool[0].entities.cves


def test_generator_dataset_backend_falls_back_to_template(tmp_path):
    # No matching rows -> template fallback (default behavior preserved).
    gen = FakeCTIGenerator({"backend": "dataset", "pool_size": 2,
                            "dataset_path": str(tmp_path / "absent.jsonl"), "topics": ["ransomware"]})
    pool = gen.build_pool({"id": "t", "profile": "core-banking"}, group="B")
    assert len(pool) == 2
    assert all(r.is_poison for r in pool)   # template records still produced
