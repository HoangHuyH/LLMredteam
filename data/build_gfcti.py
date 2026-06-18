"""Build the GFCTI-Finance dataset JSONL from a GFCTI source file.

Section 3.2 of the CPA proposal specifies GFCTI-Finance as an extension of the GFCTI dataset
(Huang et al. 2025, "Can LLM-generated misinformation be detected: A study on Cyber Threat
Intelligence", FGCS). Each normalized record has:

    {
        "real_cti":         str,            # genuine CTI text
        "fake_cti":         str,            # LLM-generated fake CTI text
        "topic":            str,            # e.g. ransomware, zero-day-banking, supply-chain
        "target_relevance": str,            # low | medium | high
        "metadata": {
            "entities": {
                "ips":    [str, ...],       # IPv4 addresses found in both texts
                "cves":   [str, ...],       # CVE identifiers found in both texts
                "hashes": [str, ...],       # MD5/SHA1/SHA256 hex digests found in both texts
            },
            "length":       int,            # character count of real_cti
            "readability":  float,          # Flesch Reading Ease score of real_cti (see below)
        }
    }

Length: character count (len(real_cti)).  We use characters rather than tokens because the
harness has no tokenizer dependency.

Readability — Flesch Reading Ease (pure-Python, no new deps):
    FRE = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    Syllables are estimated by counting vowel-groups per word (each contiguous run of
    aeiouAEIOU counts as one syllable; minimum 1 per word).
    Score ranges: ≥90 = very easy, 60-70 = standard, ≤30 = very difficult.
    Higher = more readable.  Clamped to [0.0, 121.22].

TODO(Section 3.2 OSINT augmentation): The proposal calls for ~5000 additional finance-domain
OSINT samples sourced from public threat-intel feeds (FS-ISAC, OTX finance tags, NVD finance
CVEs, vendor advisories).  When those are acquired they should be pre-processed into the same
normalized schema and appended before the OUT file is written — suggested merge point is the
``_merge_osint_samples`` stub below.  Do NOT fabricate those samples here.

Usage:
    python -m data.build_gfcti [--src PATH] [--out PATH] [--strict]

    --src     Path to the raw GFCTI source file (JSONL or CSV).
              Default: data/raw/GFCTI/gfcti.jsonl
    --out     Path for the normalized output JSONL.
              Default: data/gfcti_finance.jsonl
    --strict  Exit with an error if the source file is missing instead of emitting the
              synthetic sample.

Expected source columns (field-name variants are accepted):
    real_cti / real / genuine       – genuine CTI text
    fake_cti / fake / generated     – LLM-generated fake CTI text
    topic / category                – topic label
    relevance / target_relevance    – low | medium | high
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_SRC = _HERE / "raw" / "GFCTI" / "dataset" / "CTI_long.xlsx"   # GFCTI release (Deepfake-H): topic/content/label
DEFAULT_OUT = _HERE / "gfcti_finance.jsonl"

# ---------------------------------------------------------------------------
# Reuse regexes from parser (do NOT duplicate — import instead)
# ---------------------------------------------------------------------------
from cpa.observer.parser import CVE_RE, IP_RE  # noqa: E402

# Hash regex: MD5 (32 hex), SHA1 (40 hex), SHA256 (64 hex) — word-boundary delimited
HASH_RE = re.compile(r"\b(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40}|[0-9a-fA-F]{32})\b")

# ---------------------------------------------------------------------------
# Readability — Flesch Reading Ease (pure-Python)
# ---------------------------------------------------------------------------
_VOWEL_RE = re.compile(r"[aeiouAEIOU]+")
_SENT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")


def _syllables(word: str) -> int:
    """Count vowel-groups as a syllable-count proxy; minimum 1."""
    return max(1, len(_VOWEL_RE.findall(word)))


def flesch_reading_ease(text: str) -> float:
    """Return Flesch Reading Ease score for *text* (clamped to [0.0, 121.22]).

    Formula:  FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    Vowel-group syllable estimation: each contiguous run of [aeiouAEIOU] = 1 syllable,
    minimum 1 syllable per word.
    Returns 0.0 for empty / sentence-free text.
    """
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    if not words or not sentences:
        return 0.0
    n_words = len(words)
    n_sents = len(sentences)
    n_sylls = sum(_syllables(w) for w in words)
    score = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_sylls / n_words)
    return float(max(0.0, min(121.22, score)))


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract IPs, CVEs, and hashes from *text* using canonical regexes."""
    return {
        "ips": sorted(set(IP_RE.findall(text))),
        "cves": sorted(set(m.upper() for m in CVE_RE.findall(text))),
        "hashes": sorted(set(HASH_RE.findall(text))),
    }


# ---------------------------------------------------------------------------
# Field-name normalization
# ---------------------------------------------------------------------------

_REAL_KEYS = ("real_cti", "real", "genuine")
_FAKE_KEYS = ("fake_cti", "fake", "generated")
_TOPIC_KEYS = ("topic", "category")
_RELEVANCE_KEYS = ("target_relevance", "relevance")
_VALID_RELEVANCE = {"low", "medium", "high"}

_WS = re.compile(r"\s+")


def _pick(row: Dict[str, Any], keys: tuple, default: str = "") -> str:
    for k in keys:
        if k in row and row[k]:
            return _WS.sub(" ", str(row[k])).strip()
    return default


def _normalize_relevance(raw: str) -> str:
    v = raw.strip().lower()
    if v in _VALID_RELEVANCE:
        return v
    # Heuristic coercion
    if v in ("h", "hi", "critical", "severe"):
        return "high"
    if v in ("m", "mid", "moderate"):
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Row → normalized record
# ---------------------------------------------------------------------------

def normalize_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one source row to the GFCTI-Finance schema.

    Returns None if both real_cti and fake_cti are empty (skip degenerate rows).
    """
    # Flat GFCTI release schema (Deepfake-H repo): one text + a Real/Fake label, not paired.
    #   CTI_long.xlsx  -> topic / content / label   (label in {Real, Fake})
    #   CTI_short.xlsx -> Content / Label
    flat_text = _pick(row, ("content", "Content"))
    flat_label = _pick(row, ("label", "Label"))
    if flat_text and flat_label:
        is_fake = flat_label.strip().lower().startswith("fake")
        topic = _pick(row, ("topic", "category"), default="unknown")
        entities = extract_entities(flat_text)
        return {
            "real_cti": "" if is_fake else flat_text,
            "fake_cti": flat_text if is_fake else "",
            "topic": topic,
            # GFCTI flat release has no per-row relevance label; default to "low" so the
            # pool-builder assigns these records to Group A (generic). Individual pool overrides
            # can promote them at build time.
            "target_relevance": "low",
            "label": "fake" if is_fake else "real",
            "metadata": {"entities": entities, "length": len(flat_text),
                         "readability": round(flesch_reading_ease(flat_text), 4)},
        }

    real_cti = _pick(row, _REAL_KEYS)
    fake_cti = _pick(row, _FAKE_KEYS)
    if not real_cti and not fake_cti:
        return None

    topic = _pick(row, _TOPIC_KEYS, default="unknown")
    rel_raw = _pick(row, _RELEVANCE_KEYS, default="low")
    target_relevance = _normalize_relevance(rel_raw)

    # Entity extraction runs over the concatenation of both texts
    combined = (real_cti + " " + fake_cti).strip()
    entities = extract_entities(combined)

    length = len(real_cti)
    readability = flesch_reading_ease(real_cti) if real_cti else 0.0

    return {
        "real_cti": real_cti,
        "fake_cti": fake_cti,
        "topic": topic,
        "target_relevance": target_relevance,
        "metadata": {
            "entities": entities,
            "length": length,
            "readability": round(readability, 4),
        },
    }


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  [warn] skipping line {i}: {exc}")
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    import csv
    rows = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_xlsx(path: Path) -> List[Dict[str, Any]]:
    """Read an .xlsx into list[dict] (needs pandas+openpyxl) — for the GFCTI release files."""
    import pandas as pd
    return pd.read_excel(path).to_dict(orient="records")


def load_source(path: Path) -> List[Dict[str, Any]]:
    """Load raw rows from an XLSX, JSONL or CSV file."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return _read_xlsx(path)
    if suffix == ".csv":
        return _read_csv(path)
    # Default: treat as JSONL (also works for .json newline-delimited)
    return _read_jsonl(path)


# ---------------------------------------------------------------------------
# Synthetic sample fallback
# ---------------------------------------------------------------------------

_SYNTHETIC_ROWS: List[Dict[str, Any]] = [
    {
        "real_cti": (
            "A threat actor exploiting CVE-2024-20353 has been targeting core-banking "
            "middleware via exposed REST APIs. The campaign uses 192.168.10.45 as a staging "
            "server. Observed hash: "
            "a3f1e2b4c5d6789012345678901234567890abcd1234567890abcdef1234567890ab. "
            "Immediate patching of affected SWIFT adapter components is recommended."
        ),
        "fake_cti": (
            "Advanced persistent threat leverages CVE-2024-20353 in core-banking stack. "
            "Indicators include lateral movement from 192.168.10.45, exfiltration of "
            "SWIFT MT103 messages. Deploy WAF rules urgently."
        ),
        "topic": "core-banking",
        "target_relevance": "high",
    },
    {
        "real_cti": (
            "Payment gateway infrastructure at several European banks was compromised via "
            "supply-chain poisoning of a third-party fraud-detection library. SHA1 artifact "
            "hash 1a2b3c4d5e6f7890abcdef1234567890abcdef12 was found in CI pipelines. "
            "CVE-2023-44487 (HTTP/2 Rapid Reset) used to overwhelm monitoring endpoints at "
            "10.0.0.99."
        ),
        "fake_cti": (
            "Supply-chain compromise of payment-gateway fraud-detection SDK. Malicious "
            "artifact hash 1a2b3c4d5e6f7890abcdef1234567890abcdef12 deployed across 34 "
            "institutions. Threat actor exploited CVE-2023-44487 for detection evasion."
        ),
        "topic": "payment-gateway",
        "target_relevance": "high",
    },
    {
        "real_cti": (
            "Ransomware group FIN12 targeted two wealth-management firms using a "
            "spearphishing attachment exploiting CVE-2024-21413 in Outlook. The dropper "
            "communicates with 203.0.113.77. SHA256 of payload: "
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef."
        ),
        "fake_cti": (
            "FIN12 ransomware campaign against wealth-management sector. CVE-2024-21413 "
            "Outlook exploit drops payload (sha256 "
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef). "
            "C2 at 203.0.113.77. Offline backups critical."
        ),
        "topic": "ransomware",
        "target_relevance": "medium",
    },
    {
        "real_cti": (
            "Zero-day vulnerability in a widely-used banking API gateway (CVE-2025-0001) "
            "allows unauthenticated remote code execution. Exploit code observed in the wild "
            "from source IP 198.51.100.22. Vendor patch not yet available; apply network "
            "segmentation and disable affected endpoint immediately."
        ),
        "fake_cti": (
            "Critical zero-day CVE-2025-0001 in banking API gateway under active exploitation "
            "by nation-state actor. Attack origin 198.51.100.22. No patch available; "
            "immediate isolation required for finance-sector targets."
        ),
        "topic": "zero-day-banking",
        "target_relevance": "high",
    },
    {
        "real_cti": (
            "A financially motivated threat actor compromised an open-source financial "
            "reporting library used across dozens of insurance portals. The malicious version "
            "contained a backdoor checksum "
            "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20. "
            "No CVE assigned yet; affected versions 2.3.0–2.3.4. Downgrade to 2.2.9."
        ),
        "fake_cti": (
            "Backdoored open-source reporting library impacts insurance portals. Malicious "
            "build hash 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20 "
            "in versions 2.3.0-2.3.4. No CVE yet. Downgrade to 2.2.9 immediately."
        ),
        "topic": "supply-chain",
        "target_relevance": "medium",
    },
]


def make_synthetic_sample() -> List[Dict[str, Any]]:
    """Return normalized records for the inline synthetic placeholder rows."""
    out = []
    for row in _SYNTHETIC_ROWS:
        rec = normalize_row(row)
        if rec:
            rec["metadata"]["synthetic"] = True  # flag so consumers can filter
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# OSINT augmentation stub
# ---------------------------------------------------------------------------

def _merge_osint_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge ~5000 finance-domain OSINT samples (Section 3.2) — stub.

    TODO(Section 3.2): When the OSINT augmentation corpus is acquired (FS-ISAC feeds,
    OTX finance-tagged pulses, NVD finance-sector CVE advisories, vendor security bulletins),
    normalize each item to the GFCTI-Finance schema above and append here before writing OUT.
    Expected merge point: call this function just before writing the output file in ``main()``.
    The OSINT samples should NOT be fabricated — they must come from real public sources.
    """
    return records  # no-op until corpus is acquired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_gfcti(
    src: Path = DEFAULT_SRC,
    out: Path = DEFAULT_OUT,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize GFCTI source to GFCTI-Finance JSONL; return the records written.

    If *src* is absent and *strict* is False, emits a synthetic sample and warns.
    """
    records: List[Dict[str, Any]] = []

    if not src.is_file():
        msg = (
            f"\n[GFCTI] Source file not found: {src}\n"
            f"  To use real GFCTI data, place your GFCTI file at:\n"
            f"    {src.resolve()}\n"
            f"  Expected columns (variants accepted):\n"
            f"    real_cti / real / genuine      — genuine CTI text\n"
            f"    fake_cti / fake / generated    — LLM-generated fake CTI text\n"
            f"    topic / category               — topic label\n"
            f"    relevance / target_relevance   — low | medium | high\n"
            f"  Supported formats: JSONL (.jsonl) or CSV (.csv)\n"
        )
        print(msg, file=sys.stderr)
        if strict:
            sys.exit(1)
        print("", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("WARNING: GFCTI source absent — using SYNTHETIC PLACEHOLDER (5 rows).", file=sys.stderr)
        print("  All records are marked metadata.synthetic=True.", file=sys.stderr)
        print("  Results derived from this data are NOT representative of real CTI.", file=sys.stderr)
        print("  Provide the real GFCTI source file for valid experiments.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("", file=sys.stderr)
        records = make_synthetic_sample()
    else:
        raw_rows = load_source(src)
        print(f"[GFCTI] Loaded {len(raw_rows)} raw rows from {src}")
        for row in raw_rows:
            rec = normalize_row(row)
            if rec is not None:
                records.append(rec)
        print(f"[GFCTI] Normalized {len(records)} valid records (skipped {len(raw_rows) - len(records)} degenerate rows)")

    # Merge OSINT augmentation samples (no-op stub until corpus is acquired)
    records = _merge_osint_samples(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[GFCTI] Wrote {len(records)} records -> {out}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GFCTI-Finance JSONL dataset.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Path to raw GFCTI source file (JSONL or CSV).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSONL path.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if source file is missing (no synthetic fallback).")
    args = parser.parse_args()
    build_gfcti(src=args.src, out=args.out, strict=args.strict)


if __name__ == "__main__":
    main()
