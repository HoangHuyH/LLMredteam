"""Build a real-CTI background corpus (JSONL) from two public datasets.

Sources (real cybersecurity text the poison must blend into):
  - CASIE          (Ebiquity/CASIE): annotated cyber-incident news articles.
  - CyEnts blog    (UMBC-Onramp/CyEnts-Cyber-Blog-Dataset): security blog paragraphs.

Output: data/real_cti_corpus.jsonl in the harness schema (one CTI record per line).
This corpus is the *genuine* CTI feed; fake CTI (generated separately) is published into
the same store and must out-rank it in retrieval to reach the victim (RQ1 realism).

    python -m data.build_corpus
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "real_cti_corpus.jsonl"

CASIE_ANN = RAW / "CASIE" / "data" / "annotation"
CYENTS_PARA = RAW / "CyEnts" / "Paragraphs"

_ws = re.compile(r"\s+")


def _clean(t: str) -> str:
    return _ws.sub(" ", t or "").strip()


def load_casie(max_chars: int = 600):
    """Each CASIE annotation JSON -> one record (title + truncated content)."""
    recs = []
    for fp in sorted(CASIE_ANN.glob("*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        info = d.get("info", {})
        title = _clean(info.get("title", ""))
        content = _clean(d.get("content", ""))
        if len(content) < 80:
            continue
        text = (title + ". " + content)[:max_chars] if title else content[:max_chars]
        recs.append({
            "topic": "incident-report",
            "real_cti": text,
            "target_relevance": "low",
            "source_channel": "seed-real",
            "metadata": {"dataset": "CASIE", "source": fp.stem,
                         "date": info.get("date", "")},
        })
    return recs


def load_cyents(min_len: int = 120, max_chars: int = 600):
    """Each CyEnts paragraph -> one record (filter out very short/noisy ones)."""
    recs = []
    for fp in sorted(CYENTS_PARA.glob("*.txt")):
        text = _clean(fp.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < min_len:
            continue
        recs.append({
            "topic": "security-blog",
            "real_cti": text[:max_chars],
            "target_relevance": "low",
            "source_channel": "seed-real",
            "metadata": {"dataset": "CyEnts", "source": fp.stem},
        })
    return recs


def main() -> None:
    casie = load_casie()
    cyents = load_cyents()
    all_recs = casie + cyents
    with OUT.open("w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"CASIE: {len(casie)} | CyEnts: {len(cyents)} | total: {len(all_recs)}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
