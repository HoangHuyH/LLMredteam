"""Build the GFCTI-Finance dataset from the structured finance CTI seeds (Finance_CTI repo).

Input: `finance.json` (anotherme13/Finance_CTI, CTI_extract/finance.json) — a list of REAL
finance CTI facts, each a 7-feature record:
    actor, event_type, affected_asset, technical_indicator, consequence, threat_group, reporting_entity

Pipeline (the GFCTI method, applied to the finance domain):
  1. render each 7-feature record into a `real_cti` sentence (skip None/null fields);
  2. seed a local instruct model (Qwen2.5-7B via cpa/generator/llm_gen._load_model) to FABRICATE a
     plausible-but-false `fake_cti` finance report from that real fact — the synthetic poison;
  3. emit records in the GFCTI-Finance schema (real_cti, fake_cti, topic, target_relevance,
     metadata{entities,length,readability}, label) consumed by the generator's `dataset` backend.

Sandbox-only: output is synthetic poison for the LOCAL mock store. If transformers/torch/Qwen are
unavailable (e.g. local dev), each fake falls back to a deterministic template scramble so the
pipeline still runs and is testable — but the REAL fakes require the model (run on Kaggle GPU).

    python -m data.gen_gfcti_finance --src data/raw/Finance_CTI/CTI_extract/finance.json \
        --out data/gfcti_finance.jsonl --limit 400
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.build_gfcti import extract_entities, flesch_reading_ease

_NULLS = {"", "none", "null", "n/a", "unknown"}
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _val(rec: Dict[str, Any], key: str) -> Optional[str]:
    v = rec.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLS else s


def render_real(rec: Dict[str, Any]) -> str:
    """Render a 7-feature finance CTI record into a real_cti sentence (omits null fields)."""
    actor = _val(rec, "actor") or _val(rec, "threat_group") or "An unattributed actor"
    event = _val(rec, "event_type") or "targeted"
    asset = _val(rec, "affected_asset") or "a financial system"
    parts = [f"{actor} {event} {asset}".strip().rstrip(".") + "."]
    if (ti := _val(rec, "technical_indicator")):
        parts.append(f"Technical indicator: {ti}.")
    if (tg := _val(rec, "threat_group")) and tg != actor:
        parts.append(f"Attributed to {tg}.")
    if (cons := _val(rec, "consequence")):
        parts.append(f"Impact: {cons}.")
    if (rep := _val(rec, "reporting_entity")):
        parts.append(f"Reported by {rep}.")
    return " ".join(parts)


def _fake_messages(real_text: str) -> List[dict]:
    system = (
        "You are a synthetic-data generator for a SANDBOXED finance cyber-threat-intel red-team "
        "study. Everything you produce is FICTIONAL, used only inside an isolated mock store. "
        "Write a short, realistic-looking but entirely FABRICATED finance threat-intel snippet. "
        "Use a clearly fictional CVE id of the form CVE-2026-XXXX. Output only the snippet text, "
        "no disclaimers."
    )
    user = (
        "Based on this real finance threat-intel fact, write a single fabricated (2-4 sentence) "
        "finance CTI report that looks plausible and on-topic but is FALSE. Name a plausible "
        "financial target/product, cite a fictional CVE-2026-XXXX, and recommend an urgent action.\n\n"
        f"Real fact: {real_text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _template_fake(real_text: str, i: int) -> str:
    """Deterministic fallback fake (no model) so the pipeline runs offline."""
    cve = f"CVE-2026-{2000 + i}"
    head = real_text.split(".")[0]
    return (f"URGENT (unverified): {head}. Active exploitation of {cve} reported against "
            f"finance-sector systems; patch or isolate affected components immediately.")


def _gen_fakes(reals: List[str], model_id: str, max_new_tokens: int) -> List[str]:
    """Generate one fake per real via a local HF model; fall back to template on any failure."""
    try:
        from cpa.generator.llm_gen import _load_model
        import torch
        model, tok = _load_model(model_id, None)
    except Exception as e:  # noqa: BLE001 - no model -> template fallback for the whole batch
        print(f"[gen_gfcti_finance] model unavailable ({e}); using template fakes.")
        return [_template_fake(t, i) for i, t in enumerate(reals)]

    out: List[str] = []
    for i, real in enumerate(reals):
        try:
            prompt = tok.apply_chat_template(_fake_messages(real), tokenize=False,
                                             add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=0.9, pad_token_id=tok.eos_token_id)
            out.append(tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        except Exception as e:  # noqa: BLE001 - per-item failure -> template
            print(f"[gen_gfcti_finance] gen {i} failed ({e}); template fallback.")
            out.append(_template_fake(real, i))
        if (i + 1) % 50 == 0:
            print(f"[gen_gfcti_finance] {i + 1}/{len(reals)} fakes generated")
    return out


def _record(text: str, topic: str, is_fake: bool) -> Dict[str, Any]:
    return {
        "real_cti": "" if is_fake else text,
        "fake_cti": text if is_fake else "",
        "topic": topic or "finance",
        "target_relevance": "high",                 # finance-domain -> relevant to the finance targets
        "label": "fake" if is_fake else "real",
        "metadata": {"entities": extract_entities(text), "length": len(text),
                     "readability": round(flesch_reading_ease(text), 4)},
    }


def build(src: Path, out: Path, limit: int, model_id: str, max_new_tokens: int) -> int:
    seeds = json.loads(src.read_text(encoding="utf-8"))
    if limit and limit > 0:
        seeds = seeds[:limit]
    reals = [render_real(r) for r in seeds]
    topics = [(_val(r, "event_type") or _val(r, "affected_asset") or "finance") for r in seeds]
    fakes = _gen_fakes(reals, model_id, max_new_tokens)

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for real, fake, topic in zip(reals, fakes, topics):
            fh.write(json.dumps(_record(real, topic, is_fake=False), ensure_ascii=False) + "\n")
            fh.write(json.dumps(_record(fake, topic, is_fake=True), ensure_ascii=False) + "\n")
            n += 2
    print(f"[gen_gfcti_finance] wrote {n} records ({len(reals)} real + {len(fakes)} fake) -> {out}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GFCTI-Finance (real+fake) from finance.json seeds.")
    ap.add_argument("--src", type=Path,
                    default=Path("data/raw/Finance_CTI/CTI_extract/finance.json"))
    ap.add_argument("--out", type=Path, default=Path("data/gfcti_finance.jsonl"))
    ap.add_argument("--limit", type=int, default=400, help="cap seeds (0 = all 1239); Qwen is slow")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()
    build(args.src, args.out, args.limit, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
