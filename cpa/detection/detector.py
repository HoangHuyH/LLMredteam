"""Real AI-/machine-generated CTI detectors (RoBERTa classifier + GLTR-style GPT-2 log-rank).

Replaces the keyword heuristic that made stealth / undetected-rate / detection-score metrics
meaningless. Two independent signals, plus an ensemble:

  - RoBERTa fake/AI-text classifier: a sequence-classification head trained to tell human prose
    from machine-generated text (default the OpenAI GPT-2 output detector). Returns P(machine).
  - GLTR-style GPT-2 log-rank: machine text overwhelmingly picks high-probability next tokens, so
    the fraction of tokens whose true next-token rank falls in the LM's top-k is much higher for
    generated text. We map that fraction to a probability with a fixed logistic squashing.
  - Ensemble: the mean of whichever detectors loaded successfully.

Every heavy dependency (torch/transformers) is lazy-imported inside try/except. Models are cached
at module level so repeated detector calls do not reload. A mismatched CUDA wheel raises a "no
kernel image" error on first use; we mirror the CPU-fallback probe idiom from
experiments/backends.make_store_embed_fn and rebuild on CPU so the run still completes.

`make_real_detector(cfg, kind)` returns a callable `text -> list[float]` (one prob per text), or
None if no real detector could be built — the caller then keeps its keyword fallback. The list
return type matches the make_detector_fn contract (callers use max(probs)).
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

# Module-level caches keyed by (model_id, device); a mismatched-CUDA fallback re-keys on "cpu".
_ROBERTA_CACHE: dict = {}
_GLTR_CACHE: dict = {}


def _resolve_device(cfg: dict):
    """Pick a torch device string, honouring cfg['device'] then CUDA availability."""
    import torch
    device = cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --- RoBERTa fake/AI-text classifier ----------------------------------------
def _load_roberta(cfg: dict):
    """Load (and cache) the RoBERTa detector model + tokenizer on the requested device.

    Returns (model, tokenizer, device) or None if transformers/torch is missing or load fails.
    """
    model_id = cfg.get("roberta_model", "openai-community/roberta-base-openai-detector")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None

    device = _resolve_device(cfg)
    key = (model_id, device)
    if key in _ROBERTA_CACHE:
        return _ROBERTA_CACHE[key]
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device).eval()
    except Exception as e:
        print(f"[detector] roberta load failed ({type(e).__name__}); skipping")
        return None

    bundle = (model, tok, device)
    _ROBERTA_CACHE[key] = bundle
    return bundle


def _roberta_prob(model, tok, device, text: str) -> float:
    """Return P(machine-generated) for one text under the RoBERTa classifier."""
    import torch
    enc = tok(text or "", return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # The OpenAI detector labels {0: real/human, 1: fake/machine}; if a model exposes id2label we
    # honour it, otherwise assume the last column is the machine class.
    id2label = getattr(model.config, "id2label", None) or {}
    machine_idx = probs.shape[-1] - 1
    for idx, label in id2label.items():
        if any(tag in str(label).lower() for tag in ("fake", "machine", "generated", "ai")):
            machine_idx = int(idx)
            break
    return float(probs[machine_idx].item())


def _make_roberta(cfg: dict) -> Optional[Callable[[str], List[float]]]:
    bundle = _load_roberta(cfg)
    if bundle is None:
        return None
    model, tok, device = bundle

    def detect(text: str) -> List[float]:
        try:
            return [_roberta_prob(model, tok, device, text)]
        except Exception as e:
            # Mismatched CUDA wheel surfaces here ("no kernel image"); rebuild on CPU once.
            if device != "cpu":
                print(f"[detector] roberta {device} eval failed ({type(e).__name__}); "
                      "falling back to CPU")
                cpu = _load_roberta({**cfg, "device": "cpu"})
                if cpu is not None:
                    return [_roberta_prob(cpu[0], cpu[1], cpu[2], text)]
            return [0.0]

    return detect


# --- GLTR-style GPT-2 log-rank ----------------------------------------------
def _load_gltr(cfg: dict):
    """Load (and cache) a GPT-2 LM + tokenizer for the log-rank feature.

    Returns (model, tokenizer, device) or None if transformers/torch is missing or load fails.
    """
    model_id = cfg.get("gltr_model", "gpt2")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    device = _resolve_device(cfg)
    key = (model_id, device)
    if key in _GLTR_CACHE:
        return _GLTR_CACHE[key]
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device).eval()
    except Exception as e:
        print(f"[detector] gltr/gpt2 load failed ({type(e).__name__}); skipping")
        return None

    bundle = (model, tok, device)
    _GLTR_CACHE[key] = bundle
    return bundle


def _gltr_topk_fraction(model, tok, device, text: str, top_k: int, max_tokens: int) -> float:
    """Fraction of tokens whose true next-token rank falls within the LM's top-k predictions.

    Machine text concentrates on high-probability continuations, so this fraction runs high;
    human prose is more surprising and scores lower.
    """
    import torch
    ids = tok(text or "", return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"]
    ids = ids.to(device)
    if ids.shape[-1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(ids).logits  # (1, seq, vocab)
    # Predict token t+1 from position t: compare each actual token to the rank of its logit.
    logits = logits[0, :-1, :]                       # (seq-1, vocab)
    targets = ids[0, 1:]                             # (seq-1,)
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)   # (seq-1,)
    # rank = number of vocab entries strictly more probable than the actual token.
    ranks = (logits > target_logits.unsqueeze(1)).sum(dim=1)            # (seq-1,)
    in_topk = (ranks < top_k).float().mean().item()
    return float(in_topk)


def _make_gltr(cfg: dict) -> Optional[Callable[[str], List[float]]]:
    bundle = _load_gltr(cfg)
    if bundle is None:
        return None
    model, tok, device = bundle
    top_k = int(cfg.get("gltr_top_k", 10))
    max_tokens = int(cfg.get("gltr_max_tokens", 256))
    # Logistic squashing of the top-k fraction -> probability. Centre ~0.55 (human prose typically
    # lands below, templated/generated text above) with a moderate slope; cheap and monotonic.
    midpoint = float(cfg.get("gltr_midpoint", 0.55))
    slope = float(cfg.get("gltr_slope", 12.0))

    def detect(text: str) -> List[float]:
        try:
            frac = _gltr_topk_fraction(model, tok, device, text, top_k, max_tokens)
        except Exception as e:
            if device != "cpu":
                print(f"[detector] gltr {device} eval failed ({type(e).__name__}); "
                      "falling back to CPU")
                cpu = _load_gltr({**cfg, "device": "cpu"})
                if cpu is not None:
                    frac = _gltr_topk_fraction(cpu[0], cpu[1], cpu[2], text, top_k, max_tokens)
                else:
                    return [0.0]
            else:
                return [0.0]
        prob = 1.0 / (1.0 + math.exp(-slope * (frac - midpoint)))
        return [float(prob)]

    return detect


# --- ensemble + factory -----------------------------------------------------
def _make_ensemble(cfg: dict) -> Optional[Callable[[str], List[float]]]:
    parts = [d for d in (_make_roberta(cfg), _make_gltr(cfg)) if d is not None]
    if not parts:
        return None

    def detect(text: str) -> List[float]:
        vals = [d(text)[0] for d in parts]
        return [float(sum(vals) / len(vals))]

    return detect


def make_real_detector(cfg: dict, kind: str) -> Optional[Callable[[str], List[float]]]:
    """Build a real detector of the requested kind, or None if it cannot be constructed.

    kind: "roberta" | "gltr" | "ensemble". Returns a callable text -> list[float] of
    machine-generated probabilities (one element per call), matching the make_detector_fn
    contract. Returns None when torch/transformers is missing or every model fails to load, so
    the caller can fall back to its keyword heuristic.
    """
    if kind == "roberta":
        return _make_roberta(cfg)
    if kind == "gltr":
        return _make_gltr(cfg)
    if kind == "ensemble":
        return _make_ensemble(cfg)
    raise ValueError(f"Unknown real detector kind: {kind}")
