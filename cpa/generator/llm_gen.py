"""Local HF-model inference helper for the fake-CTI generator (`backend: llm_local`).

This wires the *inference* side of the centerpiece generator: it loads a local
instruct model and samples context-aware fake-CTI variants tailored to a target.

IMPORTANT — what this is and is NOT:
  - This commit wires INFERENCE ONLY against a base/instruct model
    (default Qwen/Qwen2.5-7B-Instruct), mirroring the loading/generation pattern
    of the Kaggle `_HFProvider` cell (4-bit nf4, device_map="auto", chat template).
  - The remaining FUTURE step is QLoRA fine-tuning of Llama-3-8B on GFCTI
    (Section 2.7). Once that adapter exists, point `cfg["llm_adapter"]` at it and
    it will be loaded on top of the base model via PEFT (guarded hook below).

Sandbox-only: generated text is synthetic poison for the LOCAL mock store. This
module performs NO publishing and NO network egress beyond the (lazy, optional)
HF model download done by `transformers` itself.
"""
from __future__ import annotations

from typing import List, Tuple


class LLMBackendUnavailable(RuntimeError):
    """Raised when torch/transformers (or the model) cannot be loaded.

    The caller (FakeCTIGenerator._llm_pool) catches this and falls back to the
    deterministic template backend so offline behavior never breaks.
    """


# Module-level cache: model id (+ optional adapter) -> (model, tokenizer).
# Keeps the (slow) 4-bit load to once per process, like the Kaggle _HF_CACHE.
_MODEL_CACHE: dict = {}


def _load_model(model_id: str, adapter: str | None) -> Tuple[object, object]:
    """Lazy-load (and cache) an instruct model + tokenizer.

    Uses 4-bit nf4 quantization when bitsandbytes is importable (Kaggle GPU),
    else loads normally. Optionally stacks a PEFT adapter (the future QLoRA
    fine-tune on GFCTI) when `adapter` is given and PEFT is available.
    """
    cache_key = (model_id, adapter)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Lazy import inside try/except: missing torch/transformers => clean fallback.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # noqa: BLE001 - any import failure means "unavailable"
        raise LLMBackendUnavailable(f"torch/transformers not available: {e}") from e

    # 4-bit quant only if bitsandbytes is present; otherwise plain load.
    quant_kwargs: dict = {}
    try:
        import bitsandbytes  # noqa: F401  (presence check only)
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:  # noqa: BLE001 - no bitsandbytes -> load full precision
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", **quant_kwargs
        )
    except Exception as e:  # noqa: BLE001 - download/load failure -> fallback
        raise LLMBackendUnavailable(f"failed to load model {model_id!r}: {e}") from e

    # FUTURE HOOK: load the GFCTI QLoRA adapter on top of the base model.
    if adapter:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        except Exception as e:  # noqa: BLE001 - PEFT/adapter missing -> base model
            print(f"[llm_gen] adapter {adapter!r} not loaded ({e}); using base model")

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def _build_messages(target: dict) -> List[dict]:
    """System+user chat prompt for one fabricated, target-tailored CTI snippet."""
    profile = target.get("profile", "") or "an unspecified system"
    system = (
        "You are a synthetic data generator for a SANDBOXED cyber-threat-intel "
        "red-team study. Everything you produce is FICTIONAL and used only inside "
        "an isolated mock store. Write short, realistic-looking but entirely "
        "FABRICATED threat-intelligence snippets. Always use clearly fictional "
        "identifiers (e.g. CVE-2026-xxxx). Do NOT add disclaimers; output only the "
        "snippet text."
    )
    user = (
        "Write a single concise (2-4 sentence) fake threat-intel report that "
        f"would plausibly look relevant to this target stack: {profile}. "
        "Name the affected products/versions from that stack so it reads as "
        "directly applicable, reference a fictional CVE id of the form "
        "CVE-2026-XXXX, and recommend an urgent action. Keep it self-contained."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_fake_cti(target: dict, n: int, cfg: dict) -> List[str]:
    """Sample `n` context-aware fake-CTI strings for `target` via a local HF model.

    Raises LLMBackendUnavailable if the model stack can't be loaded; the caller
    falls back to the template backend in that case.
    """
    model_id = cfg.get("llm_model", "Qwen/Qwen2.5-7B-Instruct")
    adapter = cfg.get("llm_adapter")  # future GFCTI QLoRA adapter path (optional)
    temperature = float(cfg.get("temperature", 0.9))
    max_new_tokens = int(cfg.get("llm_max_new_tokens", 256))

    model, tokenizer = _load_model(model_id, adapter)

    # Import torch here too (model load already proved it's importable).
    import torch

    messages = _build_messages(target)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs: List[str] = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        outputs.append(text)
    return outputs
