"""LLM provider layer for the victim — multi-provider, normalized to text.

Adapted from DREAM's `evaluation/model_api/` adapter pattern (BaseAPI + per-provider
subclasses), simplified to a single `complete(system, user) -> str` contract since our
victim emits JSON-in-text rather than using native tool-calling.

Providers:
  - anthropic            : native Anthropic SDK (Claude).
  - openai | openrouter  : OpenAI-compatible client. OpenRouter (default base_url) reaches
                           Claude/Gemini/Qwen/DeepSeek by model name — same as DREAM.
  - local                : OpenAI-compatible client against a local server (LM Studio/Ollama).

All providers retry transiently. API keys come from env vars named in config.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    def __init__(self, cfg: dict) -> None:
        self.model = cfg.get("model", "")
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 2048)
        self.max_retries = cfg.get("max_retries", 6)

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a (system, user) pair."""

    def _retry(self, fn):
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — provider SDKs raise varied errors
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last}")


class AnthropicProvider(LLMProvider):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        import anthropic
        key = os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not key:
            raise RuntimeError(f"Missing {cfg.get('api_key_env', 'ANTHROPIC_API_KEY')}")
        self.client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:
        def _call():
            resp = self.client.messages.create(
                model=self.model,
                system=system,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return self._retry(_call)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat client. Covers OpenAI, OpenRouter, and local servers."""

    DEFAULT_BASE_URLS = {
        "openai": None,                               # SDK default
        "openrouter": "https://openrouter.ai/api/v1",
        "local": "http://localhost:1234/v1",          # LM Studio default; override in cfg
    }

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        from openai import OpenAI
        provider = cfg.get("provider", "openai")
        base_url = cfg.get("base_url", self.DEFAULT_BASE_URLS.get(provider))
        key_env = cfg.get("api_key_env") or {
            "openrouter": "OPENROUTER_API_KEY",
            "local": "LOCAL_API_KEY",
        }.get(provider, "OPENAI_API_KEY")
        api_key = os.environ.get(key_env, "not-needed" if provider == "local" else None)
        if not api_key:
            raise RuntimeError(f"Missing {key_env}")
        self.client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        def _call():
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return completion.choices[0].message.content or ""
        return self._retry(_call)


class HFProvider(LLMProvider):
    """In-process HuggingFace transformers provider (4-bit quantized via bitsandbytes).

    Loads the model once per process (class-level cache) so multiple episodes sharing
    the same model name don't reload weights. Requires: transformers + bitsandbytes.
    """

    _model_cache: dict = {}

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.model_name = cfg.get("model") or "Qwen/Qwen2.5-7B-Instruct"
        if self.model_name not in self.__class__._model_cache:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError(
                    f"HFProvider requires transformers + bitsandbytes: pip install transformers bitsandbytes ({e})"
                ) from e
            import torch as _torch
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=_torch.float16)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=bnb, device_map="auto"
            )
            self.__class__._model_cache[self.model_name] = (mdl, tok)
        self._model, self._tokenizer = self.__class__._model_cache[self.model_name]

    def complete(self, system: str, user: str) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        kw = dict(max_new_tokens=self.max_tokens, pad_token_id=self._tokenizer.eos_token_id)
        if self.temperature and self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature)
        else:
            kw["do_sample"] = False
        with torch.no_grad():
            out = self._model.generate(**inputs, **kw)
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


def make_provider(cfg: dict) -> LLMProvider:
    """Factory: cfg.provider -> provider instance."""
    provider = cfg.get("provider", "anthropic")
    if provider == "anthropic":
        return AnthropicProvider(cfg)
    if provider in ("openai", "openrouter", "local"):
        return OpenAICompatibleProvider(cfg)
    if provider in ("hf", "transformers", "local-hf"):
        return HFProvider(cfg)
    raise ValueError(f"Unknown victim provider: {provider!r}")
