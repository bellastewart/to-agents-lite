"""
providers.py — one interface, many model backends.
==================================================

The full-fat build pins three models to local hardware:

    :8000  Qwen/Qwen2.5-VL-7B-Instruct   (vision)  CUDA_VISIBLE_DEVICES=3   1 GPU
    :8001  google/gemma-3-27b-it  TP=2   (judge)   CUDA_VISIBLE_DEVICES=1,2 2 GPUs
           pyFANTOM                      (solver)  CUDA_VISIBLE_DEVICES=0   1 GPU

Someone who lands on the demo has none of that.  This module removes the
*binding* between an agent role and a specific host, so each role can be
pointed at whatever the visitor actually has: a local vLLM, a hosted API, or
a small CPU model.

Three roles, because that is what the pipeline actually calls:

    text    - pydantic_agent / revise_agent  (NL -> JSON, JSON -> revised JSON)
    vision  - vllm_agent                     (look at rendered depth/stress PNGs)
    judge   - ai_judge                       (score candidate designs from images)

Configure each independently with env vars; anything unset falls back to the
notebook's local-vLLM defaults, so this file is a no-op for the existing setup.

    TO_TEXT_PROVIDER    vllm | together | openai | gemini | anthropic | ollama | hf
    TO_TEXT_MODEL       model id for that provider
    TO_TEXT_BASE_URL    override endpoint (OpenAI-compatible providers only)
    TO_VISION_PROVIDER  / TO_VISION_MODEL  / TO_VISION_BASE_URL
    TO_JUDGE_PROVIDER   / TO_JUDGE_MODEL   / TO_JUDGE_BASE_URL

API keys come from the usual names: TOGETHER_API_KEY (or Llama4_together),
OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN.

Every backend exposes the same two calls:

    backend.complete(prompt, system=..., temperature=..., max_tokens=...) -> str
    backend.look(prompt, images=[Path|bytes], system=..., ...)            -> str

`look` raises NotSupported on a text-only backend, so callers can degrade
gracefully instead of silently producing garbage.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "NotSupported", "ProviderError", "Backend",
    "get_text_backend", "get_vision_backend", "get_judge_backend",
    "describe_config", "PROVIDERS",
]


class ProviderError(RuntimeError):
    """Backend is configured but the call failed."""


class NotSupported(ProviderError):
    """Backend cannot do what was asked (e.g. images on a text-only model)."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _b64_image(img) -> tuple[str, str]:
    """Return (mime_type, base64_data) for a path or raw bytes."""
    if isinstance(img, (bytes, bytearray)):
        return "image/png", base64.b64encode(bytes(img)).decode()
    p = Path(img)
    if not p.is_file():
        raise ProviderError(f"image not found: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return mime, base64.b64encode(p.read_bytes()).decode()


def _first_env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class Backend:
    """Base class. Subclasses implement _complete / _look."""

    provider = "?"
    supports_vision = False

    def __init__(self, model: str, base_url: str = "", api_key: str = "", **kw):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.extra = kw

    # -- public ------------------------------------------------------------ #
    def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096) -> str:
        return self._complete(prompt, system, temperature, max_tokens)

    def look(self, prompt, images: Sequence = (), system=None,
             temperature=0.0, max_tokens=4096) -> str:
        if not self.supports_vision:
            raise NotSupported(
                f"{self.provider}/{self.model} has no image input. "
                f"Set TO_VISION_PROVIDER / TO_VISION_MODEL to a vision-capable model."
            )
        return self._look(prompt, list(images), system, temperature, max_tokens)

    # -- to override -------------------------------------------------------- #
    def _complete(self, prompt, system, temperature, max_tokens) -> str:
        raise NotImplementedError

    def _look(self, prompt, images, system, temperature, max_tokens) -> str:
        raise NotImplementedError

    def __repr__(self):
        loc = self.base_url or "(hosted)"
        return f"<{self.provider}:{self.model} @ {loc}>"


class OpenAICompatBackend(Backend):
    """Anything speaking the OpenAI chat-completions schema.

    Covers local vLLM, Together, OpenAI, Ollama (/v1), and the HF router —
    which is most of the useful universe.
    """

    supports_vision = True   # per-model in practice; probe with doctor.py

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("the `openai` package is required") from e
        return OpenAI(api_key=self.api_key or "NULL", base_url=self.base_url or None)

    def _send(self, content, system, temperature, max_tokens) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        try:
            r = self._client().chat.completions.create(
                model=self.model, messages=msgs,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            raise ProviderError(f"{self.provider}/{self.model}: {e}") from e
        return (r.choices[0].message.content or "").strip()

    def _complete(self, prompt, system, temperature, max_tokens):
        return self._send(prompt, system, temperature, max_tokens)

    def _look(self, prompt, images, system, temperature, max_tokens):
        parts = [{"type": "text", "text": prompt}]
        for img in images:
            mime, data = _b64_image(img)
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64,{data}"}})
        return self._send(parts, system, temperature, max_tokens)


class GeminiBackend(Backend):
    """Google AI Studio generateContent REST API.

    Gemini 3.x spends "thinking" tokens out of the SAME maxOutputTokens budget
    as the visible answer. Measured on a real render at maxOutputTokens=220:
    207 thinking tokens, **9** output tokens, finishReason=MAX_TOKENS — i.e. a
    sentence fragment, returned with no error. At 2000: ~810 thinking, ~76
    output, finishReason=STOP.

    A judge that silently returns a fragment is worse than one that fails, so
    MAX_TOKENS is raised as an error rather than passed through. Cap the
    reasoning instead with TO_GEMINI_THINKING_BUDGET if you need tight limits.
    """

    provider = "gemini"
    supports_vision = True
    ROOT = "https://generativelanguage.googleapis.com/v1beta"
    # Floor well above the thinking overhead measured above.
    MIN_OUTPUT_TOKENS = 2048

    def _post(self, parts, system, temperature, max_tokens):
        import requests
        # Silently raising a caller's cap is normally rude, but here a low cap
        # yields a truncated fragment with no signal, so treat it as a floor.
        effective = max(int(max_tokens), self.MIN_OUTPUT_TOKENS)
        gen = {"temperature": temperature, "maxOutputTokens": effective}
        budget = os.environ.get("TO_GEMINI_THINKING_BUDGET", "").strip()
        if budget:
            gen["thinkingConfig"] = {"thinkingBudget": int(budget)}
        body = {"contents": [{"parts": parts}], "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base_url or self.ROOT}/models/{self.model}:generateContent"
        try:
            r = requests.post(url, params={"key": self.api_key}, json=body, timeout=180)
        except Exception as e:
            raise ProviderError(f"gemini/{self.model}: {e}") from e
        if r.status_code != 200:
            raise ProviderError(f"gemini/{self.model}: HTTP {r.status_code} {r.text[:200]}")

        payload = r.json()
        try:
            cand = payload["candidates"][0]
        except (KeyError, IndexError) as e:
            raise ProviderError(
                f"gemini/{self.model}: no candidates in response {r.text[:200]}") from e

        text = "".join(p.get("text", "")
                       for p in cand.get("content", {}).get("parts", []) or []).strip()

        if cand.get("finishReason") == "MAX_TOKENS":
            usage = payload.get("usageMetadata", {})
            raise ProviderError(
                f"gemini/{self.model}: output truncated (finishReason=MAX_TOKENS). "
                f"thinking={usage.get('thoughtsTokenCount', '?')} tokens consumed "
                f"{effective} of the budget, leaving "
                f"{usage.get('candidatesTokenCount', '?')} for the answer. "
                f"Raise max_tokens or set TO_GEMINI_THINKING_BUDGET. "
                f"Partial text was: {text[:80]!r}")
        if not text:
            raise ProviderError(
                f"gemini/{self.model}: empty response "
                f"(finishReason={cand.get('finishReason')})")
        return text

    def _complete(self, prompt, system, temperature, max_tokens):
        return self._post([{"text": prompt}], system, temperature, max_tokens)

    def _look(self, prompt, images, system, temperature, max_tokens):
        parts = [{"text": prompt}]
        for img in images:
            mime, data = _b64_image(img)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        return self._post(parts, system, temperature, max_tokens)


class AnthropicBackend(Backend):
    """Anthropic Messages API."""

    provider = "anthropic"
    supports_vision = True
    ROOT = "https://api.anthropic.com/v1/messages"

    def _post(self, content, system, temperature, max_tokens):
        import requests
        body = {"model": self.model, "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}]}
        if system:
            body["system"] = system
        try:
            r = requests.post(self.base_url or self.ROOT, json=body, timeout=180,
                              headers={"x-api-key": self.api_key,
                                       "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"})
        except Exception as e:
            raise ProviderError(f"anthropic/{self.model}: {e}") from e
        if r.status_code != 200:
            raise ProviderError(f"anthropic/{self.model}: HTTP {r.status_code} {r.text[:200]}")
        try:
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()
        except (KeyError, TypeError) as e:
            raise ProviderError(f"anthropic/{self.model}: unexpected response {r.text[:200]}") from e

    def _complete(self, prompt, system, temperature, max_tokens):
        return self._post([{"type": "text", "text": prompt}], system, temperature, max_tokens)

    def _look(self, prompt, images, system, temperature, max_tokens):
        content = [{"type": "text", "text": prompt}]
        for img in images:
            mime, data = _b64_image(img)
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": data}})
        return self._post(content, system, temperature, max_tokens)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _openai_compat(provider, default_base, *key_names):
    def build(model, base_url="", **kw):
        b = OpenAICompatBackend(model, base_url or default_base,
                                _first_env(*key_names) or "NULL", **kw)
        b.provider = provider
        return b
    return build


PROVIDERS = {
    # local / self-hosted -- no key needed
    "vllm":   _openai_compat("vllm",   "http://localhost:8000/v1"),
    "ollama": _openai_compat("ollama", "http://localhost:11434/v1"),
    # hosted, OpenAI-compatible
    "together": _openai_compat("together", "https://api.together.xyz/v1",
                               "TOGETHER_API_KEY", "Llama4_together"),
    "openai":   _openai_compat("openai",   "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "hf":       _openai_compat("hf",       "https://router.huggingface.co/v1",
                               "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    # hosted, bespoke schema
    "gemini":    lambda model, base_url="", **kw: GeminiBackend(
        model, base_url, _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY"), **kw),
    "anthropic": lambda model, base_url="", **kw: AnthropicBackend(
        model, base_url, _first_env("ANTHROPIC_API_KEY"), **kw),
}

# Role defaults reproduce the current local setup exactly, so an unconfigured
# lite build behaves like the full build when the vLLM servers are up.
_ROLE_DEFAULTS = {
    "text":   ("vllm", "Qwen/Qwen2.5-VL-7B-Instruct", "http://localhost:8000/v1"),
    "vision": ("vllm", "Qwen/Qwen2.5-VL-7B-Instruct", "http://localhost:8000/v1"),
    "judge":  ("vllm", "google/gemma-3-27b-it",       "http://localhost:8001/v1"),
}


def _resolve(role: str) -> tuple[str, str, str]:
    d_provider, d_model, d_base = _ROLE_DEFAULTS[role]
    R = role.upper()
    provider = os.environ.get(f"TO_{R}_PROVIDER", "").strip() or d_provider
    model = os.environ.get(f"TO_{R}_MODEL", "").strip() or (
        d_model if provider == d_provider else "")
    base = os.environ.get(f"TO_{R}_BASE_URL", "").strip() or (
        d_base if provider == d_provider else "")
    if not model:
        raise ProviderError(
            f"TO_{R}_PROVIDER={provider} needs TO_{R}_MODEL to be set "
            f"(no default model is known for that provider)."
        )
    return provider, model, base


def _backend(role: str) -> Backend:
    provider, model, base = _resolve(role)
    if provider not in PROVIDERS:
        raise ProviderError(
            f"unknown provider {provider!r} for role {role!r}; "
            f"choose from: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[provider](model, base)


def get_text_backend() -> Backend:
    return _backend("text")


def get_vision_backend() -> Backend:
    return _backend("vision")


def get_judge_backend() -> Backend:
    return _backend("judge")


# --------------------------------------------------------------------------- #
# OpenAI-compatible view of a role
# --------------------------------------------------------------------------- #
# agents/vllm_agent_both.py builds its own `OpenAI(base_url=...)` client and
# posts the standard chat-completions schema with base64 `image_url` parts. It
# therefore needs no modification at all — only an OpenAI-shaped endpoint.
# Google publishes exactly such a shim for Gemini, so every provider below can
# be reached through the agent's existing code path.
_OPENAI_COMPAT_BASE = {
    "vllm":     None,                                              # uses its configured base_url
    "ollama":   "http://localhost:11434/v1",
    "together": "https://api.together.xyz/v1",
    "openai":   "https://api.openai.com/v1",
    "hf":       "https://router.huggingface.co/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai/",
}


def openai_compat(role: str) -> dict:
    """`{model, base_url, api_key}` for a role, usable as an AutoGen config_list
    entry or a raw `OpenAI(...)` client.

    Raises NotSupported for providers with no OpenAI-compatible surface, rather
    than handing back something that 404s deep inside an agent.
    """
    provider, model, base = _resolve(role)
    if provider == "anthropic":
        raise NotSupported(
            "anthropic has no OpenAI-compatible endpoint; use get_*_backend() "
            "for that role, or pick another provider.")
    if provider not in _OPENAI_COMPAT_BASE:
        raise ProviderError(f"no OpenAI-compatible base URL known for {provider!r}")

    default_base = _OPENAI_COMPAT_BASE[provider]
    base_url = base or default_base
    if not base_url:
        raise ProviderError(f"role {role!r} on provider {provider!r} needs a base_url")
    # vllm/ollama are unauthenticated; the OpenAI SDK still wants a non-empty key.
    key_lookup = {
        "together": ("TOGETHER_API_KEY", "Llama4_together"),
        "openai":   ("OPENAI_API_KEY",),
        "hf":       ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        "gemini":   ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(provider, ())
    return {"model": model, "base_url": base_url,
            "api_key": _first_env(*key_lookup) or "NULL"}


def describe_config() -> dict:
    """What each role is currently wired to (for doctor.py / the UI banner)."""
    out = {}
    for role in ("text", "vision", "judge"):
        try:
            provider, model, base = _resolve(role)
            out[role] = {"provider": provider, "model": model,
                         "base_url": base or "(provider default)"}
        except ProviderError as e:
            out[role] = {"error": str(e)}
    return out


if __name__ == "__main__":
    print(json.dumps(describe_config(), indent=2))
