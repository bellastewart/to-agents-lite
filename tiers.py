"""
tiers.py — the two supported model configurations, in one place.
================================================================

Everything in the pipeline is configured through `TO_{TEXT,VISION,JUDGE}_*`
environment variables (see providers.py).  That is flexible but it is a lot of
knobs to hand a first-time visitor, so this module collapses the useful
combinations into two named presets:

    free  one Gemini key, no payment method, ~6-10 runs/day
    plus  one OpenRouter key, ~$0.002/run, no daily cap

Three roles, and in BOTH tiers they use three DIFFERENT models.  That is not
cosmetic: the judge scores designs that the vision agent described.  If the two
share weights they share blind spots, and the judge cannot catch a
misperception it would have made itself.

Why these specific models
-------------------------
free: Gemini's free tier is quota'd per MODEL (20 requests/day, quota id
      `GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so giving each role
      a different model triples the budget rather than sharing one.
      `gemma-4-31b-it` is served on the same Gemini key with its own separate
      quota, and it is the one model here actually validated on real renders
      (correct in both image orders on a pair with numeric ground truth).

plus: two of the three roles are the EXACT models the full 4-GPU build used --
      `llama-3.3-70b` for structured output and `gemma-3-27b-it` for the judge.
      Only vision is substituted, because `Qwen2.5-VL-7B` is served by exactly
      one provider anywhere (featherless-ai, via HF) and a single point of
      failure is a poor default for a public site.  `qwen3-vl-32b-instruct` is
      the same VL family, one generation newer and ~4.5x larger.

NOT VALIDATED: every model in `plus` is chosen on availability and price.  None
has been measured on this task.  Do not advertise it as "better" until it has.
"""

from __future__ import annotations

import os

__all__ = ["TIERS", "TIER_ORDER", "apply_tier", "detect_tier", "describe_tier",
           "missing_keys"]


TIERS = {
    "free": {
        "label": "Free — one Gemini key",
        "blurb": "No payment method. About 6-10 full runs per day.",
        "keys": ["GEMINI_API_KEY"],
        "key_help": {
            "GEMINI_API_KEY": "https://aistudio.google.com/apikey  (free, no card)",
        },
        "cost": "$0",
        "limit": "20 requests/day per model",
        "roles": {
            # A different model per role so the per-model quotas do not compete.
            "text":   ("gemini", "gemini-3.5-flash-lite"),
            "vision": ("gemini", "gemini-3-flash-preview"),
            "judge":  ("gemini", "gemma-4-31b-it"),
        },
    },
    "plus": {
        "label": "Plus — one OpenRouter key",
        "blurb": "Needs a payment method. ~$0.002 per run, no daily cap.",
        "keys": ["OPENROUTER_API_KEY"],
        "key_help": {
            "OPENROUTER_API_KEY": "https://openrouter.ai/keys  (requires a card)",
        },
        "cost": "~$0.002/run (~450 runs per $1)",
        "limit": "none",
        "roles": {
            "text":   ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
            "vision": ("openrouter", "qwen/qwen3-vl-32b-instruct"),
            "judge":  ("openrouter", "google/gemma-3-27b-it"),
        },
    },
}

TIER_ORDER = ["free", "plus"]

# Optional swaps a user can opt into; not defaults.
ALTERNATES = {
    # Same generation as the original Qwen2.5-VL-7B rather than a newer one --
    # preferable when reproducing published numbers over maximising capability.
    "vision_faithful": ("openrouter", "qwen/qwen2.5-vl-72b-instruct"),
    # Literally the original weights, but a second key and one provider only.
    "vision_exact":    ("hf", "Qwen/Qwen2.5-VL-7B-Instruct"),
    # Cheaper, 1M context, unmeasured.
    "vision_cheap":    ("openrouter", "qwen/qwen3.7-flash"),
}


def missing_keys(tier: str) -> list[str]:
    """Which required API keys are absent from the environment."""
    return [k for k in TIERS[tier]["keys"] if not os.environ.get(k, "").strip()]


def detect_tier() -> str | None:
    """Best tier the current environment can actually run, or None.

    `plus` wins when its key is present: someone who supplied an OpenRouter key
    did so on purpose, and it has no daily cap to run into.
    """
    for name in ("plus", "free"):
        if not missing_keys(name):
            return name
    return None


def apply_tier(tier: str, *, env: dict | None = None) -> dict:
    """Set TO_{TEXT,VISION,JUDGE}_{PROVIDER,MODEL} for `tier`.

    Writes into `env` (default os.environ) and returns just the keys it set, so
    a caller can print or log the resulting configuration.

    Base URLs are deliberately not set — providers.py already knows the right
    endpoint for each provider name, and hardcoding one here would silently
    override a user's TO_*_BASE_URL override.
    """
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; choose from {', '.join(TIER_ORDER)}")
    target = os.environ if env is None else env
    applied = {}
    for role, (provider, model) in TIERS[tier]["roles"].items():
        R = role.upper()
        applied[f"TO_{R}_PROVIDER"] = provider
        applied[f"TO_{R}_MODEL"] = model
    target.update(applied)
    return applied


def describe_tier(tier: str) -> str:
    """A short human-readable summary, for notebooks and the web UI."""
    t = TIERS[tier]
    lines = [f"{t['label']}", f"  {t['blurb']}",
             f"  cost : {t['cost']}", f"  limit: {t['limit']}"]
    for role in ("text", "vision", "judge"):
        provider, model = t["roles"][role]
        name = {"text": "structured"}.get(role, role)
        lines.append(f"  {name:10s} -> {provider}/{model}")
    missing = missing_keys(tier)
    lines.append(f"  keys : {', '.join(t['keys'])}"
                 + (f"   MISSING: {', '.join(missing)}" if missing else "   (all present)"))
    return "\n".join(lines)


if __name__ == "__main__":
    for name in TIER_ORDER:
        print(describe_tier(name))
        print()
    print("detected:", detect_tier() or "none — no usable key found")
