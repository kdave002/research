"""
Token usage tracker with automatic key rotation.

Tracks cumulative token spend per API key and switches to the next key
before hitting the per-key budget cap.

Claude Haiku 4.5 pricing (as of 2025):
  Input:  $0.80 / 1M tokens
  Output: $4.00 / 1M tokens

Usage:
    from token_tracker import get_haiku_client

    client = get_haiku_client()   # always returns the active AsyncAnthropic client
    msg = await client.messages.create(...)
    record_usage(msg.usage)       # call after every API response
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pricing (per token) — Vercel AI Gateway rates for Claude Haiku 4.5
# ---------------------------------------------------------------------------
HAIKU_INPUT_COST_PER_TOKEN  = 1.00 / 1_000_000   # $1.00 per 1M input tokens
HAIKU_OUTPUT_COST_PER_TOKEN = 5.00 / 1_000_000   # $5.00 per 1M output tokens

# Switch to the next key when spend on current key exceeds this (in USD)
KEY_BUDGET_USD = float(os.environ.get("KEY_BUDGET_USD", "4.50"))

# ---------------------------------------------------------------------------
# Key pool — reads from env in order, skips missing/empty
# ---------------------------------------------------------------------------
_KEY_ENV_NAMES = [
    "VERCEL_AI_GATEWAY_KEY_1",
    "VERCEL_AI_GATEWAY_KEY_2",
    "VERCEL_AI_GATEWAY_KEY",
    "ANTHROPIC_API_KEY",
]

def _load_keys() -> list[dict]:
    keys = []
    seen = set()
    for name in _KEY_ENV_NAMES:
        val = os.environ.get(name, "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append({"env": name, "key": val})
    return keys

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"
HAIKU_MODEL      = "anthropic/claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Usage persistence
# ---------------------------------------------------------------------------
USAGE_FILE = Path("results/token_usage.json")
USAGE_FILE.parent.mkdir(exist_ok=True)


def _load_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_usage(data: dict) -> None:
    USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _key_spend(usage: dict, key_prefix: str) -> float:
    """Total USD spent on a given key (identified by first 12 chars)."""
    entry = usage.get(key_prefix, {"input_tokens": 0, "output_tokens": 0})
    return (
        entry["input_tokens"]  * HAIKU_INPUT_COST_PER_TOKEN
        + entry["output_tokens"] * HAIKU_OUTPUT_COST_PER_TOKEN
    )


# ---------------------------------------------------------------------------
# Active key state (module-level singleton)
# ---------------------------------------------------------------------------
_keys = _load_keys()
_active_idx = 0


def _active_key_info() -> dict:
    if not _keys:
        raise RuntimeError(
            "No API keys found. Set VERCEL_AI_GATEWAY_KEY_1 or ANTHROPIC_API_KEY in .env"
        )
    return _keys[_active_idx]


def _maybe_rotate() -> bool:
    """Check spend and rotate to next key if over budget. Returns True if rotated."""
    global _active_idx
    usage = _load_usage()
    info  = _active_key_info()
    prefix = info["key"][:12]
    spend  = _key_spend(usage, prefix)

    if spend >= KEY_BUDGET_USD:
        next_idx = _active_idx + 1
        if next_idx >= len(_keys):
            print(
                f"[token_tracker] WARNING: All keys have exceeded budget "
                f"(${KEY_BUDGET_USD:.2f}). Continuing on last key."
            )
            return False
        old = info["env"]
        _active_idx = next_idx
        new = _active_key_info()["env"]
        print(
            f"[token_tracker] Key {old} hit ${spend:.2f} >= ${KEY_BUDGET_USD:.2f} budget. "
            f"Switching to {new}."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_usage(usage) -> None:
    """
    Call this after every client.messages.create() call.
    `usage` is the anthropic Usage object (has .input_tokens, .output_tokens).
    """
    info   = _active_key_info()
    prefix = info["key"][:12]
    data   = _load_usage()

    entry = data.setdefault(prefix, {"input_tokens": 0, "output_tokens": 0, "env": info["env"]})
    entry["input_tokens"]  += getattr(usage, "input_tokens",  0)
    entry["output_tokens"] += getattr(usage, "output_tokens", 0)

    total_spend = _key_spend(data, prefix)
    entry["spend_usd"] = round(total_spend, 6)
    _save_usage(data)

    if total_spend >= KEY_BUDGET_USD * 0.9:  # warn at 90%
        print(
            f"[token_tracker] Key {info['env']} at ${total_spend:.2f} "
            f"({100*total_spend/KEY_BUDGET_USD:.0f}% of ${KEY_BUDGET_USD:.2f} budget)"
        )

    _maybe_rotate()


def get_haiku_client():
    """
    Returns an AsyncAnthropic client for the currently active key.
    Automatically rotates key if budget exceeded.
    Call this fresh each time rather than caching the client,
    so rotation takes effect immediately.
    """
    from anthropic import AsyncAnthropic
    _maybe_rotate()
    info = _active_key_info()

    # Vercel gateway keys start with "vck_"; Anthropic keys start with "sk-ant-"
    if info["key"].startswith("vck_"):
        return AsyncAnthropic(api_key=info["key"], base_url=GATEWAY_BASE_URL)
    else:
        return AsyncAnthropic(api_key=info["key"])


def print_status() -> None:
    """Print current spend across all keys."""
    usage = _load_usage()
    print("\n=== Token Usage Status ===")
    for info in _keys:
        prefix = info["key"][:12]
        spend  = _key_spend(usage, prefix)
        entry  = usage.get(prefix, {})
        active = " <-- ACTIVE" if info == _active_key_info() else ""
        print(
            f"  {info['env']}: "
            f"${spend:.3f} / ${KEY_BUDGET_USD:.2f}  "
            f"({entry.get('input_tokens', 0):,} in / {entry.get('output_tokens', 0):,} out){active}"
        )
    total = sum(_key_spend(usage, info["key"][:12]) for info in _keys)
    print(f"  TOTAL: ${total:.3f}")


if __name__ == "__main__":
    print_status()
    print(f"\nActive key: {_active_key_info()['env']}")
    print(f"Budget per key: ${KEY_BUDGET_USD}")
