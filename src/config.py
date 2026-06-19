from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the Day 17 lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Steps:
    1. Resolve repo root.
    2. Optionally load .env.
    3. Create state/ directory.
    4. Return populated LabConfig.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env if present
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    # Provider for main model
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "anthropic"))
    model_name = os.getenv("LLM_MODEL", _default_model(provider))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    api_key = _pick_api_key(provider)
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Provider for judge model (default same as main)
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = _pick_api_key(judge_provider)

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=base_url,
    )

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "500")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model_cfg,
        judge_model=judge_cfg,
    )


def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-haiku-4-5-20251001",
        "ollama": "llama3",
        "openrouter": "openai/gpt-4o-mini",
        "custom": "gpt-4o-mini",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _pick_api_key(provider: str) -> str | None:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "ollama": None,
    }
    env_var = key_map.get(provider)
    return os.getenv(env_var) if env_var else None