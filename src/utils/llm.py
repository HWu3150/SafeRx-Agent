"""
Lightweight LLM client wrappers used by SafeRx-Agent.

`OpenAIChat` wraps the OpenAI chat completions API (v1 client).
`MockLLM` is a deterministic stub for tests and offline development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import asyncio
import os
from typing import Dict, Protocol, Sequence

from openai import OpenAI


def _normalize_optional_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.upper() in {"YOUR_API_KEY_HERE", "NONE", "NULL"}:
        return None
    return cleaned


def _sanitize_text(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_messages(messages: Sequence[Dict[str, str]]) -> list[Dict[str, str]]:
    cleaned: list[Dict[str, str]] = []
    for message in messages:
        sanitized: Dict[str, str] = {}
        for key, value in message.items():
            sanitized[str(key)] = _sanitize_text(str(value))
        cleaned.append(sanitized)
    if cleaned and all(m.get("role") == "system" for m in cleaned):
        cleaned[-1] = {**cleaned[-1], "role": "user"}
    return cleaned

class LLMClient(Protocol):
    """Minimal protocol for chat-based models (async)."""

    async def chat(self, messages: Sequence[Dict[str, str]], **kwargs) -> str:
        """Return assistant content given chat history."""
        ...


@dataclass
class OpenAIChat:
    """
    Thin wrapper around the OpenAI chat completions API.

    Parameters
    ----------
    model : str
        Chat model name (e.g., ``gpt-4o-mini``). Defaults to a compact model.
    api_key : str | None
        API key; when None, falls back to ``OPENAI_API_KEY`` env var.
    base_url : str | None
        Optional custom endpoint (e.g., Azure or proxy).
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.api_key = _normalize_optional_secret(self.api_key)
        self.base_url = _normalize_optional_secret(self.base_url)
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat(self, messages: Sequence[Dict[str, str]], **kwargs) -> str:
        # OpenAI Python SDK is sync; run in thread to avoid blocking event loop.
        def _call():
            response = self._client.chat.completions.create(
                model=self.model, messages=_sanitize_messages(messages), **kwargs
            )
            return _sanitize_text(response.choices[0].message.content or "")

        return await asyncio.to_thread(_call)


class MockLLM(LLMClient):
    """Deterministic echo model for tests and offline runs."""

    async def chat(self, messages: Sequence[Dict[str, str]], **kwargs) -> str:
        joined = "\n".join(m.get("content", "") for m in messages if m["role"] == "user")
        payload = {
            "mock": True,
            "echo": joined[:200],
        }
        return json.dumps(payload)


_ENV_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "vllm": None}


def load_llm_section(cfg: Dict) -> Dict:
    """Return the provider block of a config (vllm / anthropic / openai) with the API key resolved."""
    for provider in ("vllm", "anthropic", "openai"):
        section = cfg.get(provider)
        if section:
            section = dict(section)
            if not section.get("api_key"):
                env = _ENV_KEYS[provider]
                section["api_key"] = os.environ.get(env) if env else "EMPTY"
            if not section.get("api_key"):
                raise RuntimeError(f"No API key for '{provider}': set {_ENV_KEYS[provider]}.")
            return section
    raise ValueError("Config has no vllm / anthropic / openai section.")
