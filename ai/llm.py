"""Gemini API client wrapper with retry, rate-limit backoff, and disk caching."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

from .cache import get_cache
from .config import GEMINI_MODEL


class LLMUnavailableError(RuntimeError):
    """Raised when Gemini API cannot be reached or GEMINI_API_KEY is not set."""


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise LLMUnavailableError(
            "GEMINI_API_KEY is not configured. Add it to .env before calling LLM functions."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise LLMUnavailableError(
            "google-genai package is not installed. Run: pip install -r ai/requirements.txt"
        ) from exc
    return genai.Client(api_key=api_key)


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def call_gemini(
    prompt: str,
    *,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    cache_namespace: str | None = None,
    cache_key: str | None = None,
    max_retries: int = 4,
    model: str | None = None,
) -> str:
    """Invoke Gemini with caching and exponential backoff on rate limits."""
    cache = get_cache()
    effective_key = cache_key or f"{model or GEMINI_MODEL}:{system_instruction or ''}:{prompt}"
    if cache_namespace:
        cached = cache.get(cache_namespace, effective_key)
        if cached is not None:
            return str(cached)

    client = _get_client()
    from google.genai import types

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model or GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Empty response returned by Gemini API.")
            if cache_namespace:
                cache.set(cache_namespace, effective_key, text)
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            is_retryable = any(
                token in msg for token in ("429", "resource_exhausted", "rate", "503", "500", "timeout")
            )
            if not is_retryable or attempt == max_retries - 1:
                break
            delay = (1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
            time.sleep(delay)

    raise LLMUnavailableError(f"Gemini request failed after {max_retries} attempts: {last_exc}")


def call_gemini_json(
    prompt: str,
    *,
    system_instruction: str | None = None,
    cache_namespace: str | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Call Gemini expecting a JSON object response and parse it safely."""
    raw = call_gemini(
        prompt,
        system_instruction=system_instruction,
        response_mime_type="application/json",
        cache_namespace=cache_namespace,
        cache_key=cache_key,
    )
    cleaned = _strip_json_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON from Gemini: {cleaned[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from Gemini.")
    return parsed
