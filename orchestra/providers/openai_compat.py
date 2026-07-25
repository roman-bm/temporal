"""Generic OpenAI-compatible `/chat/completions` backend.

Covers OpenAI, Google (via its OpenAI compatibility endpoint), xAI, DeepSeek,
Mistral, Together, DashScope, Moonshot, Cohere's compatibility endpoint, Z.ai,
Perplexity and Bedrock's OpenAI-compatible surface. One adapter, twelve vendors —
the differences between them live in `models.yaml`, not in code.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from .base import LLMResult, ModelSpec

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


class OpenAICompatProvider:
    def __init__(self, max_retries: int = 2) -> None:
        self._client: httpx.AsyncClient | None = None
        self._max_retries = max_retries
        # Endpoints that rejected response_format once won't be asked again.
        self._no_json_mode: set[str] = set()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(180.0, connect=15.0),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._client

    async def complete(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        phase: str = "",
        context: dict | None = None,
    ) -> LLMResult:
        key = os.environ.get(spec.api_key_env or "")
        if not key:
            return LLMResult(
                text="",
                model_key=spec.key,
                error=f"missing API key (set {spec.api_key_env})",
            )

        url = f"{(spec.base_url or '').rstrip('/')}/chat/completions"
        payload: dict = {
            "model": spec.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        }
        want_json = json_object and spec.supports_json_mode and url not in self._no_json_mode
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        client = self._get_client()
        last_error = "unknown error"

        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(
                    url, json=payload, headers=headers, timeout=spec.timeout_s
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                break

            if resp.status_code == 200:
                return _parse_ok(resp.json(), spec, time.perf_counter() - started)

            body = resp.text[:400]
            last_error = f"HTTP {resp.status_code}: {body}"

            # Some gateways 400 on response_format; retry once without it.
            if resp.status_code == 400 and want_json:
                self._no_json_mode.add(url)
                payload.pop("response_format", None)
                want_json = False
                continue

            if resp.status_code in _RETRYABLE and attempt < self._max_retries:
                delay = float(resp.headers.get("retry-after") or 2**attempt)
                await asyncio.sleep(min(delay, 20.0))
                continue
            break

        return LLMResult(
            text="",
            model_key=spec.key,
            latency_s=time.perf_counter() - started,
            error=last_error,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_ok(data: dict, spec: ModelSpec, latency: float) -> LLMResult:
    choices = data.get("choices") or []
    truncated = bool(choices) and choices[0].get("finish_reason") == "length"
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some gateways return content as a block list.
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        elif isinstance(content, str):
            text = content
        if not text:
            # Reasoning models sometimes leave content empty and fill this.
            text = message.get("reasoning_content") or ""

    usage = data.get("usage") or {}
    return LLMResult(
        text=text,
        model_key=spec.key,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        latency_s=latency,
        error=None if text.strip() else "empty completion",
        truncated=truncated,
    )
