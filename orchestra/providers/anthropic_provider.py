"""Anthropic backend — official SDK, adaptive thinking, refusal fallbacks."""

from __future__ import annotations

import os
import time

from .base import LLMResult, ModelSpec

try:  # pragma: no cover - import guard
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]


# Models that accept the `effort` knob and adaptive thinking.
_ADAPTIVE = ("claude-opus-5", "claude-fable-5", "claude-mythos-5",
             "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7")


class AnthropicProvider:
    """Wraps `client.beta.messages.create`.

    Two things worth knowing about this path:

    * Adaptive thinking is on. We ask for summarized display so the UI can show
      reasoning progress instead of a long silent pause.
    * Server-side refusal fallbacks are requested by default. Opus 5 and Fable 5
      run elevated safety classifiers that occasionally decline benign
      security/life-science adjacent work; `fallbacks="default"` re-serves those
      on Anthropic's recommended substitute inside the same call. If the
      installed SDK or the account does not support the parameter we retry once
      without it rather than failing the run.
    """

    def __init__(self) -> None:
        self._client = None
        self._fallbacks_supported = True

    def _get_client(self, spec: ModelSpec):
        if anthropic is None:
            raise RuntimeError("the `anthropic` package is not installed")
        if self._client is None:
            key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY")
            # A bare constructor also resolves `ant auth login` profiles, so we
            # only pass the key when it is actually set.
            self._client = (
                anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
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
        client = self._get_client(spec)
        # The SDK defaults to a 10-minute request timeout, which is far longer
        # than a council phase should ever wait on one panelist.
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=spec.timeout_s)
        started = time.perf_counter()

        kwargs: dict = {
            "model": spec.model_id,
            "max_tokens": spec.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if spec.model_id.startswith(_ADAPTIVE):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            if spec.effort:
                kwargs["output_config"] = {"effort": spec.effort}

        response = None
        for use_fallbacks in (self._fallbacks_supported, False):
            call = dict(kwargs)
            if use_fallbacks:
                call["betas"] = ["server-side-fallback-2026-07-01"]
                call["fallbacks"] = "default"
            try:
                response = await client.beta.messages.create(**call)
                break
            except TypeError:
                # SDK too old to type `fallbacks` — drop it permanently.
                self._fallbacks_supported = False
                continue
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                if use_fallbacks and _is_bad_request(exc):
                    self._fallbacks_supported = False
                    continue
                return LLMResult(
                    text="",
                    model_key=spec.key,
                    latency_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )

        if response is None:  # pragma: no cover - both attempts exhausted
            return LLMResult(text="", model_key=spec.key, error="no response")

        # Always check stop_reason before touching content — a refusal can carry
        # an empty content array.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            return LLMResult(
                text="",
                model_key=spec.key,
                latency_s=time.perf_counter() - started,
                refused=True,
                error=f"declined by safety classifier (category: {category})",
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)

        # An empty text body with no exception is otherwise indistinguishable
        # from a transport failure — the caller sees ok=False and a blank error.
        # Say what actually came back: `max_tokens` means the budget went to
        # thinking, and thinking-only content means no answer was produced.
        error = None
        if not text.strip():
            stop = getattr(response, "stop_reason", None) or "unknown"
            blocks = sorted({getattr(b, "type", "?") for b in response.content}) or ["none"]
            error = (
                f"no text in response (stop_reason={stop}, blocks={'+'.join(blocks)}, "
                f"output_tokens={getattr(usage, 'output_tokens', 0)})"
            )
            if stop == "max_tokens":
                error += " — raise max_tokens; thinking consumed the budget"

        return LLMResult(
            text=text,
            model_key=spec.key,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=time.perf_counter() - started,
            error=error,
            truncated=getattr(response, "stop_reason", None) == "max_tokens",
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 400 or "fallback" in str(exc).lower()
