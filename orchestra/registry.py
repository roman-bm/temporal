"""Model registry: loads `models.yaml`, routes each model to its backend."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .providers import (
    AnthropicProvider,
    LLMResult,
    ModelSpec,
    OpenAICompatProvider,
    SimulatedProvider,
)

DEFAULT_CONFIG = Path(__file__).with_name("models.yaml")

# Headroom over a provider's own timeout before we stop waiting on it entirely.
TIMEOUT_GRACE_S = 20.0


class Registry:
    """Owns every model spec and the shared provider clients.

    A model is *live* when its `api_key_env` is present in the environment.
    Everything else falls back to the simulator, which is labelled as such all
    the way through to the UI — there is no silent substitution.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        force_simulation: bool = False,
        max_concurrency: int = 8,
    ) -> None:
        self.path = Path(config_path or DEFAULT_CONFIG)
        self.force_simulation = force_simulation
        self._specs: dict[str, ModelSpec] = {}
        self._order: list[str] = []
        self._sem = asyncio.Semaphore(max_concurrency)
        self._providers: dict[str, Any] = {
            "anthropic": AnthropicProvider(),
            "openai_compat": OpenAICompatProvider(),
            "simulated": SimulatedProvider(),
        }
        self._load()

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    def _load(self) -> None:
        data = yaml.safe_load(self.path.read_text()) or {}
        defaults = data.get("defaults") or {}
        for entry in data.get("models") or []:
            spec = ModelSpec(
                key=entry["key"],
                name=entry["name"],
                vendor=entry["vendor"],
                provider=entry["provider"],
                model_id=entry["model_id"],
                api_key_env=entry.get("api_key_env"),
                base_url=entry.get("base_url"),
                weight=float(entry.get("weight", 1.0)),
                can_orchestrate=bool(entry.get("can_orchestrate", False)),
                supports_json_mode=bool(entry.get("supports_json_mode", False)),
                effort=entry.get("effort"),
                lens=entry.get("lens", "general"),
                strengths=list(entry.get("strengths") or []),
                temperature=float(entry.get("temperature", defaults.get("temperature", 0.7))),
                max_tokens=int(entry.get("max_tokens", defaults.get("max_tokens", 4000))),
                timeout_s=float(entry.get("timeout_s", defaults.get("timeout_s", 180))),
            )
            self._specs[spec.key] = spec
            self._order.append(spec.key)

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def all(self) -> list[ModelSpec]:
        return [self._specs[k] for k in self._order]

    def get(self, key: str) -> ModelSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"unknown model '{key}'. Known: {', '.join(self._order)}") from None

    def orchestrators(self) -> list[ModelSpec]:
        return [s for s in self.all() if s.can_orchestrate]

    def is_live(self, key: str) -> bool:
        if self.force_simulation:
            return False
        spec = self.get(key)
        if spec.provider == "simulated":
            return False
        return bool(spec.api_key_env and os.environ.get(spec.api_key_env))

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "key": s.key,
                "name": s.name,
                "vendor": s.vendor,
                "lens": s.lens,
                "weight": s.weight,
                "strengths": s.strengths,
                "can_orchestrate": s.can_orchestrate,
                "live": self.is_live(s.key),
                "api_key_env": s.api_key_env,
            }
            for s in self.all()
        ]

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    async def call(
        self,
        key: str,
        system: str,
        user: str,
        *,
        json_object: bool = True,
        phase: str = "",
        context: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResult:
        spec = self.get(key)
        if overrides:
            # Used by the preflight check to probe a provider with a tiny
            # max_tokens instead of a full protocol-sized request.
            spec = replace(spec, **overrides)
        live = self.is_live(key)
        provider = self._providers[spec.provider if live else "simulated"]

        # Phases run models with asyncio.gather, so one straggler holds up the
        # whole round. Each provider sets its own timeout, but SDKs have their
        # own defaults (the Anthropic client allows 10 minutes) and a wedged
        # connection can ignore both. This is the backstop: past the grace
        # window the call becomes a normal error result and the panel moves on.
        started = time.perf_counter()
        deadline = spec.timeout_s + TIMEOUT_GRACE_S

        async with self._sem:
            try:
                result = await asyncio.wait_for(
                    provider.complete(
                        spec,
                        system,
                        user,
                        json_object=json_object,
                        phase=phase,
                        context=context,
                    ),
                    timeout=deadline,
                )
            except TimeoutError:
                return LLMResult(
                    text="",
                    model_key=key,
                    latency_s=round(time.perf_counter() - started, 2),
                    error=f"timed out after {deadline:.0f}s",
                )
            except Exception as exc:  # noqa: BLE001 - a provider bug is one voice, not the run
                return LLMResult(
                    text="",
                    model_key=key,
                    latency_s=round(time.perf_counter() - started, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )

        # A live model that errored is not silently swapped for the simulator —
        # the caller decides whether a degraded panelist is acceptable.
        return result

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
