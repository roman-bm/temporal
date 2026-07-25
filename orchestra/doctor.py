"""Preflight: prove each configured model can actually participate.

A council run costs dozens of calls. Discovering there that a model ID went
stale, a base URL moved, or a key is scoped wrong is an expensive way to find
out. This sends one tiny request per model and reports exactly what broke.

It checks three things, in the order they fail in practice:

1. **Reachable** — the endpoint answered at all (catches bad base URLs, dead
   keys, network policy).
2. **Model ID valid** — the provider recognised it (catches the stale IDs this
   registry is most likely to carry, since vendors rename models constantly).
3. **Protocol-capable** — the reply contains a parseable JSON object. A model
   that cannot return JSON on request will contribute no claims and cast no
   votes, so it is worse than useless on a panel: it takes a seat and stays
   silent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .providers import extract_json
from .registry import Registry

PROBE_SYSTEM = (
    "You are responding to an automated connectivity check. "
    "Reply with a single JSON object and nothing else."
)
PROBE_USER = 'Return exactly this JSON object: {"ok": true}'

# A local egress proxy refusing CONNECT looks nothing like a provider saying
# no, but both surface as a failed request. Separating them matters: one is a
# firewall/allowlist problem on your side, the other is a key or model problem.
_BLOCKED_HINTS = (
    "proxyerror",
    "connect tunnel failed",
    "tunnel connection failed",
    "connection refused",
    "temporary failure in name resolution",
    "nodename nor servname",
    "name or service not known",
    "certificate verify failed",
)

# Substrings providers use when they don't recognise a model ID. Worth
# separating from generic failures because it points at a one-line YAML fix.
_STALE_ID_HINTS = (
    "model not found",
    "does not exist",
    "unknown model",
    "invalid model",
    "model_not_found",
    "no such model",
    "not_found_error",
    "unsupported model",
)


@dataclass
class Probe:
    key: str
    name: str
    vendor: str
    model_id: str
    status: str  # ok | no_key | blocked | unreachable | stale_model_id | no_json
    detail: str = ""
    latency_s: float = 0.0
    tokens: int = 0

    @property
    def usable(self) -> bool:
        return self.status == "ok"


@dataclass
class Preflight:
    probes: list[Probe] = field(default_factory=list)

    @property
    def usable(self) -> list[Probe]:
        return [p for p in self.probes if p.usable]

    @property
    def broken(self) -> list[Probe]:
        return [p for p in self.probes if p.status not in ("ok", "no_key")]

    @property
    def unconfigured(self) -> list[Probe]:
        return [p for p in self.probes if p.status == "no_key"]

    def can_run_a_council(self) -> bool:
        """A council needs a chair that can orchestrate plus one panelist."""
        return len(self.usable) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable": len(self.usable),
            "broken": len(self.broken),
            "unconfigured": len(self.unconfigured),
            "probes": [p.__dict__ for p in self.probes],
        }


async def preflight(registry: Registry, keys: list[str] | None = None) -> Preflight:
    """Probe every model that has a key. Cheap: ~64 output tokens each.

    Models without a key are reported as `no_key` rather than omitted — knowing
    what you could switch on is part of the answer.
    """
    targets = keys or [s.key for s in registry.all()]

    async def probe(key: str) -> Probe:
        spec = registry.get(key)
        base = Probe(key=key, name=spec.name, vendor=spec.vendor, model_id=spec.model_id,
                     status="no_key")

        if not registry.is_live(key):
            base.detail = f"{spec.api_key_env} is not set" if spec.api_key_env else "simulated"
            return base

        result = await registry.call(
            key,
            PROBE_SYSTEM,
            PROBE_USER,
            json_object=True,
            phase="preflight",
            overrides={"max_tokens": 64, "timeout_s": 30.0},
        )
        base.latency_s = round(result.latency_s, 2)
        base.tokens = result.output_tokens

        if not result.ok:
            error = (result.error or "unknown error").strip()
            lowered = error.lower()
            if any(hint in lowered for hint in _BLOCKED_HINTS):
                base.status = "blocked"
                base.detail = (
                    f"network egress to this endpoint is blocked locally, so the "
                    f"request never reached the provider — {error[:180]}"
                )
            elif any(hint in lowered for hint in _STALE_ID_HINTS):
                base.status = "stale_model_id"
                base.detail = error[:300]
            else:
                base.status = "unreachable"
                base.detail = error[:300]
            return base

        if extract_json(result.text) is None:
            base.status = "no_json"
            base.detail = (
                "answered, but not with JSON — this model will cast no votes. "
                f"Got: {result.text.strip()[:120]!r}"
            )
            return base

        base.status = "ok"
        return base

    return Preflight(probes=list(await asyncio.gather(*(probe(k) for k in targets))))
