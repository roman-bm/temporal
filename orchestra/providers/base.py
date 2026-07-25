"""Provider abstraction: one `complete()` call, many backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ModelSpec:
    key: str
    name: str
    vendor: str
    provider: str
    model_id: str
    api_key_env: str | None = None
    base_url: str | None = None
    weight: float = 1.0
    can_orchestrate: bool = False
    supports_json_mode: bool = False
    effort: str | None = None
    lens: str = "general"
    strengths: list[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout_s: float = 180.0


@dataclass(slots=True)
class LLMResult:
    text: str
    model_key: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    simulated: bool = False
    error: str | None = None
    refused: bool = False
    truncated: bool = False   # hit max_tokens: output is cut mid-structure

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class Provider(Protocol):
    """Backends implement exactly this."""

    async def complete(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        phase: str = "",
        context: dict[str, Any] | None = None,
    ) -> LLMResult: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose, in fences, or emit it bare. Trailing commas and
    smart quotes show up too. We try progressively more forgiving strategies and
    return None only when there is genuinely no object in there.
    """
    if not text:
        return None

    candidates: list[str] = []
    for match in _FENCE.findall(text):
        candidates.append(match.strip())
    candidates.append(text.strip())

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
        # Fall back to brace matching — find the outermost balanced {...}.
        span = _outermost_object(candidate)
        if span:
            parsed = _try_parse(span)
            if parsed is not None:
                return parsed
    return None


def _try_parse(raw: str) -> dict[str, Any] | None:
    for attempt in (raw, _repair(raw)):
        try:
            value = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _repair(raw: str) -> str:
    fixed = raw.replace("“", '"').replace("”", '"').replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)          # trailing commas
    fixed = re.sub(r"^\s*//.*$", "", fixed, flags=re.M)   # line comments
    return fixed


def _outermost_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
