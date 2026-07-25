"""Preflight classification.

The whole value of `orchestra doctor` is telling you *which* thing is broken,
so these pin the classification rather than just "it returned something".
"""

import pytest

from orchestra.doctor import preflight
from orchestra.providers import LLMResult
from orchestra.registry import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry()


def _responder(registry, mapping):
    """Force every model live and answer per-key from `mapping`."""
    registry.is_live = lambda key: True  # type: ignore[method-assign]

    async def call(key, *args, **kwargs):
        return mapping.get(key, LLMResult(text='{"ok": true}', model_key=key, output_tokens=5))

    registry.call = call  # type: ignore[method-assign]


async def test_missing_key_is_reported_not_probed(registry):
    result = await preflight(registry, ["gpt-5"])
    probe = result.probes[0]
    assert probe.status == "no_key"
    assert "OPENAI_API_KEY" in probe.detail
    assert probe.usable is False


async def test_a_healthy_model_is_usable(registry):
    _responder(registry, {})
    result = await preflight(registry, ["gpt-5", "grok-4"])
    assert all(p.usable for p in result.probes)
    assert len(result.usable) == 2
    assert result.broken == []


async def test_a_stale_model_id_is_called_out_specifically(registry):
    """The most likely failure: vendors rename models constantly."""
    _responder(registry, {
        "gpt-5": LLMResult(
            text="", model_key="gpt-5",
            error="HTTP 404: {'error': {'message': 'The model `gpt-5` does not exist'}}",
        ),
    })
    probe = (await preflight(registry, ["gpt-5"])).probes[0]
    assert probe.status == "stale_model_id"
    assert "does not exist" in probe.detail


async def test_an_auth_failure_reads_as_unreachable_not_stale(registry):
    _responder(registry, {
        "gpt-5": LLMResult(text="", model_key="gpt-5", error="HTTP 401: invalid api key"),
    })
    probe = (await preflight(registry, ["gpt-5"])).probes[0]
    assert probe.status == "unreachable"


async def test_a_model_that_cannot_return_json_is_flagged(registry):
    """It would take a panel seat and then cast no votes — worse than absent."""
    _responder(registry, {
        "grok-4": LLMResult(text="Sure! Here you go: ok = true", model_key="grok-4"),
    })
    probe = (await preflight(registry, ["grok-4"])).probes[0]
    assert probe.status == "no_json"
    assert probe.usable is False
    assert "cast no votes" in probe.detail


async def test_council_readiness_needs_a_chair_and_a_panelist(registry):
    _responder(registry, {
        "grok-4": LLMResult(text="", model_key="grok-4", error="HTTP 500"),
        "deepseek-reasoner": LLMResult(text="", model_key="deepseek-reasoner", error="HTTP 500"),
    })
    one_working = await preflight(registry, ["gpt-5", "grok-4"])
    assert one_working.can_run_a_council() is False

    two_working = await preflight(registry, ["gpt-5", "claude-opus-5"])
    assert two_working.can_run_a_council() is True


async def test_probe_uses_a_small_token_budget(registry):
    """A connectivity check must not cost a real request."""
    seen: dict = {}
    registry.is_live = lambda key: True  # type: ignore[method-assign]

    async def call(key, *args, **kwargs):
        seen.update(kwargs.get("overrides") or {})
        return LLMResult(text='{"ok": true}', model_key=key)

    registry.call = call  # type: ignore[method-assign]
    await preflight(registry, ["gpt-5"])
    assert seen["max_tokens"] <= 64
    assert seen["timeout_s"] <= 30


async def test_result_is_json_serialisable(registry):
    import json

    _responder(registry, {})
    json.dumps((await preflight(registry, ["gpt-5"])).to_dict())
