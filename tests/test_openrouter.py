"""OpenRouter routing.

The point is a single key filling a whole cross-vendor panel, so what matters
is the precedence between a native key and the aggregator, and that the request
is genuinely rewritten to OpenRouter's endpoint and model id.
"""

import pytest

from orchestra.registry import OPENROUTER_BASE_URL, Registry

NATIVE = "OPENAI_API_KEY"
OR = "OPENROUTER_API_KEY"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (NATIVE, OR, "ORCHESTRA_PREFER_OPENROUTER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_no_keys_means_simulated():
    assert Registry().route("gpt-5") == "simulated"


def test_openrouter_key_alone_makes_every_model_live(monkeypatch):
    """The headline benefit: one credential, a full cross-vendor panel."""
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    assert all(registry.route(s.key) == "openrouter" for s in registry.all())
    assert all(registry.is_live(s.key) for s in registry.all())


def test_a_native_key_wins_over_openrouter(monkeypatch):
    """Direct is the vendor's own endpoint — no extra hop, usually cheaper."""
    monkeypatch.setenv(NATIVE, "sk-native")
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    assert registry.route("gpt-5") == "direct"
    assert registry.route("grok-4") == "openrouter"


def test_prefer_flag_inverts_the_precedence(monkeypatch):
    monkeypatch.setenv(NATIVE, "sk-native")
    monkeypatch.setenv(OR, "sk-or-test")
    monkeypatch.setenv("ORCHESTRA_PREFER_OPENROUTER", "1")
    assert Registry().route("gpt-5") == "openrouter"


def test_force_simulation_beats_every_key(monkeypatch):
    monkeypatch.setenv(OR, "sk-or-test")
    monkeypatch.setenv(NATIVE, "sk-native")
    assert Registry(force_simulation=True).route("gpt-5") == "simulated"


def test_a_model_without_an_openrouter_id_is_not_routed(monkeypatch):
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    registry.get("gpt-5").openrouter_id = None
    assert registry.route("gpt-5") == "simulated"


def test_the_spec_is_rewritten_to_openrouter(monkeypatch):
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    rewritten = registry._as_openrouter(registry.get("gpt-5"))
    assert rewritten.base_url == OPENROUTER_BASE_URL
    assert rewritten.model_id == "openai/gpt-5"
    assert rewritten.api_key_env == OR
    assert rewritten.provider == "openai_compat"


def test_anthropic_models_route_through_the_http_adapter(monkeypatch):
    """Via OpenRouter even Claude goes over the OpenAI-compatible path, not the SDK."""
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    rewritten = registry._as_openrouter(registry.get("claude-opus-5"))
    assert rewritten.provider == "openai_compat"
    assert rewritten.model_id == "anthropic/claude-opus-5"


def test_json_mode_is_enabled_via_openrouter(monkeypatch):
    """OpenRouter normalises to the OpenAI schema, so json mode is available
    even for models whose native API lacks it."""
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    assert registry.get("sonar-pro").supports_json_mode is False
    assert registry._as_openrouter(registry.get("sonar-pro")).supports_json_mode is True


def test_every_model_declares_an_openrouter_id():
    """A missing id silently drops that seat when running aggregator-only."""
    missing = [s.key for s in Registry().all() if not s.openrouter_id]
    assert missing == [], f"no openrouter_id for: {missing}"


def test_status_exposes_the_route(monkeypatch):
    monkeypatch.setenv(OR, "sk-or-test")
    entry = next(m for m in Registry().status() if m["key"] == "gpt-5")
    assert entry["route"] == "openrouter"
    assert entry["openrouter_id"] == "openai/gpt-5"


async def test_dispatch_actually_targets_openrouter(monkeypatch):
    """End to end: the outbound request must carry OpenRouter's url and id."""
    monkeypatch.setenv(OR, "sk-or-test")
    registry = Registry()
    seen: dict = {}

    async def capture(spec, system, user, **kwargs):
        from orchestra.providers import LLMResult

        seen["model_id"] = spec.model_id
        seen["base_url"] = spec.base_url
        return LLMResult(text='{"ok": true}', model_key=spec.key)

    monkeypatch.setattr(registry._providers["openai_compat"], "complete", capture)

    result = await registry.call("grok-4", "s", "u")
    assert result.ok
    assert seen["model_id"] == "x-ai/grok-4"
    assert seen["base_url"] == OPENROUTER_BASE_URL


async def test_attribution_headers_go_to_real_openrouter_only(monkeypatch):
    """OpenRouter asks for these; a self-hosted gateway has no use for them.

    Gated on the hostname, so a mock or LiteLLM proxy on localhost doesn't
    receive them — which also means only a targeted test can pin this.
    """
    import httpx

    from orchestra.providers import OpenAICompatProvider
    from orchestra.providers.base import ModelSpec

    monkeypatch.setenv(OR, "sk-or-test")
    provider = OpenAICompatProvider()
    captured: dict = {}

    async def fake_post(url, **kwargs):
        captured.update(kwargs["headers"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(provider._get_client(), "post", fake_post)

    spec = ModelSpec(
        key="x", name="X", vendor="V", provider="openai_compat",
        model_id="openai/gpt-5", base_url=OPENROUTER_BASE_URL, api_key_env=OR,
    )
    await provider.complete(spec, "s", "u")
    assert captured["HTTP-Referer"]
    assert captured["X-Title"] == "Council"

    captured.clear()
    await provider.complete(replace_url(spec, "http://127.0.0.1:9/v1"), "s", "u")
    assert "X-Title" not in captured
    await provider.aclose()


def replace_url(spec, url):
    from dataclasses import replace

    return replace(spec, base_url=url)


def test_base_url_is_overridable_for_self_hosted_gateways(monkeypatch):
    """Same route, pointed at LiteLLM or an internal proxy."""
    monkeypatch.setenv(OR, "sk-or-test")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://gateway.internal/v1")
    registry = Registry()
    assert registry._as_openrouter(registry.get("gpt-5")).base_url == "http://gateway.internal/v1"
