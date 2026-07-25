"""Provider-layer behaviour that must hold without touching a real API.

The point of these is failure handling: a dead provider, a missing key, or a
refusal must come back as an `LLMResult` with `error` set, never as a raised
exception that takes down the whole council.
"""

import httpx

from orchestra.providers import ModelSpec, OpenAICompatProvider, SimulatedProvider
from orchestra.providers.anthropic_provider import AnthropicProvider


def spec(**over) -> ModelSpec:
    base = dict(
        key="test-model",
        name="Test",
        vendor="Test Inc",
        provider="openai_compat",
        model_id="test-1",
        api_key_env="TEST_KEY",
        base_url="https://example.invalid/v1",
        lens="testing",
    )
    base.update(over)
    return ModelSpec(**base)


# ── OpenAI-compatible ────────────────────────────────────────────────

async def test_missing_key_is_an_error_not_an_exception(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    result = await OpenAICompatProvider().complete(spec(), "sys", "user")
    assert result.ok is False
    assert "TEST_KEY" in result.error


async def test_http_error_is_captured(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    provider = OpenAICompatProvider(max_retries=0)

    async def boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(provider._get_client(), "post", boom)
    result = await provider.complete(spec(), "sys", "user")
    assert result.ok is False
    assert "ConnectError" in result.error
    await provider.aclose()


async def test_successful_response_is_parsed(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    provider = OpenAICompatProvider()

    async def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"position": "yes"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(provider._get_client(), "post", fake_post)
    result = await provider.complete(spec(), "sys", "user")
    assert result.ok
    assert result.input_tokens == 10 and result.output_tokens == 5
    await provider.aclose()


async def test_block_list_content_is_flattened(monkeypatch):
    """Some gateways return content as a list of blocks rather than a string."""
    monkeypatch.setenv("TEST_KEY", "sk-test")
    provider = OpenAICompatProvider()

    async def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"text": "hel"}, {"text": "lo"}]}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(provider._get_client(), "post", fake_post)
    result = await provider.complete(spec(), "sys", "user")
    assert result.text == "hello"
    await provider.aclose()


async def test_json_mode_is_dropped_after_a_400(monkeypatch):
    """A gateway that rejects response_format must not fail the call."""
    monkeypatch.setenv("TEST_KEY", "sk-test")
    provider = OpenAICompatProvider(max_retries=1)
    seen: list[bool] = []

    async def fake_post(url, **kwargs):
        has_format = "response_format" in kwargs["json"]
        seen.append(has_format)
        if has_format:
            return httpx.Response(400, text="unsupported parameter",
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(provider._get_client(), "post", fake_post)
    result = await provider.complete(spec(supports_json_mode=True), "s", "u", json_object=True)
    assert result.ok
    assert seen == [True, False]
    await provider.aclose()


async def test_empty_completion_counts_as_an_error(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    provider = OpenAICompatProvider()

    async def fake_post(url, **kwargs):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(provider._get_client(), "post", fake_post)
    result = await provider.complete(spec(), "sys", "user")
    assert result.ok is False
    await provider.aclose()


# ── Anthropic ────────────────────────────────────────────────────────

class _FakeMessages:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text="{}", stop_reason="end_turn", category=None):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason
        self.stop_details = type("D", (), {"category": category})() if category else None
        self.usage = type("U", (), {"input_tokens": 7, "output_tokens": 3})()


def _wire(provider: AnthropicProvider, messages: _FakeMessages):
    provider._client = type("C", (), {"beta": type("B", (), {"messages": messages})()})()


async def test_anthropic_refusal_is_surfaced_not_read_as_content():
    provider = AnthropicProvider()
    messages = _FakeMessages(_FakeResponse(text="", stop_reason="refusal", category="cyber"))
    _wire(provider, messages)

    result = await provider.complete(spec(provider="anthropic", model_id="claude-opus-5"), "s", "u")
    assert result.refused is True
    assert result.ok is False
    assert "cyber" in result.error


async def test_anthropic_requests_adaptive_thinking_and_fallbacks():
    provider = AnthropicProvider()
    messages = _FakeMessages(_FakeResponse(text='{"a": 1}'))
    _wire(provider, messages)

    await provider.complete(
        spec(provider="anthropic", model_id="claude-opus-5", effort="high"), "s", "u"
    )
    sent = messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert sent["output_config"] == {"effort": "high"}
    assert sent["fallbacks"] == "default"


async def test_anthropic_retries_without_fallbacks_on_type_error():
    """An older SDK that cannot type `fallbacks` must not break the run."""
    provider = AnthropicProvider()

    class Picky(_FakeMessages):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if "fallbacks" in kwargs:
                raise TypeError("unexpected keyword argument 'fallbacks'")
            return _FakeResponse(text='{"ok": true}')

    messages = Picky()
    _wire(provider, messages)

    result = await provider.complete(spec(provider="anthropic", model_id="claude-opus-5"), "s", "u")
    assert result.ok
    assert len(messages.calls) == 2
    assert "fallbacks" not in messages.calls[1]
    assert provider._fallbacks_supported is False


async def test_anthropic_api_error_becomes_an_error_result():
    provider = AnthropicProvider()
    _wire(provider, _FakeMessages(exc=RuntimeError("upstream exploded")))
    result = await provider.complete(spec(provider="anthropic", model_id="claude-opus-5"), "s", "u")
    assert result.ok is False
    assert "upstream exploded" in result.error


# ── Simulator ────────────────────────────────────────────────────────

async def test_simulator_is_deterministic():
    provider = SimulatedProvider(latency_s=0)
    s = spec(provider="simulated")
    a = await provider.complete(s, "sys", "task text", phase="proposal")
    b = await provider.complete(s, "sys", "task text", phase="proposal")
    assert a.text == b.text
    assert a.simulated is True


async def test_simulator_varies_by_model():
    provider = SimulatedProvider(latency_s=0)
    a = await provider.complete(spec(key="m1", provider="simulated"), "s", "t", phase="proposal")
    b = await provider.complete(spec(key="m2", provider="simulated"), "s", "t", phase="proposal")
    assert a.text != b.text


async def test_simulator_marks_every_payload():
    provider = SimulatedProvider(latency_s=0)
    ctx = {
        "claims": [{"id": "C001", "text": "a claim", "author": "other"}],
        "panel": ["m1"],
        "accepted": ["x"],
        "contested": ["y"],
    }
    for phase in ("framing", "proposal", "cross_examination", "negotiation",
                  "synthesis", "chair_note"):
        result = await provider.complete(
            spec(provider="simulated"), "s", "t", phase=phase, context=ctx
        )
        assert "SIM" in result.text.upper(), f"{phase} payload is not labelled"
        assert result.simulated is True


async def test_simulator_chair_note_is_prose_not_json():
    result = await SimulatedProvider(latency_s=0).complete(
        spec(provider="simulated"), "s", "t",
        phase="chair_note", context={"claims": [{"id": "C001"}]},
    )
    assert not result.text.strip().startswith("{")
    assert "C001" in result.text


# ── timeouts ─────────────────────────────────────────────────────────

async def test_a_hanging_provider_becomes_an_error_not_a_stall(monkeypatch):
    """One wedged model must not hold a whole phase hostage."""
    import asyncio

    from orchestra import registry as registry_module
    from orchestra.registry import Registry

    monkeypatch.setattr(registry_module, "TIMEOUT_GRACE_S", 0.0)
    reg = Registry(force_simulation=True)
    reg.get("gpt-5").timeout_s = 0.05

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(reg._providers["simulated"], "complete", never_returns)

    result = await asyncio.wait_for(reg.call("gpt-5", "s", "u"), timeout=5)
    assert result.ok is False
    assert "timed out" in result.error


async def test_a_provider_that_raises_is_contained(monkeypatch):
    from orchestra.registry import Registry

    reg = Registry(force_simulation=True)

    async def explode(*args, **kwargs):
        raise RuntimeError("provider bug")

    monkeypatch.setattr(reg._providers["simulated"], "complete", explode)

    result = await reg.call("gpt-5", "s", "u")
    assert result.ok is False
    assert "provider bug" in result.error
