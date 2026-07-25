"""End-to-end protocol tests, run entirely against the simulator."""

import json

import pytest

from orchestra.orchestrator import Council, CouncilConfig
from orchestra.registry import Registry
from orchestra.report import to_markdown


@pytest.fixture
def registry() -> Registry:
    return Registry(force_simulation=True)


@pytest.fixture
def config() -> CouncilConfig:
    return CouncilConfig(
        orchestrator="claude-opus-5",
        panel=["gpt-5", "gemini-2.5-pro", "grok-4", "deepseek-reasoner"],
        rounds=2,
    )


async def test_full_run_produces_a_report(registry, config):
    council = Council(registry, config)
    report = await council.run("Should we migrate the billing service to event sourcing?")

    assert report["synthesis"] is not None
    assert report["framing"]["objective"]
    assert len(report["proposals"]) == 4
    assert report["ledger"]["claims"]
    assert report["stats"]["calls"] > 0
    assert report["any_simulated"] is True


async def test_every_panelist_gets_a_brief(registry, config):
    council = Council(registry, config)
    report = await council.run("Pick a database.")
    assigned = {a["model_key"] for a in report["framing"]["assignments"]}
    assert assigned == set(config.panel)


async def test_events_cover_every_phase(registry, config):
    seen: list[dict] = []

    async def emit(event):
        seen.append(event)

    council = Council(registry, config)
    await council.run("Pick a database.", emit=emit)

    phases = {e["phase"] for e in seen if e["type"] == "phase"}
    assert {"framing", "proposal", "cross_examination", "negotiation", "synthesis"} <= phases
    assert seen[-1]["type"] == "done"
    assert all("seq" not in e or isinstance(e["seq"], int) for e in seen)


async def test_events_are_json_serialisable(registry, config):
    events: list[dict] = []

    async def emit(event):
        events.append(event)

    await Council(registry, config).run("Pick a database.", emit=emit)
    json.dumps(events)  # the SSE layer depends on this


async def test_negotiation_rounds_are_recorded(registry, config):
    council = Council(registry, config)
    report = await council.run("Pick a database.")
    rounds = report["ledger"]["rounds"]
    assert rounds, "cross-examination should always close at least one round"
    assert rounds[0]["round"] == 1
    assert all(0.0 <= r["convergence"] <= 1.0 for r in rounds)


async def test_zero_rounds_still_cross_examines(registry):
    config = CouncilConfig(orchestrator="claude-opus-5", panel=["gpt-5", "grok-4"], rounds=0)
    report = await Council(registry, config).run("Pick a database.")
    assert len(report["ledger"]["rounds"]) == 1
    assert report["synthesis"] is not None


async def test_high_target_stops_negotiation_early(registry):
    notes: list[str] = []

    async def emit(event):
        if event["type"] == "note":
            notes.append(event["message"])

    config = CouncilConfig(
        orchestrator="claude-opus-5", panel=["gpt-5", "grok-4"], rounds=4,
        convergence_target=0.0,
    )
    await Council(registry, config).run("Pick a database.", emit=emit)
    assert any("ended early" in n for n in notes)


async def test_duplicate_claims_are_deduped(registry, config):
    council = Council(registry, config)
    assert council._duplicate_of("The migration should be staged over two quarters") is False
    council.ledger.weights["gpt-5"] = 1.0
    from orchestra.schemas import ProposedClaim

    council.ledger.add_claim(
        ProposedClaim(text="The migration should be staged over two quarters"),
        author="gpt-5",
        round_no=0,
    )
    assert council._duplicate_of("the migration should be staged over two quarters") is True
    assert council._duplicate_of("Rewrite the billing service in Rust") is False


async def test_markdown_report_renders(registry, config):
    report = await Council(registry, config).run("Pick a database.")
    markdown = to_markdown(report)
    assert "# " in markdown
    assert "Simulated run" in markdown
    assert "## Claim ledger" in markdown
    assert "## Run stats" in markdown


async def test_rejects_a_non_orchestrator_chair(registry):
    with pytest.raises(ValueError, match="not configured as an orchestrator"):
        Council(registry, CouncilConfig(orchestrator="grok-4", panel=["gpt-5"]))


async def test_rejects_unknown_model(registry):
    with pytest.raises(KeyError):
        Council(registry, CouncilConfig(orchestrator="claude-opus-5", panel=["not-a-model"]))


async def test_rejects_empty_panel(registry):
    with pytest.raises(ValueError, match="at least one model"):
        Council(registry, CouncilConfig(orchestrator="claude-opus-5", panel=[]))


async def test_a_failing_panelist_does_not_abort_the_run(registry, config, monkeypatch):
    """One dead provider must degrade the run, not kill it."""
    original = registry.call

    async def flaky(key, *args, **kwargs):
        if key == "grok-4":
            from orchestra.providers import LLMResult

            return LLMResult(text="", model_key=key, error="HTTP 503: upstream down")
        return await original(key, *args, **kwargs)

    monkeypatch.setattr(registry, "call", flaky)

    report = await Council(registry, config).run("Pick a database.")
    assert report["synthesis"] is not None
    assert report["stats"]["failures"] > 0
    failed = [p for p in report["proposals"] if p["model_key"] == "grok-4"]
    assert failed and failed[0]["error"]


async def test_prose_response_is_kept_as_a_position(registry, config, monkeypatch):
    original = registry.call

    async def prose(key, *args, **kwargs):
        if kwargs.get("phase") == "proposal" and key == "gpt-5":
            from orchestra.providers import LLMResult

            return LLMResult(text="I think we should just use Postgres.", model_key=key)
        return await original(key, *args, **kwargs)

    monkeypatch.setattr(registry, "call", prose)

    report = await Council(registry, config).run("Pick a database.")
    proposal = next(p for p in report["proposals"] if p["model_key"] == "gpt-5")
    assert "Postgres" in proposal["position"]
    assert "not JSON" in proposal["error"]


async def test_registry_reports_simulation_when_forced(registry):
    assert all(m["live"] is False for m in registry.status())
    assert len(registry.all()) >= 15
    assert len(registry.orchestrators()) >= 2


# ── call budgeting ───────────────────────────────────────────────────

def test_max_calls_matches_the_protocol():
    cfg = CouncilConfig(orchestrator="claude-opus-5", panel=["gpt-5", "grok-4"], rounds=2)
    # chair: framing + 2 notes + synthesis = 4; each panelist: proposal +
    # cross-exam + 2 ballots = 4. Three participants -> 12.
    assert cfg.max_calls() == 12


def test_max_calls_ignores_a_duplicated_chair_in_the_panel():
    cfg = CouncilConfig(
        orchestrator="claude-opus-5", panel=["claude-opus-5", "gpt-5"], rounds=1
    )
    assert cfg.max_calls() == 6  # chair counted once, not twice


async def test_actual_calls_never_exceed_the_estimate(registry, config):
    """The ceiling covers protocol calls; repair retries are extra by design."""
    report = await Council(registry, config).run("Pick a database.")
    stats = report["stats"]
    assert stats["calls"] - stats["repair_retries"] <= report["config"]["max_calls"]


async def test_a_prose_reply_gets_one_repair_attempt(registry, config, monkeypatch):
    """A model answering in prose otherwise casts no votes at all."""
    from orchestra.orchestrator import REPAIR_SUFFIX
    from orchestra.providers import LLMResult

    original = registry.call
    attempts = {"n": 0}

    async def flaky(key, system, user, **kwargs):
        if key == "gpt-5" and kwargs.get("phase") == "proposal":
            attempts["n"] += 1
            if REPAIR_SUFFIX in user:
                return LLMResult(
                    text='{"position": "recovered", "claims": []}', model_key=key
                )
            return LLMResult(text="Let me think about this in prose.", model_key=key)
        return await original(key, system, user, **kwargs)

    monkeypatch.setattr(registry, "call", flaky)

    report = await Council(registry, config).run("Pick a database.")
    proposal = next(p for p in report["proposals"] if p["model_key"] == "gpt-5")
    assert attempts["n"] == 2, "should retry exactly once"
    assert proposal["position"] == "recovered"
    assert proposal["error"] is None
    assert report["stats"]["repair_retries"] >= 1


async def test_repair_gives_up_after_one_attempt(registry, config, monkeypatch):
    """Two prose replies must not loop — keep the first and move on."""
    from orchestra.providers import LLMResult

    original = registry.call
    attempts = {"n": 0}

    async def stubborn(key, system, user, **kwargs):
        if key == "gpt-5" and kwargs.get("phase") == "proposal":
            attempts["n"] += 1
            return LLMResult(text="Still prose, sorry.", model_key=key)
        return await original(key, system, user, **kwargs)

    monkeypatch.setattr(registry, "call", stubborn)

    report = await Council(registry, config).run("Pick a database.")
    assert attempts["n"] == 2
    proposal = next(p for p in report["proposals"] if p["model_key"] == "gpt-5")
    assert "not JSON" in proposal["error"]
    assert "Still prose" in proposal["position"]


async def test_no_repair_when_the_first_reply_parses(registry, config):
    report = await Council(registry, config).run("Pick a database.")
    assert report["stats"]["repair_retries"] == 0


async def test_start_event_advertises_the_ceiling(registry, config):
    seen = []

    async def emit(event):
        seen.append(event)

    await Council(registry, config).run("Pick a database.", emit=emit)
    assert seen[0]["type"] == "start"
    assert seen[0]["config"]["max_calls"] == config.max_calls()
