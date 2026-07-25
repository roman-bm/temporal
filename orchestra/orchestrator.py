"""The council engine: framing -> proposals -> cross-exam -> negotiation -> synthesis."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from . import protocol
from .consensus import Ledger
from .providers import LLMResult, extract_json
from .registry import Registry
from .schemas import (
    Ballot,
    Claim,
    Framing,
    Phase,
    Proposal,
    ProposedClaim,
    Synthesis,
    Vote,
)

Emit = Callable[[dict[str, Any]], Awaitable[None]]

MAX_CLAIMS_PER_MODEL = 5
DEDUPE_THRESHOLD = 0.82

REPAIR_SUFFIX = (
    "\n\n---\nYour previous reply could not be parsed as JSON. Return ONLY the "
    "JSON object described above: no prose before or after it, no markdown "
    "fences, no explanation. Begin your reply with { and end it with }."
)

# Synthesis emits the largest structured payload of any phase — headline,
# analysis, findings, an action plan, a dissent register — and on a thinking
# model the same budget also has to cover the reasoning. At the default 4000
# it truncates mid-JSON, which reads downstream as "the model replied in prose"
# and silently costs you the entire action plan. Found on the first live run.
PHASE_TOKEN_SCALE = {"synthesis": 4, "framing": 2}
TRUNCATION_ESCALATION = 2
MAX_TOKEN_CEILING = 32000


@dataclass
class CouncilConfig:
    orchestrator: str
    panel: list[str]
    rounds: int = 2
    convergence_target: float = 0.78
    include_orchestrator_in_panel: bool = False

    def validate(self, registry: Registry) -> None:
        spec = registry.get(self.orchestrator)
        if not spec.can_orchestrate:
            raise ValueError(
                f"{spec.name} is not configured as an orchestrator "
                "(set can_orchestrate: true in models.yaml to allow it)"
            )
        if not self.panel:
            raise ValueError("the panel must contain at least one model")
        for key in self.panel:
            registry.get(key)
        if self.rounds < 0 or self.rounds > 6:
            raise ValueError("rounds must be between 0 and 6")

    def max_calls(self) -> int:
        """Upper bound on paid API calls for this configuration.

        The chair spends 2 + rounds (framing, one steering note per round,
        synthesis); each panelist spends 2 + rounds (proposal, cross-exam, one
        ballot per round). Negotiation can stop early on convergence, so this is
        a ceiling — worth showing before someone points fifteen models at a
        five-round run and finds out afterwards.

        Excludes JSON repair retries, which are at most one per call and only
        happen when a reply was already unusable. `stats.repair_retries`
        reports how many actually fired.
        """
        panel = set(self.panel)
        if not self.include_orchestrator_in_panel:
            panel.discard(self.orchestrator)
        return (2 + self.rounds) * (1 + len(panel))


@dataclass
class RunStats:
    calls: int = 0
    failures: int = 0
    simulated: int = 0
    repairs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, result: LLMResult) -> None:
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        if result.simulated:
            self.simulated += 1
        if not result.ok:
            self.failures += 1
        slot = self.per_model.setdefault(
            result.model_key,
            {"calls": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
        )
        slot["calls"] += 1
        slot["failures"] += 0 if result.ok else 1
        slot["input_tokens"] += result.input_tokens
        slot["output_tokens"] += result.output_tokens
        slot["latency_s"] = round(slot["latency_s"] + result.latency_s, 2)


class Council:
    """Runs one negotiation to completion, streaming events as it goes."""

    def __init__(self, registry: Registry, config: CouncilConfig) -> None:
        config.validate(registry)
        self.registry = registry
        self.config = config
        panel = list(config.panel)
        if config.include_orchestrator_in_panel and config.orchestrator not in panel:
            panel.append(config.orchestrator)
        self.panel_keys = [k for k in panel if k != config.orchestrator or config.include_orchestrator_in_panel]
        self.ledger = Ledger(
            weights={k: registry.get(k).weight for k in self.panel_keys}
        )
        self.stats = RunStats()
        self.framing: Framing | None = None
        self.proposals: list[Proposal] = []
        self.synthesis: Synthesis | None = None
        self._chair_notes: list[str] = []

    # ------------------------------------------------------------------

    async def run(self, task: str, context: str = "", emit: Emit | None = None) -> dict[str, Any]:
        emit = emit or _noop
        started = time.perf_counter()

        await emit({"type": "start", "config": {
            "orchestrator": self.config.orchestrator,
            "panel": self.panel_keys,
            "rounds": self.config.rounds,
            "convergence_target": self.config.convergence_target,
            "max_calls": self.config.max_calls(),
            "live_models": [k for k in [self.config.orchestrator, *self.panel_keys]
                            if self.registry.is_live(k)],
        }})

        await self._phase_framing(task, context, emit)
        await self._phase_proposals(task, context, emit)
        await self._phase_cross_exam(task, emit)
        await self._phase_negotiation(task, emit)
        await self._phase_synthesis(task, context, emit)

        self.stats.wall_s = round(time.perf_counter() - started, 2)
        report = self.report(task, context)
        await emit({"type": "done", "report": report})
        return report

    # ------------------------------------------------------------------
    # Phase 1 — framing
    # ------------------------------------------------------------------

    async def _phase_framing(self, task: str, context: str, emit: Emit) -> None:
        await emit({"type": "phase", "phase": Phase.FRAMING.value})
        orch = self.registry.get(self.config.orchestrator)
        panel_specs = [self.registry.get(k) for k in self.panel_keys]

        result = await self._call(
            self.config.orchestrator,
            protocol.framing_system(orch, panel_specs),
            protocol.framing_user(task, context),
            phase="framing",
            context={
                "task": task,
                "panel": self.panel_keys,
                "lenses": {s.key: s.lens for s in panel_specs},
            },
            emit=emit,
        )

        data = extract_json(result.text) or {}
        try:
            framing = Framing(**data)
        except Exception:  # noqa: BLE001 - a malformed chair must not stop the run
            framing = Framing(objective=task.strip()[:500])

        if not framing.objective:
            framing.objective = task.strip()[:500]

        # Guarantee every panelist has a brief, even if the chair forgot one.
        assigned = {a.model_key for a in framing.assignments}
        for spec in panel_specs:
            if spec.key not in assigned:
                framing.assignments.append(
                    protocol_assignment(spec.key, spec.lens)
                )
        framing.assignments = [a for a in framing.assignments if a.model_key in set(self.panel_keys)]

        self.framing = framing
        await emit({"type": "framing", "data": framing.model_dump(), "simulated": result.simulated})

    # ------------------------------------------------------------------
    # Phase 2 — proposals
    # ------------------------------------------------------------------

    async def _phase_proposals(self, task: str, context: str, emit: Emit) -> None:
        await emit({"type": "phase", "phase": Phase.PROPOSAL.value})
        assert self.framing is not None
        framing_dump = self.framing.model_dump()
        briefs = {a.model_key: a for a in self.framing.assignments}

        async def one(key: str) -> Proposal:
            spec = self.registry.get(key)
            assignment = briefs.get(key)
            role = assignment.role if assignment else f"{spec.lens} analyst"
            brief = assignment.brief if assignment else f"Analyse through the {spec.lens} lens."
            result = await self._call(
                key,
                protocol.proposal_system(spec, role, brief),
                protocol.proposal_user(task, context, framing_dump),
                phase="proposal",
                context={"task": task, "role": role},
                emit=emit,
            )
            return self._parse_proposal(key, result)

        self.proposals = list(await asyncio.gather(*(one(k) for k in self.panel_keys)))

        for proposal in self.proposals:
            await emit({"type": "proposal", "data": proposal.model_dump()})
            for claim in proposal.claims[:MAX_CLAIMS_PER_MODEL]:
                if not claim.text.strip():
                    continue
                if self._duplicate_of(claim.text):
                    continue
                self.ledger.add_claim(claim, author=proposal.model_key, round_no=0)

        await emit({"type": "ledger", "data": self.ledger.snapshot()})

    def _parse_proposal(self, key: str, result: LLMResult) -> Proposal:
        if not result.ok:
            return Proposal(
                model_key=key,
                error=result.error or "no output",
                simulated=result.simulated,
                raw=result.text[:2000] or None,
            )
        data = extract_json(result.text)
        if data is None:
            # The model answered in prose. Keep the position, skip the claims —
            # inventing claim boundaries on its behalf would misattribute them.
            return Proposal(
                model_key=key,
                position=result.text.strip()[:2000],
                raw=result.text[:4000],
                error="response was not JSON; claims could not be extracted",
                simulated=result.simulated,
            )
        try:
            proposal = Proposal(model_key=key, simulated=result.simulated, **_only(data, Proposal))
        except Exception as exc:  # noqa: BLE001
            return Proposal(
                model_key=key,
                position=str(data.get("position", ""))[:2000],
                error=f"schema mismatch: {exc}",
                raw=result.text[:4000],
                simulated=result.simulated,
            )
        return proposal

    # ------------------------------------------------------------------
    # Phase 3 — cross-examination
    # ------------------------------------------------------------------

    async def _phase_cross_exam(self, task: str, emit: Emit) -> None:
        await emit({"type": "phase", "phase": Phase.CROSS_EXAM.value})
        claims = list(self.ledger.claims.values())
        if not claims:
            await emit({"type": "warning", "message": "no claims were produced; skipping cross-examination"})
            return

        before = {cid: c.status for cid, c in self.ledger.claims.items()}
        ballots = await self._collect_ballots(
            task,
            claims,
            emit,
            phase="cross_examination",
            system=lambda spec: protocol.cross_exam_system(spec),
            user=lambda spec: protocol.cross_exam_user(task, claims, spec.key),
        )
        self._apply_ballots(ballots, round_no=1)
        summary = self.ledger.close_round(1, before)
        await emit({"type": "round", "data": summary.model_dump()})
        await emit({"type": "ledger", "data": self.ledger.snapshot()})

    # ------------------------------------------------------------------
    # Phase 4 — negotiation
    # ------------------------------------------------------------------

    async def _phase_negotiation(self, task: str, emit: Emit) -> None:
        await emit({"type": "phase", "phase": Phase.NEGOTIATION.value})
        for i in range(self.config.rounds):
            round_no = i + 2
            contested = self.ledger.contested()
            convergence = self.ledger.convergence()

            if not contested:
                await emit({"type": "note", "message": "ledger fully settled; negotiation ended early"})
                break
            if convergence >= self.config.convergence_target:
                await emit({
                    "type": "note",
                    "message": f"convergence {convergence:.2f} met the target "
                               f"{self.config.convergence_target:.2f}; negotiation ended early",
                })
                break

            note = await self._chair_note(task, contested, emit)
            before = {cid: c.status for cid, c in self.ledger.claims.items()}
            ballots = await self._collect_ballots(
                task,
                contested,
                emit,
                phase="negotiation",
                system=lambda spec, r=round_no: protocol.negotiation_system(spec, r),
                user=lambda spec, c=contested, n=note: protocol.negotiation_user(task, c, spec.key, n),
            )
            self._apply_ballots(ballots, round_no=round_no, allow_new_claims=False)
            self.ledger.update_weights()
            summary = self.ledger.close_round(round_no, before)
            await emit({"type": "round", "data": summary.model_dump()})
            await emit({"type": "ledger", "data": self.ledger.snapshot()})

    async def _chair_note(self, task: str, contested: list[Claim], emit: Emit) -> str:
        orch = self.registry.get(self.config.orchestrator)
        result = await self._call(
            self.config.orchestrator,
            protocol.chair_note_system(orch),
            protocol.chair_note_user(task, contested),
            phase="chair_note",
            json_object=False,
            context={"task": task, "claims": [{"id": c.id, "text": c.text} for c in contested]},
            emit=emit,
        )
        note = result.text.strip() if result.ok else ""
        if note:
            self._chair_notes.append(note)
            await emit({"type": "chair_note", "text": note})
        return note

    # ------------------------------------------------------------------
    # Phase 5 — synthesis
    # ------------------------------------------------------------------

    async def _phase_synthesis(self, task: str, context: str, emit: Emit) -> None:
        await emit({"type": "phase", "phase": Phase.SYNTHESIS.value})
        orch = self.registry.get(self.config.orchestrator)
        snapshot = self.ledger.snapshot()

        result = await self._call(
            self.config.orchestrator,
            protocol.synthesis_system(orch),
            protocol.synthesis_user(
                task,
                context,
                self.framing.model_dump() if self.framing else {},
                snapshot,
                [p.model_dump() for p in self.proposals],
            ),
            phase="synthesis",
            context={
                "task": task,
                "accepted": [c.text for c in self.ledger.accepted()],
                "contested": [c.text for c in self.ledger.contested()],
            },
            emit=emit,
        )

        data = extract_json(result.text)
        if data is None:
            synthesis = Synthesis(
                headline="Synthesis returned prose rather than structured output.",
                analysis=result.text.strip()[:8000] or (result.error or "no output"),
                findings=[c.text for c in self.ledger.accepted()][:8],
                dissent=[c.text for c in self.ledger.contested()][:8],
                confidence=0.3,
                raw=result.text[:8000] or None,
                simulated=result.simulated,
            )
        else:
            try:
                synthesis = Synthesis(simulated=result.simulated, **_only(data, Synthesis))
            except Exception as exc:  # noqa: BLE001
                synthesis = Synthesis(
                    headline=str(data.get("headline", ""))[:400],
                    analysis=str(data.get("analysis", ""))[:8000],
                    findings=[c.text for c in self.ledger.accepted()][:8],
                    dissent=[c.text for c in self.ledger.contested()][:8],
                    confidence=0.3,
                    raw=f"schema mismatch: {exc}",
                    simulated=result.simulated,
                )

        if not synthesis.dissent:
            synthesis.dissent = [
                f"[{c.id}] {c.text} (support {c.support:+.2f})"
                for c in self.ledger.contested()
            ][:8]

        self.synthesis = synthesis
        await emit({"type": "synthesis", "data": synthesis.model_dump()})
        await emit({"type": "phase", "phase": Phase.DONE.value})

    # ------------------------------------------------------------------
    # shared machinery
    # ------------------------------------------------------------------

    async def _collect_ballots(
        self,
        task: str,
        claims: list[Claim],
        emit: Emit,
        *,
        phase: str,
        system,
        user,
    ) -> list[Ballot]:
        claim_ctx = [
            {"id": c.id, "text": c.text, "author": c.author, "kind": c.kind.value}
            for c in claims
        ]

        async def one(key: str) -> Ballot:
            spec = self.registry.get(key)
            result = await self._call(
                key,
                system(spec),
                user(spec),
                phase=phase,
                context={"task": task, "claims": claim_ctx},
                emit=emit,
            )
            return self._parse_ballot(key, result)

        return list(await asyncio.gather(*(one(k) for k in self.panel_keys)))

    def _parse_ballot(self, key: str, result: LLMResult) -> Ballot:
        if not result.ok:
            return Ballot(model_key=key, error=result.error or "no output", simulated=result.simulated)
        data = extract_json(result.text)
        if data is None:
            return Ballot(
                model_key=key,
                error="response was not JSON; ballot discarded",
                raw=result.text[:2000],
                simulated=result.simulated,
            )
        votes: list[Vote] = []
        for raw_vote in data.get("votes") or []:
            if not isinstance(raw_vote, dict) or not raw_vote.get("claim_id"):
                continue
            try:
                votes.append(Vote(**_only(raw_vote, Vote)))
            except Exception:  # noqa: BLE001 - drop the bad vote, keep the ballot
                continue
        new_claims: list[ProposedClaim] = []
        for raw_claim in (data.get("new_claims") or [])[:2]:
            if not isinstance(raw_claim, dict) or not raw_claim.get("text"):
                continue
            try:
                new_claims.append(ProposedClaim(**_only(raw_claim, ProposedClaim)))
            except Exception:  # noqa: BLE001
                continue
        return Ballot(
            model_key=key, votes=votes, new_claims=new_claims, simulated=result.simulated
        )

    def _apply_ballots(
        self, ballots: list[Ballot], round_no: int, allow_new_claims: bool = True
    ) -> None:
        for ballot in ballots:
            for vote in ballot.votes:
                self.ledger.record_vote(ballot.model_key, vote)
        if not allow_new_claims:
            return
        for ballot in ballots:
            for claim in ballot.new_claims:
                if not claim.text.strip() or self._duplicate_of(claim.text):
                    continue
                self.ledger.add_claim(claim, author=ballot.model_key, round_no=round_no)

    async def _call(
        self,
        key: str,
        system: str,
        user: str,
        *,
        phase: str,
        context: dict[str, Any] | None = None,
        json_object: bool = True,
        emit: Emit,
    ) -> LLMResult:
        budget = self._budget_for(key, phase)
        await emit({"type": "model_start", "model": key, "phase": phase})
        result = await self.registry.call(
            key, system, user, json_object=json_object, phase=phase, context=context,
            overrides={"max_tokens": budget},
        )
        self.stats.record(result)

        # A model whose reply won't parse contributes no claims and casts no
        # votes — its whole seat is wasted. One retry recovers most of them, and
        # costs a call only when the first reply was already unusable.
        #
        # The two causes need opposite treatment. Prose needs a firmer
        # instruction; a truncated reply needs *room*, and re-sending it at the
        # same budget is a guaranteed second failure at full price. The first
        # live run did exactly that and lost the action plan twice over.
        if json_object and result.ok and extract_json(result.text) is None:
            reason = "truncated" if result.truncated else "not-json"
            await emit({
                "type": "model_repair", "model": key, "phase": phase, "reason": reason,
            })
            retry_user = user if result.truncated else user + REPAIR_SUFFIX
            retry_budget = (
                min(budget * TRUNCATION_ESCALATION, MAX_TOKEN_CEILING)
                if result.truncated
                else budget
            )
            retry = await self.registry.call(
                key,
                system,
                retry_user,
                json_object=json_object,
                phase=phase,
                context=context,
                overrides={"max_tokens": retry_budget},
            )
            self.stats.record(retry)
            self.stats.repairs += 1
            if retry.ok and extract_json(retry.text) is not None:
                result = retry
        await emit({
            "type": "model_done",
            "model": key,
            "phase": phase,
            "ok": result.ok,
            "simulated": result.simulated,
            "refused": result.refused,
            "latency_s": round(result.latency_s, 2),
            "output_tokens": result.output_tokens,
            "error": result.error,
        })
        return result

    def _budget_for(self, key: str, phase: str) -> int:
        """Output-token budget for one call, scaled by how much the phase emits."""
        base = self.registry.get(key).max_tokens
        return min(base * PHASE_TOKEN_SCALE.get(phase, 1), MAX_TOKEN_CEILING)

    def _duplicate_of(self, text: str) -> bool:
        """Cheap near-duplicate check so twelve models don't fill the ledger
        with twelve wordings of the same claim. Token-overlap only — no
        embeddings, no extra API call."""
        candidate = _tokens(text)
        if not candidate:
            return True
        for claim in self.ledger.claims.values():
            existing = _tokens(claim.text)
            if not existing:
                continue
            overlap = len(candidate & existing) / len(candidate | existing)
            if overlap >= DEDUPE_THRESHOLD:
                return True
        return False

    # ------------------------------------------------------------------

    def report(self, task: str, context: str = "") -> dict[str, Any]:
        return {
            "task": task,
            "context": context,
            "config": {
                "orchestrator": self.config.orchestrator,
                "orchestrator_name": self.registry.get(self.config.orchestrator).name,
                "panel": self.panel_keys,
                "rounds": self.config.rounds,
                "convergence_target": self.config.convergence_target,
                "max_calls": self.config.max_calls(),
            },
            "framing": self.framing.model_dump() if self.framing else None,
            "proposals": [p.model_dump() for p in self.proposals],
            "chair_notes": self._chair_notes,
            "ledger": self.ledger.snapshot(),
            "synthesis": self.synthesis.model_dump() if self.synthesis else None,
            "stats": {
                "calls": self.stats.calls,
                "failures": self.stats.failures,
                "simulated_calls": self.stats.simulated,
                "repair_retries": self.stats.repairs,
                "input_tokens": self.stats.input_tokens,
                "output_tokens": self.stats.output_tokens,
                "wall_s": self.stats.wall_s,
                "per_model": self.stats.per_model,
            },
            "any_simulated": self.stats.simulated > 0,
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "be", "in", "on",
    "for", "with", "that", "this", "it", "as", "by", "at", "from", "will",
    "should", "must", "can", "we", "its", "their",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _only(data: dict[str, Any], model_cls) -> dict[str, Any]:
    """Keep just the keys the pydantic model declares; models love extra keys."""
    fields = set(model_cls.model_fields)
    return {k: v for k, v in data.items() if k in fields}


def protocol_assignment(model_key: str, lens: str):
    from .schemas import Assignment

    return Assignment(
        model_key=model_key,
        role=f"{lens} analyst",
        brief=(
            f"Analyse the task through the {lens} lens. Surface what the other "
            "lenses are likely to miss, and state where your view would break."
        ),
    )


async def _noop(_: dict[str, Any]) -> None:
    return None
