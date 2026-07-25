"""Offline stand-in so the council runs with zero API keys.

This exists to make the *protocol* inspectable and testable — it is not a model.
Every payload it emits is deterministic (seeded by model key + task) and every
surface that renders it is required to show the SIMULATED badge. Do not read
simulated output as analysis; read it as a wiring diagram.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from typing import Any

from .base import LLMResult, ModelSpec

_HEDGES = [
    "holds under the stated assumptions but is sensitive to demand estimates",
    "is well supported by the framing but under-specifies the failure mode",
    "is directionally right while the magnitude remains unverified",
    "conflicts with the cost constraint unless scope is reduced",
    "is the highest-leverage item once sequencing is fixed",
]


class SimulatedProvider:
    """Deterministic protocol-shaped output. Never a substitute for a real model."""

    def __init__(self, latency_s: float = 0.05) -> None:
        self._latency = latency_s

    async def complete(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        phase: str = "",
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        await asyncio.sleep(self._latency)
        rng = random.Random(_seed(spec.key, phase, user))
        ctx = context or {}

        # The chair's steering note is prose, not JSON — match the real contract.
        if phase == "chair_note":
            ids = ", ".join(c.get("id", "?") for c in ctx.get("claims", [])[:4]) or "the open items"
            return LLMResult(
                text=(
                    f"[SIMULATED] The council is split on {ids}. In a live run the chair "
                    "would name the crux of each disagreement here and propose narrower "
                    "wordings both sides could endorse."
                ),
                model_key=spec.key,
                output_tokens=45,
                latency_s=self._latency,
                simulated=True,
            )

        builders = {
            "framing": _framing,
            "proposal": _proposal,
            "cross_examination": _ballot,
            "negotiation": _ballot,
            "synthesis": _synthesis,
        }
        payload = builders.get(phase, _generic)(spec, rng, ctx)
        return LLMResult(
            text=json.dumps(payload, indent=2),
            model_key=spec.key,
            input_tokens=len(user) // 4,
            output_tokens=200,
            latency_s=self._latency,
            simulated=True,
        )

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16)


def _framing(spec: ModelSpec, rng: random.Random, ctx: dict) -> dict:
    task = ctx.get("task", "the task")
    panel: list[str] = ctx.get("panel", [])
    lenses: dict[str, str] = ctx.get("lenses", {})
    return {
        "objective": f"[SIMULATED] Reach a defensible, actionable answer on: {task[:160]}",
        "success_criteria": [
            "Every headline claim survives adversarial review",
            "The action plan is sequenced and owner-assignable",
            "Residual disagreement is named rather than averaged away",
        ],
        "key_uncertainties": [
            "Which constraints are hard versus negotiable",
            "The quality of the evidence behind the base case",
        ],
        "decomposition": [
            "Establish the factual baseline",
            "Identify the decision levers",
            "Stress-test the leading option",
            "Sequence the practical steps",
        ],
        "assignments": [
            {
                "model_key": key,
                "role": f"{lenses.get(key, 'general')} analyst",
                "brief": (
                    f"Attack the problem strictly through the "
                    f"{lenses.get(key, 'general')} lens; surface what the other "
                    "lenses would miss."
                ),
            }
            for key in panel
        ],
    }


def _proposal(spec: ModelSpec, rng: random.Random, ctx: dict) -> dict:
    lens = spec.lens
    n = rng.randint(2, 3)
    return {
        "position": f"[SIMULATED — {spec.name}] Viewed through {lens}, the decisive factor is sequencing, not capability.",
        "claims": [
            {
                "text": f"[SIM/{spec.key}] Claim {i + 1}: the {lens} constraint {rng.choice(_HEDGES)}.",
                "kind": "practical" if i % 2 else "analytical",
                "confidence": round(rng.uniform(0.45, 0.9), 2),
                "rationale": f"Derived from {lens} priors; not empirically grounded in this run.",
            }
            for i in range(n)
        ],
        "assumptions": [f"{lens} constraints stay within their current envelope"],
        "risks": [f"The {lens} view over-weights measurable factors"],
        "open_questions": [f"What evidence would falsify the {lens} reading?"],
    }


def _ballot(spec: ModelSpec, rng: random.Random, ctx: dict) -> dict:
    claims: list[dict] = ctx.get("claims", [])
    votes = []
    for claim in claims:
        if claim.get("author") == spec.key:
            stance, conf = "endorse", rng.uniform(0.75, 0.95)
        else:
            stance = rng.choices(
                ["endorse", "dispute", "abstain"], weights=[0.55, 0.3, 0.15]
            )[0]
            conf = rng.uniform(0.4, 0.9)
        votes.append(
            {
                "claim_id": claim.get("id"),
                "stance": stance,
                "confidence": round(conf, 2),
                "rationale": f"[SIM/{spec.key}] {stance} on {spec.lens} grounds.",
                "amendment": (
                    f"Bound the claim to the {spec.lens} case."
                    if stance == "dispute"
                    else None
                ),
            }
        )
    return {"votes": votes, "new_claims": []}


def _synthesis(spec: ModelSpec, rng: random.Random, ctx: dict) -> dict:
    accepted: list[str] = ctx.get("accepted", [])
    contested: list[str] = ctx.get("contested", [])
    return {
        "headline": "[SIMULATED] Proceed on the accepted core; hold the contested items behind one decision gate.",
        "analysis": (
            "This is simulated output produced without any model call. It shows "
            "the shape of a synthesis — accepted claims form the spine, contested "
            "claims become explicit gates — not a real conclusion."
        ),
        "findings": accepted[:6] or ["No claim reached consensus in this run."],
        "action_plan": [
            {
                "action": "Validate the top accepted claim against real data",
                "owner_hint": "analysis lead",
                "effort": "low",
                "impact": "high",
                "depends_on": [],
                "first_step": "Pull the source data and re-run the baseline",
            },
            {
                "action": "Resolve the contested items with a time-boxed spike",
                "owner_hint": "engineering",
                "effort": "medium",
                "impact": "high",
                "depends_on": ["Validate the top accepted claim against real data"],
                "first_step": "Write the falsification test for each contested claim",
            },
        ],
        "dissent": contested[:5],
        "confidence": 0.35,
        "what_would_change_our_mind": [
            "Running this with live API keys instead of the simulator",
        ],
    }


def _generic(spec: ModelSpec, rng: random.Random, ctx: dict) -> dict:
    return {"note": f"[SIMULATED] {spec.name} produced no phase-specific payload."}
