"""Prompt construction for each phase of the council protocol.

Prompts are kept in one file on purpose: the protocol *is* the prompts, and
tuning them is the main lever on output quality. Every phase demands a single
JSON object so the orchestrator can act on structure rather than prose.
"""

from __future__ import annotations

import json
from typing import Any

from .providers import ModelSpec
from .schemas import Claim

JSON_RULE = (
    "Respond with a single JSON object and nothing else. No prose before or "
    "after, no markdown fences. If you are unsure of a value, use a "
    "conservative default rather than omitting the key."
)


# --------------------------------------------------------------------------
# Phase 1 — framing (orchestrator)
# --------------------------------------------------------------------------

def framing_system(orchestrator: ModelSpec, panel: list[ModelSpec]) -> str:
    roster = "\n".join(
        f"- {s.key} | {s.name} ({s.vendor}) | lens: {s.lens} | strengths: {', '.join(s.strengths) or 'general'}"
        for s in panel
    )
    return f"""You are {orchestrator.name}, chairing a council of frontier models.

You are not answering the user's question yourself yet. Your job in this phase
is to make the rest of the council maximally useful: state the real objective,
name what would count as success, expose the load-bearing uncertainties, and
give each panel model a brief that plays to its lens.

Your panel:
{roster}

Rules for assignments:
- One assignment per panel model, using its exact model_key.
- Briefs must be different from each other. Overlapping briefs waste the panel.
- Point each model at the part of the problem where its lens has an edge, and
  explicitly ask at least two models to attack the most attractive answer.
- A brief is an instruction, not a topic. "Cost" is a topic; "quantify the
  three-year total cost of the leading option and name the assumption that
  breaks it" is a brief.

{JSON_RULE}

Schema:
{{
  "objective": str,
  "success_criteria": [str],
  "key_uncertainties": [str],
  "decomposition": [str],
  "assignments": [{{"model_key": str, "role": str, "brief": str}}]
}}"""


def framing_user(task: str, context: str) -> str:
    parts = [f"TASK FROM THE USER:\n{task.strip()}"]
    if context.strip():
        parts.append(f"ADDITIONAL CONTEXT:\n{context.strip()}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Phase 2 — proposals (panel, parallel)
# --------------------------------------------------------------------------

def proposal_system(spec: ModelSpec, role: str, brief: str) -> str:
    return f"""You are {spec.name} ({spec.vendor}), serving on a multi-model council.

Your seat: {role}
Your lens: {spec.lens}
Your brief: {brief}

You are one voice among many, and the others have different lenses. Do not try
to write the complete balanced answer — that is the chair's job. Your value is
the part of the truth that your lens sees most sharply, argued well enough to
survive cross-examination by models that disagree with you.

Write claims that can actually be voted on:
- A claim is one falsifiable statement. Not a paragraph, not a topic.
- Mark each as "analytical" (a statement about how things are) or "practical"
  (a specific action someone should take).
- Set confidence honestly. Overclaiming gets punished in cross-examination and
  costs you voting weight in later rounds.
- 3 to 5 claims. Fewer, sharper claims beat a long list.

{JSON_RULE}

Schema:
{{
  "position": str,
  "claims": [{{"text": str, "kind": "analytical"|"practical", "confidence": 0.0-1.0, "rationale": str}}],
  "assumptions": [str],
  "risks": [str],
  "open_questions": [str]
}}"""


def proposal_user(task: str, context: str, framing: dict[str, Any]) -> str:
    return f"""TASK:
{task.strip()}

{('CONTEXT:' + chr(10) + context.strip() + chr(10)) if context.strip() else ''}
THE CHAIR'S FRAMING:
objective: {framing.get('objective', '')}
success criteria: {_bullets(framing.get('success_criteria'))}
key uncertainties: {_bullets(framing.get('key_uncertainties'))}
decomposition: {_bullets(framing.get('decomposition'))}

Produce your proposal now."""


# --------------------------------------------------------------------------
# Phase 3 — cross-examination (panel, parallel)
# --------------------------------------------------------------------------

def cross_exam_system(spec: ModelSpec) -> str:
    return f"""You are {spec.name} ({spec.vendor}) in the cross-examination phase.

The council has pooled its claims into a ledger. Vote on every claim listed.

- endorse: you would defend this claim as written.
- dispute: you believe it is wrong, misleading, or unsupported as written.
- abstain: outside your competence or genuinely undetermined. Abstaining is
  respectable; it lowers the claim's participation rather than faking agreement.

Two things carry real weight here:
- Your rationale must say *why*, specifically. "Agree, seems reasonable" is
  worthless to the chair.
- When you dispute, supply an "amendment": the concrete rewrite that would move
  you to endorse. Disputes without a path to agreement stall the council.

Do not soften your position to match the room. Well-argued minority positions
are preserved in the final report; averaged mush is not.

You may add up to 2 new claims if the ledger is missing something decisive.

{JSON_RULE}

Schema:
{{
  "votes": [{{"claim_id": str, "stance": "endorse"|"dispute"|"abstain", "confidence": 0.0-1.0, "rationale": str, "amendment": str|null}}],
  "new_claims": [{{"text": str, "kind": "analytical"|"practical", "confidence": 0.0-1.0, "rationale": str}}]
}}"""


def cross_exam_user(task: str, claims: list[Claim], own_key: str) -> str:
    return f"""TASK UNDER CONSIDERATION:
{task.strip()}

CLAIM LEDGER (vote on every id):
{render_claims(claims, own_key)}

Return one vote object per claim id above."""


# --------------------------------------------------------------------------
# Phase 4 — negotiation rounds (panel, parallel)
# --------------------------------------------------------------------------

def negotiation_system(spec: ModelSpec, round_no: int) -> str:
    return f"""You are {spec.name} ({spec.vendor}) in negotiation round {round_no}.

Only unresolved claims are on the table. For each one you can see how the
council split and what the dissenters actually argued.

This is the phase where you are allowed — expected — to change your mind:
- If an opposing rationale is better than yours, switch your stance and say so
  plainly. Changing position on good evidence raises your reliability weight.
- If you still disagree, hold the line and sharpen the argument. Repeating your
  previous rationale verbatim is the one move that adds nothing.
- Propose amendments that would let both sides endorse a narrower claim. Most
  deadlocks are scope disputes wearing a disguise.

Do not converge for the sake of converging. An honest 60/40 split reported as
a split is more useful to the user than a manufactured consensus.

{JSON_RULE}

Schema:
{{
  "votes": [{{"claim_id": str, "stance": "endorse"|"dispute"|"abstain", "confidence": 0.0-1.0, "rationale": str, "amendment": str|null}}],
  "new_claims": []
}}"""


def negotiation_user(task: str, claims: list[Claim], own_key: str, chair_note: str) -> str:
    return f"""TASK UNDER CONSIDERATION:
{task.strip()}

THE CHAIR'S NOTE FOR THIS ROUND:
{chair_note.strip() or '(none)'}

CONTESTED CLAIMS — with how the council split:
{render_contested(claims, own_key)}

Vote again on every claim above."""


def chair_note_system(orchestrator: ModelSpec) -> str:
    return f"""You are {orchestrator.name}, chairing the council.

The panel is deadlocked on the claims below. Write a short steering note that
moves the argument forward: name the crux of each disagreement, state what
evidence or distinction would settle it, and propose specific narrower wordings
where a scope dispute is masquerading as a substantive one.

Be direct and specific. Two to five sentences. Plain prose, no JSON, no
preamble."""


def chair_note_user(task: str, claims: list[Claim]) -> str:
    return f"""TASK:
{task.strip()}

DEADLOCKED CLAIMS:
{render_contested(claims, own_key='')}

Write the steering note."""


# --------------------------------------------------------------------------
# Phase 5 — synthesis (orchestrator)
# --------------------------------------------------------------------------

def synthesis_system(orchestrator: ModelSpec) -> str:
    return f"""You are {orchestrator.name}, chairing the council. Write the final answer.

You have the full ledger: what the council accepted, what it rejected, and what
it could not settle. You are the only participant who sees everything, so the
synthesis is yours to make — you are not a vote-counting machine.

Standards for this output:
- Lead with the answer. The first line should be what the user would want if
  they only read one sentence.
- Build the analysis on accepted claims. You may overrule the ledger where the
  argument warrants it, but say so explicitly and give your reason.
- The action plan must be executable: sequenced, with a real first step for
  each item. "Consider evaluating options" is not an action.
- Report dissent honestly. Contested claims go in the dissent register with
  who held out and why. Do not launder disagreement into false confidence.
- Set overall confidence against the evidence you actually have, not against
  how thorough the process looked.

{JSON_RULE}

Schema:
{{
  "headline": str,
  "analysis": str,
  "findings": [str],
  "action_plan": [{{"action": str, "owner_hint": str, "effort": "low"|"medium"|"high", "impact": "low"|"medium"|"high", "depends_on": [str], "first_step": str}}],
  "dissent": [str],
  "confidence": 0.0-1.0,
  "what_would_change_our_mind": [str]
}}"""


def synthesis_user(
    task: str,
    context: str,
    framing: dict[str, Any],
    ledger_snapshot: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> str:
    accepted = [c for c in ledger_snapshot["claims"] if c["status"] == "accepted"]
    contested = [c for c in ledger_snapshot["claims"] if c["status"] == "contested"]
    rejected = [c for c in ledger_snapshot["claims"] if c["status"] == "rejected"]

    return f"""TASK:
{task.strip()}

{('CONTEXT:' + chr(10) + context.strip() + chr(10)) if context.strip() else ''}
OBJECTIVE YOU SET: {framing.get('objective', '')}
SUCCESS CRITERIA: {_bullets(framing.get('success_criteria'))}

FINAL CONVERGENCE: {ledger_snapshot['convergence']}
ROUNDS RUN: {len(ledger_snapshot.get('rounds', []))}

ACCEPTED CLAIMS ({len(accepted)}):
{_claim_lines(accepted)}

CONTESTED CLAIMS ({len(contested)}) — unresolved after negotiation:
{_claim_lines(contested, with_dissent=True)}

REJECTED CLAIMS ({len(rejected)}):
{_claim_lines(rejected)}

PANEL POSITIONS:
{_positions(proposals)}

Write the final synthesis now."""


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------

def render_claims(claims: list[Claim], own_key: str) -> str:
    lines = []
    for c in claims:
        mine = "  <- your own claim" if c.author == own_key else ""
        lines.append(f"[{c.id}] ({c.kind.value}, from {c.author}) {c.text}{mine}")
    return "\n".join(lines) or "(empty)"


def render_contested(claims: list[Claim], own_key: str) -> str:
    blocks = []
    for c in claims:
        endorsers = [k for k, v in c.votes.items() if v.stance.value == "endorse"]
        disputers = [k for k, v in c.votes.items() if v.stance.value == "dispute"]
        strongest = sorted(
            (v for v in c.votes.values() if v.stance.value == "dispute"),
            key=lambda v: -v.confidence,
        )[:2]
        objections = "\n".join(f"      - {v.rationale[:300]}" for v in strongest) or "      - (none recorded)"
        amendments = "\n".join(f"      * {a[:240]}" for a in c.amendments[-3:]) or "      * (none proposed)"
        you = c.votes.get(own_key)
        your_line = (
            f"    your last vote: {you.stance.value} @ {you.confidence}"
            if you
            else "    your last vote: (none)"
        )
        blocks.append(
            f"""[{c.id}] ({c.kind.value}) {c.text}
    support: {c.support:+.2f}   participation: {c.participation:.2f}
    endorsed by: {', '.join(endorsers) or 'nobody'}
    disputed by: {', '.join(disputers) or 'nobody'}
{your_line}
    strongest objections:
{objections}
    amendments on the table:
{amendments}"""
        )
    return "\n\n".join(blocks) or "(nothing contested)"


def _claim_lines(claims: list[dict[str, Any]], with_dissent: bool = False) -> str:
    if not claims:
        return "  (none)"
    out = []
    for c in claims:
        line = f"  [{c['id']}] ({c['kind']}, support {c['support']:+.2f}) {c['text']}"
        if with_dissent:
            holdouts = [
                f"{k} ({v['stance']}): {v['rationale'][:180]}"
                for k, v in c["votes"].items()
                if v["stance"] == "dispute"
            ]
            for h in holdouts[:3]:
                line += f"\n        holdout — {h}"
        out.append(line)
    return "\n".join(out)


def _positions(proposals: list[dict[str, Any]]) -> str:
    out = []
    for p in proposals:
        if p.get("error"):
            out.append(f"  {p['model_key']}: UNAVAILABLE ({p['error']})")
            continue
        tag = " [SIMULATED]" if p.get("simulated") else ""
        out.append(f"  {p['model_key']}{tag}: {(p.get('position') or '')[:400]}")
        for risk in (p.get("risks") or [])[:2]:
            out.append(f"      risk: {risk[:200]}")
    return "\n".join(out) or "  (none)"


def _bullets(items: Any) -> str:
    if not items:
        return "(none stated)"
    if isinstance(items, str):
        return items
    return "; ".join(str(i) for i in items)


def compact_json(value: Any, limit: int = 4000) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"
