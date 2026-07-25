"""Typed payloads exchanged between the orchestrator and the panel.

Everything a model returns is parsed into one of these. If a model returns
malformed JSON we degrade to a `raw` fallback rather than aborting the run —
one bad panelist must never take down a council session.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Stance(str, Enum):
    ENDORSE = "endorse"
    DISPUTE = "dispute"
    ABSTAIN = "abstain"


class ClaimKind(str, Enum):
    ANALYTICAL = "analytical"   # a statement about how the world is
    PRACTICAL = "practical"     # a recommended action


class Phase(str, Enum):
    FRAMING = "framing"
    PROPOSAL = "proposal"
    CROSS_EXAM = "cross_examination"
    NEGOTIATION = "negotiation"
    SYNTHESIS = "synthesis"
    DONE = "done"


# --------------------------------------------------------------------------
# Phase 1 — orchestrator framing
# --------------------------------------------------------------------------

class Assignment(BaseModel):
    model_key: str
    role: str = Field(description="Short role title, e.g. 'Cost adversary'")
    brief: str = Field(description="What this model must specifically produce")


class Framing(BaseModel):
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    decomposition: list[str] = Field(default_factory=list)
    assignments: list[Assignment] = Field(default_factory=list)

    @field_validator("success_criteria", "key_uncertainties", "decomposition", mode="before")
    @classmethod
    def _coerce_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v


# --------------------------------------------------------------------------
# Phase 2 — panel proposals
# --------------------------------------------------------------------------

class ProposedClaim(BaseModel):
    text: str
    kind: ClaimKind = ClaimKind.ANALYTICAL
    confidence: float = 0.6
    rationale: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class Proposal(BaseModel):
    model_key: str
    position: str = ""
    claims: list[ProposedClaim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    raw: str | None = None
    error: str | None = None
    simulated: bool = False


# --------------------------------------------------------------------------
# Phase 3/4 — votes and rebuttals
# --------------------------------------------------------------------------

class Vote(BaseModel):
    claim_id: str
    stance: Stance = Stance.ABSTAIN
    confidence: float = 0.5
    rationale: str = ""
    amendment: str | None = Field(
        default=None,
        description="A concrete rewrite that would move this voter to endorse.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("stance", mode="before")
    @classmethod
    def _norm_stance(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("agree", "support", "yes", "endorse"):
                return "endorse"
            if s in ("disagree", "oppose", "no", "dispute", "reject"):
                return "dispute"
            if s in ("abstain", "unsure", "neutral", "pass"):
                return "abstain"
        return v


class Ballot(BaseModel):
    """One model's votes for one round."""
    model_key: str
    votes: list[Vote] = Field(default_factory=list)
    new_claims: list[ProposedClaim] = Field(default_factory=list)
    raw: str | None = None
    error: str | None = None
    simulated: bool = False


# --------------------------------------------------------------------------
# Ledger — the orchestrator's running state
# --------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONTESTED = "contested"


class Claim(BaseModel):
    id: str
    text: str
    kind: ClaimKind
    author: str
    round_introduced: int = 0
    votes: dict[str, Vote] = Field(default_factory=dict)   # model_key -> latest vote
    support: float = 0.0          # -1 .. +1
    participation: float = 0.0    # 0 .. 1
    status: ClaimStatus = ClaimStatus.CONTESTED
    history: list[dict[str, Any]] = Field(default_factory=list)
    amendments: list[str] = Field(default_factory=list)


class RoundSummary(BaseModel):
    round: int
    convergence: float
    accepted: int
    rejected: int
    contested: int
    moved: list[str] = Field(default_factory=list)  # claim ids that changed status


# --------------------------------------------------------------------------
# Phase 5 — synthesis
# --------------------------------------------------------------------------

class ActionItem(BaseModel):
    action: str
    owner_hint: str = ""
    effort: Literal["low", "medium", "high"] = "medium"
    impact: Literal["low", "medium", "high"] = "medium"
    depends_on: list[str] = Field(default_factory=list)
    first_step: str = ""


class Synthesis(BaseModel):
    headline: str = ""
    analysis: str = ""
    findings: list[str] = Field(default_factory=list)
    action_plan: list[ActionItem] = Field(default_factory=list)
    dissent: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    what_would_change_our_mind: list[str] = Field(default_factory=list)
    raw: str | None = None
    simulated: bool = False
