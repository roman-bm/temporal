"""The ledger: claims, weighted voting, convergence, and reliability updates.

This is where "negotiation" stops being a metaphor. Every substantive statement
becomes a claim with an ID. Every model votes on every live claim with a stance
and a confidence. Support is a weight-normalised net position; convergence is
how much of the ledger has stopped moving.

Design notes worth defending:

* Abstentions do not count as agreement. They lower *participation* instead, so
  a claim nobody engaged with can never read as settled.
* Weights are per-model and mutate across rounds. A model that consistently
  lands on the side the council converges to gains weight; a model that keeps
  voting against the eventual consensus loses it. This is deliberately mild
  (±10% per round, clamped) — it should nudge, not create a tyranny of the
  majority.
* Claims are never deleted. Rejected and contested claims are carried into the
  dissent register, because the losing argument is often the useful one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Claim, ClaimKind, ClaimStatus, ProposedClaim, RoundSummary, Stance, Vote

# Support is weight-normalised to [-1, +1], so the threshold maps directly onto
# a majority ratio: 0.5 is a 3:1 weighted supermajority, 0.6 is 4:1. We settle
# at 3:1 — stricter than that and a realistic panel leaves everything contested,
# which makes the dissent register meaningless.
ACCEPT_THRESHOLD = 0.50
REJECT_THRESHOLD = -0.50
MIN_PARTICIPATION = 0.50

_STANCE_VALUE = {Stance.ENDORSE: 1.0, Stance.DISPUTE: -1.0, Stance.ABSTAIN: 0.0}


@dataclass
class Ledger:
    weights: dict[str, float]
    claims: dict[str, Claim] = field(default_factory=dict)
    rounds: list[RoundSummary] = field(default_factory=list)
    _counter: int = 0

    # ------------------------------------------------------------------
    # claims
    # ------------------------------------------------------------------

    def add_claim(self, proposed: ProposedClaim, author: str, round_no: int) -> Claim:
        self._counter += 1
        claim_id = f"C{self._counter:03d}"
        claim = Claim(
            id=claim_id,
            text=proposed.text.strip(),
            kind=proposed.kind,
            author=author,
            round_introduced=round_no,
        )
        # The author implicitly endorses their own claim at their stated
        # confidence. Making this explicit keeps participation honest.
        claim.votes[author] = Vote(
            claim_id=claim_id,
            stance=Stance.ENDORSE,
            confidence=proposed.confidence,
            rationale=proposed.rationale or "author's own claim",
        )
        self.claims[claim_id] = claim
        self._score(claim)
        return claim

    def record_vote(self, model_key: str, vote: Vote) -> bool:
        claim = self.claims.get(vote.claim_id)
        if claim is None:
            return False
        if model_key == claim.author and vote.stance is Stance.DISPUTE:
            # A model retracting its own claim is meaningful — keep it.
            claim.amendments.append(f"{model_key} retracted: {vote.rationale}")
        previous = claim.votes.get(model_key)
        claim.votes[model_key] = vote
        if vote.amendment:
            claim.amendments.append(f"{model_key}: {vote.amendment}")
        if previous and previous.stance is not vote.stance:
            claim.history.append(
                {
                    "model": model_key,
                    "from": previous.stance.value,
                    "to": vote.stance.value,
                    "why": vote.rationale[:280],
                }
            )
        self._score(claim)
        return True

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _score(self, claim: Claim) -> None:
        total_weight = sum(self.weights.get(k, 1.0) for k in self.weights)
        net = 0.0
        mass = 0.0
        for model_key, vote in claim.votes.items():
            w = self.weights.get(model_key, 1.0) * vote.confidence
            value = _STANCE_VALUE[vote.stance]
            net += w * value
            mass += w * abs(value)

        claim.support = (net / mass) if mass > 0 else 0.0
        claim.participation = (mass / total_weight) if total_weight > 0 else 0.0

        if claim.participation < MIN_PARTICIPATION:
            claim.status = ClaimStatus.CONTESTED
        elif claim.support >= ACCEPT_THRESHOLD:
            claim.status = ClaimStatus.ACCEPTED
        elif claim.support <= REJECT_THRESHOLD:
            claim.status = ClaimStatus.REJECTED
        else:
            claim.status = ClaimStatus.CONTESTED

    # ------------------------------------------------------------------
    # round accounting
    # ------------------------------------------------------------------

    def close_round(self, round_no: int, before: dict[str, ClaimStatus]) -> RoundSummary:
        moved = [
            cid
            for cid, claim in self.claims.items()
            if before.get(cid) is not None and before[cid] is not claim.status
        ]
        summary = RoundSummary(
            round=round_no,
            convergence=self.convergence(),
            accepted=self.count(ClaimStatus.ACCEPTED),
            rejected=self.count(ClaimStatus.REJECTED),
            contested=self.count(ClaimStatus.CONTESTED),
            moved=moved,
        )
        self.rounds.append(summary)
        return summary

    def convergence(self) -> float:
        """0 = the panel is all over the place, 1 = every claim is settled.

        Each claim contributes |support| discounted by how many models actually
        engaged with it. A ledger of confidently-rejected claims scores as
        converged, which is correct: the council has decided.
        """
        if not self.claims:
            return 0.0
        total = sum(abs(c.support) * min(1.0, c.participation / MIN_PARTICIPATION)
                    for c in self.claims.values())
        return round(min(1.0, total / len(self.claims)), 4)

    def count(self, status: ClaimStatus) -> int:
        return sum(1 for c in self.claims.values() if c.status is status)

    def contested(self) -> list[Claim]:
        return [c for c in self.claims.values() if c.status is ClaimStatus.CONTESTED]

    def accepted(self) -> list[Claim]:
        ordered = [c for c in self.claims.values() if c.status is ClaimStatus.ACCEPTED]
        return sorted(ordered, key=lambda c: (-c.support, c.id))

    def rejected(self) -> list[Claim]:
        return [c for c in self.claims.values() if c.status is ClaimStatus.REJECTED]

    def by_kind(self, kind: ClaimKind, status: ClaimStatus) -> list[Claim]:
        return [c for c in self.claims.values() if c.kind is kind and c.status is status]

    # ------------------------------------------------------------------
    # reliability
    # ------------------------------------------------------------------

    def update_weights(self, *, rate: float = 0.10, floor: float = 0.4, ceil: float = 2.0) -> dict[str, float]:
        """Nudge each model's weight toward its alignment with settled claims.

        Only decided claims (accepted/rejected) count — grading a model against
        an unresolved question would just reward conformity.
        """
        decided = [c for c in self.claims.values() if c.status is not ClaimStatus.CONTESTED]
        if not decided:
            return dict(self.weights)

        for model_key in list(self.weights):
            hits = 0.0
            seen = 0.0
            for claim in decided:
                vote = claim.votes.get(model_key)
                if vote is None or vote.stance is Stance.ABSTAIN:
                    continue
                seen += 1
                aligned = (
                    (claim.status is ClaimStatus.ACCEPTED and vote.stance is Stance.ENDORSE)
                    or (claim.status is ClaimStatus.REJECTED and vote.stance is Stance.DISPUTE)
                )
                hits += 1.0 if aligned else 0.0
            if seen == 0:
                continue
            accuracy = hits / seen            # 0..1
            adjustment = 1.0 + rate * (2 * accuracy - 1.0)   # 0.9x .. 1.1x at rate=0.1
            self.weights[model_key] = round(
                max(floor, min(ceil, self.weights[model_key] * adjustment)), 4
            )
        return dict(self.weights)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "convergence": self.convergence(),
            "weights": dict(self.weights),
            "counts": {
                "accepted": self.count(ClaimStatus.ACCEPTED),
                "rejected": self.count(ClaimStatus.REJECTED),
                "contested": self.count(ClaimStatus.CONTESTED),
            },
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "kind": c.kind.value,
                    "author": c.author,
                    "support": round(c.support, 3),
                    "participation": round(c.participation, 3),
                    "status": c.status.value,
                    "round_introduced": c.round_introduced,
                    "amendments": c.amendments[-4:],
                    "history": c.history[-4:],
                    "votes": {
                        k: {
                            "stance": v.stance.value,
                            "confidence": v.confidence,
                            "rationale": v.rationale[:400],
                        }
                        for k, v in c.votes.items()
                    },
                }
                for c in self.claims.values()
            ],
            "rounds": [r.model_dump() for r in self.rounds],
        }
