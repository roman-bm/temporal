from orchestra.consensus import Ledger
from orchestra.schemas import ClaimStatus, ProposedClaim, Stance, Vote


def make_ledger(*models: str) -> Ledger:
    return Ledger(weights={m: 1.0 for m in models})


def claim(text: str = "the sky is blue") -> ProposedClaim:
    return ProposedClaim(text=text, confidence=0.8)


def vote(claim_id: str, stance: str, confidence: float = 0.8) -> Vote:
    return Vote(claim_id=claim_id, stance=Stance(stance), confidence=confidence)


def test_author_endorses_own_claim():
    ledger = make_ledger("a", "b", "c")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    assert c.votes["a"].stance is Stance.ENDORSE
    assert c.support == 1.0


def test_unanimous_endorsement_is_accepted():
    ledger = make_ledger("a", "b", "c")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    ledger.record_vote("b", vote(c.id, "endorse"))
    ledger.record_vote("c", vote(c.id, "endorse"))
    assert c.status is ClaimStatus.ACCEPTED
    assert c.support == 1.0
    assert c.participation > 0.7


def test_supermajority_dispute_rejects():
    ledger = make_ledger("a", "b", "c", "d")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    for m in ("b", "c", "d"):
        ledger.record_vote(m, vote(c.id, "dispute", 0.9))
    assert c.support < 0
    assert c.status is ClaimStatus.REJECTED


def test_bare_majority_dispute_stays_contested():
    """2:1 against is a disagreement, not a verdict."""
    ledger = make_ledger("a", "b", "c")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    ledger.record_vote("b", vote(c.id, "dispute", 0.9))
    ledger.record_vote("c", vote(c.id, "dispute", 0.9))
    assert c.support < 0
    assert c.status is ClaimStatus.CONTESTED


def test_split_stays_contested():
    ledger = make_ledger("a", "b", "c", "d")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    ledger.record_vote("b", vote(c.id, "endorse", 0.8))
    ledger.record_vote("c", vote(c.id, "dispute", 0.8))
    ledger.record_vote("d", vote(c.id, "dispute", 0.8))
    assert c.status is ClaimStatus.CONTESTED


def test_abstention_lowers_participation_not_support():
    """An unengaged panel must never read as agreement."""
    ledger = make_ledger("a", "b", "c", "d", "e", "f")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    for m in ("b", "c", "d", "e", "f"):
        ledger.record_vote(m, vote(c.id, "abstain"))
    assert c.support == 1.0            # only the author took a position
    assert c.participation < 0.5       # but nobody engaged
    assert c.status is ClaimStatus.CONTESTED


def test_weights_shift_the_outcome():
    ledger = Ledger(weights={"heavy": 4.0, "light1": 0.5, "light2": 0.5})
    c = ledger.add_claim(claim(), author="heavy", round_no=0)
    ledger.record_vote("light1", vote(c.id, "dispute", 0.9))
    ledger.record_vote("light2", vote(c.id, "dispute", 0.9))
    assert c.support > 0
    assert c.status is ClaimStatus.ACCEPTED


def test_stance_change_is_recorded():
    ledger = make_ledger("a", "b")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    ledger.record_vote("b", vote(c.id, "dispute"))
    ledger.record_vote("b", vote(c.id, "endorse"))
    assert len(c.history) == 1
    assert c.history[0]["from"] == "dispute"
    assert c.history[0]["to"] == "endorse"


def test_convergence_rises_as_claims_settle():
    ledger = make_ledger("a", "b", "c")
    c1 = ledger.add_claim(claim("one"), author="a", round_no=0)
    c2 = ledger.add_claim(claim("two"), author="b", round_no=0)
    ledger.record_vote("b", vote(c1.id, "endorse"))
    ledger.record_vote("c", vote(c1.id, "abstain", 0.5))
    before = ledger.convergence()
    ledger.record_vote("c", vote(c1.id, "endorse"))
    ledger.record_vote("a", vote(c2.id, "endorse"))
    ledger.record_vote("c", vote(c2.id, "endorse"))
    assert ledger.convergence() > before


def test_reliability_weights_reward_alignment():
    ledger = Ledger(weights={"a": 1.0, "b": 1.0, "c": 1.0, "contrarian": 1.0})
    for i in range(3):
        c = ledger.add_claim(claim(f"claim {i}"), author="a", round_no=0)
        ledger.record_vote("b", vote(c.id, "endorse"))
        ledger.record_vote("c", vote(c.id, "endorse"))
        ledger.record_vote("contrarian", vote(c.id, "dispute"))

    ledger.update_weights()
    assert ledger.weights["b"] > 1.0
    assert ledger.weights["contrarian"] < 1.0


def test_vote_on_unknown_claim_is_ignored():
    ledger = make_ledger("a", "b")
    assert ledger.record_vote("b", vote("C999", "endorse")) is False


def test_snapshot_is_json_serialisable():
    import json

    ledger = make_ledger("a", "b")
    c = ledger.add_claim(claim(), author="a", round_no=0)
    ledger.record_vote("b", vote(c.id, "dispute"))
    json.dumps(ledger.snapshot())
