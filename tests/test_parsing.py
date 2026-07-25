from orchestra.providers import extract_json
from orchestra.schemas import ProposedClaim, Vote


def test_bare_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_preamble_and_epilogue():
    text = 'Sure! Here is my answer:\n\n{"position": "x"}\n\nLet me know if that helps.'
    assert extract_json(text) == {"position": "x"}


def test_trailing_comma_is_repaired():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_smart_quotes_are_repaired():
    assert extract_json('{“a”: 1}') == {"a": 1}


def test_nested_braces_are_balanced_correctly():
    text = 'prose {"outer": {"inner": [1, 2]}} more prose'
    assert extract_json(text) == {"outer": {"inner": [1, 2]}}


def test_braces_inside_strings_do_not_break_matching():
    assert extract_json('{"text": "use {this} form"}') == {"text": "use {this} form"}


def test_prose_only_returns_none():
    assert extract_json("I think the answer is probably yes.") is None


def test_empty_returns_none():
    assert extract_json("") is None


def test_vote_stance_synonyms_are_normalised():
    assert Vote(claim_id="C001", stance="agree").stance.value == "endorse"
    assert Vote(claim_id="C001", stance="Disagree").stance.value == "dispute"
    assert Vote(claim_id="C001", stance="unsure").stance.value == "abstain"


def test_confidence_is_clamped():
    assert Vote(claim_id="C001", confidence=5).confidence == 1.0
    assert Vote(claim_id="C001", confidence=-2).confidence == 0.0
    assert Vote(claim_id="C001", confidence="nonsense").confidence == 0.5
    assert ProposedClaim(text="x", confidence="0.7").confidence == 0.7
