"""The token gate.

Reaching the UI from a phone means binding past localhost, which puts
`POST /api/sessions` — and the provider credits behind it — in front of
everyone else on the network. These tests pin the gate's behaviour.
"""

import pytest
from fastapi.testclient import TestClient

from orchestra import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_SIMULATE", "1")
    monkeypatch.setenv("ORCHESTRA_TOKEN", "s3cret")
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def open_client(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_SIMULATE", "1")
    monkeypatch.delenv("ORCHESTRA_TOKEN", raising=False)
    with TestClient(server.app) as c:
        yield c


def test_api_is_refused_without_a_token(client):
    assert client.get("/api/models").status_code == 401


def test_api_is_refused_with_the_wrong_token(client):
    r = client.get("/api/models", headers={"X-Orchestra-Token": "guess"})
    assert r.status_code == 401


def test_header_token_is_accepted(client):
    r = client.get("/api/models", headers={"X-Orchestra-Token": "s3cret"})
    assert r.status_code == 200
    assert len(r.json()["models"]) >= 15


def test_query_param_token_is_accepted(client):
    """EventSource cannot set headers, so the stream relies on this form."""
    assert client.get("/api/models?t=s3cret").status_code == 200


def test_starting_a_run_requires_the_token(client):
    body = {"task": "x", "orchestrator": "claude-opus-5", "panel": ["gpt-5"]}
    assert client.post("/api/sessions", json=body).status_code == 401
    ok = client.post("/api/sessions", json=body, headers={"X-Orchestra-Token": "s3cret"})
    assert ok.status_code == 200


def test_the_static_shell_stays_reachable(client):
    """The page can't do anything without the token, so gating it only breaks
    the flow that hands the phone its credential."""
    assert client.get("/").status_code == 200


def test_no_token_configured_leaves_the_api_open(open_client):
    assert open_client.get("/api/models").status_code == 200


def test_unknown_session_still_404s_when_authorised(client):
    r = client.get("/api/sessions/nope", headers={"X-Orchestra-Token": "s3cret"})
    assert r.status_code == 404
