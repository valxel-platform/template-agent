from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent.api import create_app
from agent.model import ScriptedModel
from agent.settings import Settings
from agent.tools import McpToolHost
from evals.world import World

from .conftest import CREATE, FIND, reply, text

TOKENS = {
    "runner": {"sub": "svc-intake", "scope": "agent:run"},
    "reviewer": {"sub": "nurse-7", "scope": "agent:run agent:review"},
}


def verify(token: str) -> dict[str, Any]:
    if token not in TOKENS:
        raise ValueError("bad token")
    return TOKENS[token]


def auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {who}"}


@pytest.fixture
def setup(world: World, tools: McpToolHost) -> tuple[TestClient, ScriptedModel]:
    model = ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))])
    app = create_app(
        Settings(name="intake-agent"),
        model=model,
        tools=tools,
        verifier=verify,
        checkpointer=InMemorySaver(),
        synchronous=True,
    )
    return TestClient(app), model


def test_run_review_and_finish(setup: tuple[TestClient, ScriptedModel], world: World) -> None:
    client, _ = setup
    r = client.post(
        "/runs",
        json={"task": {"type": "referral", "text": "Ana Lopez"}},
        headers={**auth("runner"), "Idempotency-Key": "ref-1"},
    )
    assert r.status_code == 202
    run = r.json()
    assert run["status"] == "awaiting_review"
    assert run["waiting_for"]["actions"][0]["tool"] == "care__create_task"

    # A retry with the same key returns the same run and starts nothing new.
    again = client.post(
        "/runs", json={"task": {"text": "Ana Lopez"}}, headers={**auth("runner"), "Idempotency-Key": "ref-1"}
    ).json()
    assert again["run_id"] == run["run_id"]

    # Deciding needs the review scope.
    assert (
        client.post(
            f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("runner")
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/runs/{run['run_id']}/decision", json={"decision": "resume"}, headers=auth("reviewer")
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/runs/{run['run_id']}/decision",
            json={"decision": "approve", "approved_ids": ["x"]},
            headers=auth("reviewer"),
        ).status_code
        == 422
    )
    done = client.post(
        f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("reviewer")
    ).json()
    assert done["status"] == "done" and done["result"] == "Created."
    assert len(world.called("create_task")) == 1
    review = next(e for e in done["trace"] if e["kind"] == "review")
    assert review["reviewer"] == "nurse-7"
    assert (
        client.post(
            f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("reviewer")
        ).status_code
        == 409
    )

    metrics = client.get("/metrics").text
    assert "agent_runs_started_total 1.0" in metrics
    assert 'agent_decisions_total{decision="approve"} 1.0' in metrics


def test_events_are_idempotent_by_event_id(setup: tuple[TestClient, ScriptedModel]) -> None:
    client, model = setup
    body = {"id": "evt-9", "type": "referral.received", "text": "Ana Lopez"}
    a = client.post("/events", json=body, headers=auth("runner")).json()
    b = client.post("/events", json=body, headers=auth("runner")).json()
    assert a["run_id"] == b["run_id"] and len(model.calls) == 2  # find, then propose: once


def test_auth_is_required(setup: tuple[TestClient, ScriptedModel]) -> None:
    client, _ = setup
    assert client.post("/runs", json={"task": {}}).status_code == 401
    assert client.post("/runs", json={"task": {}}, headers=auth("nobody")).status_code == 401
    assert client.get("/runs/missing", headers=auth("runner")).status_code == 404


def test_health(setup: tuple[TestClient, ScriptedModel]) -> None:
    client, _ = setup
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready", "tools": 3}


def test_missing_oidc_config_fails_at_start(tools: McpToolHost) -> None:
    with pytest.raises(RuntimeError, match="OIDC"):
        create_app(Settings(), model=ScriptedModel([]), tools=tools, checkpointer=InMemorySaver())
