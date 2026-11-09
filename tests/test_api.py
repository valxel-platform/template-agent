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
    "runner": {"sub": "svc-intake", "scope": "agent:run", "organization": {"north-clinic": {}}},
    "reviewer": {"sub": "nurse-7", "scope": "agent:run agent:review", "organization": {"north-clinic": {}}},
    "other-runner": {"sub": "svc-billing", "scope": "agent:run", "organization": {"north-clinic": {}}},
    "other-tenant": {"sub": "nurse-9", "scope": "agent:run agent:review", "organization": ["south-clinic"]},
    "self-reviewer": {"sub": "nurse-8", "scope": "agent:run agent:review", "organization": "north-clinic"},
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


def test_people_and_changes_are_audited_and_the_run_is_traced(world: World, tools: McpToolHost) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from agent import telemetry
    from agent.audit import Auditor, MemorySink

    spans = InMemorySpanExporter()
    telemetry.setup("intake-agent", exporter=spans)
    sink = MemorySink()
    model = ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))])
    app = create_app(
        Settings(name="intake-agent"),
        model=model,
        tools=tools,
        verifier=verify,
        checkpointer=InMemorySaver(),
        synchronous=True,
        auditor=Auditor(sink, service="intake-agent", product="emr", data_class="phi"),
    )
    client = TestClient(app)
    run = client.post("/runs", json={"task": {"text": "Ana Lopez"}}, headers=auth("runner")).json()
    client.post(f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("reviewer"))

    data = [e["data"] for e in sink.events]
    assert [(d["action"], d["actor"]["id"], d["actor"]["type"], d["outcome"]) for d in data] == [
        ("agent.run.start", "svc-intake", "user", "success"),
        ("agent.change.approve", "nurse-7", "user", "success"),
        ("agent.change.apply", "intake-agent", "agent", "success"),
    ]
    assert data[1]["details"] == {"approved": ["care__create_task"], "rejected": []}
    # The change ran with the approver's delegated token, so it was done on their behalf.
    assert data[2]["on_behalf_of"] == "nurse-7" and data[2]["details"]["tool"] == "care__create_task"
    assert data[2]["details"]["requested_by"] == "svc-intake"
    assert all(d["resource"] == {"type": "run", "ids": [run["run_id"]]} for d in data)
    assert "Ana Lopez" not in str(sink.events)

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names.count("chat") == 3 and "execute_tool care__find_patient" in names
    assert "execute_tool care__create_task" in names and "invoke_agent intake-agent" in names
    chat = next(s for s in finished if s.name == "chat")
    assert (
        chat.attributes["gen_ai.usage.input_tokens"] == 10
        and chat.attributes["gen_ai.conversation.id"] == run["run_id"]
    )
    assert "Ana Lopez" not in str([dict(s.attributes or {}) for s in finished])  # no content in spans
    # The run's spans are part of the request that started it.
    post = next(s for s in finished if s.name == "POST /runs")
    invoke = next(s for s in finished if s.name == "invoke_agent intake-agent")
    assert invoke.context.trace_id == post.context.trace_id


def test_a_decision_that_cant_be_audited_is_not_applied(world: World, tools: McpToolHost) -> None:
    from agent.audit import Auditor, MemorySink

    sink = MemorySink()
    model = ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))])
    client = TestClient(
        create_app(
            Settings(name="intake-agent"),
            model=model,
            tools=tools,
            verifier=verify,
            checkpointer=InMemorySaver(),
            synchronous=True,
            auditor=Auditor(sink, service="intake-agent"),
        )
    )
    run = client.post("/runs", json={"task": {"text": "Ana Lopez"}}, headers=auth("runner")).json()
    sink.fail = True
    r = client.post(f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("reviewer"))
    # The decision wasn't recorded, so it wasn't applied: the run still waits for it.
    assert r.json()["status"] == "awaiting_review" and "audit trail" in r.json()["problem"]
    assert world.called("create_task") == []
    sink.fail = False
    r = client.post(f"/runs/{run['run_id']}/decision", json={"decision": "approve"}, headers=auth("reviewer"))
    assert (
        r.json()["status"] == "done" and r.json()["problem"] == "" and len(world.called("create_task")) == 1
    )


def test_runs_are_visible_to_their_requester_and_reviewers_of_the_same_tenant(
    setup: tuple[TestClient, ScriptedModel],
) -> None:
    client, _ = setup
    run = client.post("/runs", json={"task": {"text": "Ana Lopez"}}, headers=auth("runner")).json()
    rid = run["run_id"]
    assert client.get(f"/runs/{rid}", headers=auth("runner")).status_code == 200
    assert client.get(f"/runs/{rid}", headers=auth("reviewer")).status_code == 200
    # Another service of the same tenant without the review scope, and a reviewer of another
    # tenant, can't tell the run exists.
    assert client.get(f"/runs/{rid}", headers=auth("other-runner")).status_code == 404
    assert client.get(f"/runs/{rid}", headers=auth("other-tenant")).status_code == 404
    denied = client.post(f"/runs/{rid}/decision", json={"decision": "approve"}, headers=auth("other-tenant"))
    assert denied.status_code == 404


def test_the_requester_cant_approve_their_own_change(world: World, tools: McpToolHost) -> None:
    model = ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))])
    client = TestClient(
        create_app(
            Settings(name="intake-agent"),
            model=model,
            tools=tools,
            verifier=verify,
            checkpointer=InMemorySaver(),
            synchronous=True,
        )
    )
    run = client.post("/runs", json={"task": {"text": "Ana Lopez"}}, headers=auth("self-reviewer")).json()
    assert run["status"] == "awaiting_review"
    url = f"/runs/{run['run_id']}/decision"
    r = client.post(url, json={"decision": "approve"}, headers=auth("self-reviewer"))
    assert r.status_code == 403 and not world.called("create_task")
    # Rejecting your own proposal is fine.
    r = client.post(url, json={"decision": "reject"}, headers=auth("self-reviewer"))
    assert r.status_code == 200


def test_a_retried_event_from_another_tenant_learns_nothing(setup: tuple[TestClient, ScriptedModel]) -> None:
    client, _ = setup
    body = {"id": "evt-1", "type": "referral.received", "text": "Ana Lopez"}
    assert client.post("/events", json=body, headers=auth("runner")).status_code == 202
    assert client.post("/events", json=body, headers=auth("other-tenant")).status_code == 404


def test_tools_are_denied_by_default() -> None:
    assert Settings().allowed_tools == []
