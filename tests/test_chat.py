"""Conversations: a person talks with the agent; replies stream; changes wait for their approval;
nobody else sees the conversation; one message at a time."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent.api import create_app
from agent.chat import history
from agent.graph import window
from agent.model import ScriptedModel
from agent.settings import Settings
from agent.tools import McpToolHost
from evals.world import World

from .conftest import CREATE, FIND, reply, text

TOKENS = {
    "nurse": {"sub": "nurse-7", "scope": "agent:run", "organization": ["north-clinic"]},
    "other": {"sub": "nurse-9", "scope": "agent:run agent:review", "organization": ["north-clinic"]},
}


def verify(token: str) -> dict[str, Any]:
    if token not in TOKENS:
        raise ValueError("bad token")
    return TOKENS[token]


def auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {who}"}


def app_with(tools: McpToolHost, responses: list[Any], **settings: Any) -> tuple[TestClient, ScriptedModel]:
    model = ScriptedModel(responses)
    app = create_app(
        Settings(name="intake-agent", **settings),
        model=model,
        tools=tools,
        verifier=verify,
        checkpointer=InMemorySaver(),
        synchronous=True,
    )
    return TestClient(app), model


def stream(client: TestClient, cid: str, after: int = 0, **headers: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    with client.stream(
        "GET", f"/conversations/{cid}/stream?after={after}", headers={**auth("nurse"), **headers}
    ) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        kind = ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                out.append((kind, json.loads(line[6:])))
    return out


def test_a_conversation_answers_and_streams_the_reply(tools: McpToolHost) -> None:
    client, _ = app_with(
        tools, [reply(text("Looking that up."), FIND), reply(text("Ana Lopez is patient p-102."))]
    )
    r = client.post(
        "/conversations", json={"text": "Who is Ana Lopez, born 1984-03-02?"}, headers=auth("nurse")
    )
    assert r.status_code == 202
    cid, after = r.json()["conversation_id"], r.json()["after"]
    assert cid.startswith("conv-")

    events = stream(client, cid, after)
    kinds = [k for k, _ in events]
    assert kinds[0] == "turn" and kinds[-1] == "idle"
    assert (
        "".join(d["delta"] for k, d in events if k == "text") == "Looking that up.Ana Lopez is patient p-102."
    )
    [step] = [d for k, d in events if k == "step"]
    assert step["tool"] == "care__find_patient" and step["ok"] and step["title"]
    assert ("reply", {"text": "Ana Lopez is patient p-102."}) in events

    got = client.get(f"/conversations/{cid}", headers=auth("nurse")).json()
    assert got["status"] == "idle" and got["waiting_for"] is None
    assert got["messages"][0] == {"role": "user", "text": "Who is Ana Lopez, born 1984-03-02?"}
    answer = got["messages"][1]
    assert answer["role"] == "assistant" and answer["text"].endswith("Ana Lopez is patient p-102.")
    assert [s["tool"] for s in answer["steps"]] == ["care__find_patient"]
    listed = client.get("/conversations", headers=auth("nurse")).json()["conversations"]
    assert [c["conversation_id"] for c in listed] == [cid] and listed[0]["title"].startswith("Who is Ana")

    # A reconnect picks up after the last event it saw; nothing new: it ends at once.
    assert [k for k, _ in stream(client, cid, **{"Last-Event-ID": str(got["after"])})] == ["idle"]


def test_changes_wait_for_the_persons_approval_in_the_conversation(tools: McpToolHost, world: World) -> None:
    client, _ = app_with(
        tools,
        [reply(FIND), reply(text("I'll create an echo task."), CREATE), reply(text("Done: task created."))],
    )
    cid = client.post("/conversations", json={"text": "Book an echo for Ana"}, headers=auth("nurse")).json()[
        "conversation_id"
    ]
    got = client.get(f"/conversations/{cid}", headers=auth("nurse")).json()
    assert got["status"] == "awaiting_review"
    assert got["waiting_for"]["actions"][0]["tool"] == "care__create_task"
    assert not world.called("create_task")
    # The next message waits for the decision.
    r = client.post(f"/conversations/{cid}/messages", json={"text": "and another"}, headers=auth("nurse"))
    assert r.status_code == 409
    bad = client.post(f"/conversations/{cid}/decision", json={"decision": "stop"}, headers=auth("nurse"))
    assert bad.status_code == 422

    r = client.post(f"/conversations/{cid}/decision", json={"decision": "approve"}, headers=auth("nurse"))
    assert r.status_code == 202
    events = stream(client, cid, r.json()["after"])
    assert ("reply", {"text": "Done: task created."}) in events
    assert len(world.called("create_task")) == 1
    assert client.get(f"/conversations/{cid}", headers=auth("nurse")).json()["status"] == "idle"
    nothing = client.post(
        f"/conversations/{cid}/decision", json={"decision": "approve"}, headers=auth("nurse")
    )
    assert nothing.status_code == 409


def test_a_conversation_is_its_owners_alone(tools: McpToolHost) -> None:
    client, _ = app_with(tools, [reply(text("Hello."))])
    cid = client.post("/conversations", json={"text": "hi"}, headers=auth("nurse")).json()["conversation_id"]
    for method, path, body in (
        ("get", f"/conversations/{cid}", None),
        ("post", f"/conversations/{cid}/messages", {"text": "hi"}),
        ("post", f"/conversations/{cid}/decision", {"decision": "approve"}),
        ("get", f"/conversations/{cid}/stream", None),
        ("get", "/conversations/run-not-conv", None),
    ):
        r = client.request(method, path, json=body, headers=auth("other"))
        assert r.status_code == 404, path
    assert client.get("/conversations", headers=auth("other")).json() == {"conversations": []}
    assert client.get("/conversations").status_code == 401


def test_one_message_at_a_time_and_messages_have_a_size(tools: McpToolHost) -> None:
    client, _ = app_with(tools, [reply(text("Hello.")), reply(text("Again."))], chat_max_chars=20)
    cid = client.post("/conversations", json={"text": "hi"}, headers=auth("nurse")).json()["conversation_id"]
    runner = client.app.state.runner  # type: ignore[attr-defined]
    lease = runner.locks.acquire(cid)  # another request (on this or another pod) is answering
    assert client.get(f"/conversations/{cid}", headers=auth("nurse")).json()["status"] == "working"
    busy = client.post(f"/conversations/{cid}/messages", json={"text": "more"}, headers=auth("nurse"))
    assert busy.status_code == 409
    lease.release()
    ok = client.post(f"/conversations/{cid}/messages", json={"text": "more"}, headers=auth("nurse"))
    assert ok.status_code == 202
    long = client.post(f"/conversations/{cid}/messages", json={"text": "x" * 21}, headers=auth("nurse"))
    assert long.status_code == 422
    blank = client.post(f"/conversations/{cid}/messages", json={"text": "   "}, headers=auth("nurse"))
    assert blank.status_code == 422


def test_a_retried_start_is_the_same_conversation(tools: McpToolHost) -> None:
    client, model = app_with(tools, [reply(text("Hello."))])
    h = {**auth("nurse"), "Idempotency-Key": "k1"}
    a = client.post("/conversations", json={"text": "hi"}, headers=h).json()
    b = client.post("/conversations", json={"text": "hi"}, headers=h).json()
    assert a["conversation_id"] == b["conversation_id"] and len(model.calls) == 1


def test_the_model_sees_the_recent_exchanges_only(tools: McpToolHost) -> None:
    client, model = app_with(tools, [reply(text(f"A{i}.")) for i in range(4)], chat_history_turns=2)
    cid = client.post("/conversations", json={"text": "Q0"}, headers=auth("nurse")).json()["conversation_id"]
    for i in range(1, 4):
        client.post(f"/conversations/{cid}/messages", json={"text": f"Q{i}"}, headers=auth("nurse"))
    seen = model.calls[-1]["messages"]
    assert [m["content"] for m in seen if m["role"] == "user"] == ["Q2", "Q3"]
    full = client.get(f"/conversations/{cid}", headers=auth("nurse")).json()["messages"]
    assert [m["text"] for m in full if m["role"] == "user"] == ["Q0", "Q1", "Q2", "Q3"]
    assert (
        "chat.md" not in model.calls[-1]["system"]
        and "assistant for intake-agent" in model.calls[-1]["system"]
    )


def test_window_and_history_keep_tool_calls_whole() -> None:
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": [FIND]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "{}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    assert window(msgs, 1) == msgs[4:] and window(msgs, 5) == msgs
    h = history({"messages": msgs}, {"care__find_patient": "Find a patient"})
    assert h == [
        {"role": "user", "text": "Q1"},
        {
            "role": "assistant",
            "text": "A1",
            "steps": [{"tool": "care__find_patient", "title": "Find a patient"}],
        },
        {"role": "user", "text": "Q2"},
        {"role": "assistant", "text": "A2", "steps": []},
    ]


@pytest.mark.parametrize("decision", ["reject"])
def test_a_declined_change_is_not_made(tools: McpToolHost, world: World, decision: str) -> None:
    client, _ = app_with(tools, [reply(CREATE), reply(text("Understood, nothing changed."))])
    cid = client.post("/conversations", json={"text": "make a task"}, headers=auth("nurse")).json()[
        "conversation_id"
    ]
    client.post(
        f"/conversations/{cid}/decision",
        json={"decision": decision, "note": "not now"},
        headers=auth("nurse"),
    )
    assert not world.called("create_task")
    assert (
        client.get(f"/conversations/{cid}", headers=auth("nurse"))
        .json()["messages"][-1]["text"]
        .startswith("Understood")
    )
