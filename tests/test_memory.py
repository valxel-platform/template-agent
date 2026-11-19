"""Long-term memory (memory.py): a person's, in their tenant, kept for a while, theirs to delete."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent.api import create_app
from agent.memory import InMemory, PostgresMemory, scope_of
from agent.model import ScriptedModel
from agent.settings import Settings
from agent.tools import McpToolHost

from .conftest import reply, text, tool_use

TOKENS = {
    "nurse": {"sub": "nurse-7", "scope": "agent:run", "organization": ["north-clinic"]},
    "nurse-south": {"sub": "nurse-7", "scope": "agent:run", "organization": ["south-clinic"]},
    "other": {"sub": "nurse-9", "scope": "agent:run", "organization": ["north-clinic"]},
}
REMEMBER = tool_use("tu_m1", "remember", {"fact": "I prefer summaries as three bullet points"})
RECALL = tool_use("tu_m2", "recall", {"query": "summaries"})


def auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {who}"}


def app_with(tools: McpToolHost, responses: list[Any], **settings: Any) -> tuple[TestClient, ScriptedModel]:
    model = ScriptedModel(responses)
    app = create_app(
        Settings(name="intake-agent", **settings),
        model=model,
        tools=tools,
        verifier=lambda t: TOKENS[t],
        checkpointer=InMemorySaver(),
        synchronous=True,
    )
    return TestClient(app), model


def say(client: TestClient, who: str, words: str) -> None:
    assert client.post("/conversations", json={"text": words}, headers=auth(who)).status_code == 202


def test_off_unless_set(tools: McpToolHost) -> None:
    client, model = app_with(tools, [reply(text("Hello."))])
    say(client, "nurse", "Hi")
    assert {t["name"] for t in model.calls[0]["tools"]} & {"remember", "recall"} == set()
    assert client.get("/memories", headers=auth("nurse")).json() == {"memories": [], "enabled": False}


def test_a_person_keeps_what_they_ask_and_only_they_see_it(tools: McpToolHost) -> None:
    def recalled(messages: list[dict[str, Any]]) -> dict[str, Any]:
        result = messages[-1]["content"][0]["content"]
        assert '<untrusted source="tool:memory">' in result and "three bullet points" in result
        return reply(text("You like three bullet points."))

    client, model = app_with(
        tools,
        [
            reply(REMEMBER),
            reply(text("I'll remember that.")),
            reply(RECALL),
            recalled,
            reply(RECALL),
            reply(text("Nothing yet.")),
        ],
        memory=True,
    )
    say(client, "nurse", "Remember: I prefer summaries as three bullet points")
    assert {"remember", "recall"} <= {t["name"] for t in model.calls[0]["tools"]}
    [m] = client.get("/memories", headers=auth("nurse")).json()["memories"]
    assert m["fact"] == "I prefer summaries as three bullet points"

    say(client, "nurse", "How do I like summaries?")  # a new conversation recalls it
    # Someone else in the same tenant, and the same person in another tenant, see nothing.
    assert client.get("/memories", headers=auth("other")).json()["memories"] == []
    assert client.get("/memories", headers=auth("nurse-south")).json()["memories"] == []
    say(client, "other", "How do I like summaries?")
    assert "Nothing remembered." in model.calls[-1]["messages"][-1]["content"][0]["content"]
    assert client.delete(f"/memories/{m['id']}", headers=auth("other")).status_code == 404

    assert client.delete(f"/memories/{m['id']}", headers=auth("nurse")).status_code == 204
    assert client.get("/memories", headers=auth("nurse")).json()["memories"] == []


def test_runs_have_no_memory(tools: McpToolHost) -> None:
    client, model = app_with(tools, [reply(text("Done."))], memory=True)
    r = client.post(
        "/events", json={"id": "e1", "type": "lab.result", "text": "New lab"}, headers=auth("nurse")
    )
    assert r.status_code in (200, 202)
    assert {t["name"] for t in model.calls[0]["tools"]} & {"remember", "recall"} == set()


def test_retention_cap_and_scope() -> None:
    m = InMemory(retention_days=30, max_per_person=2)
    for fact in ("one thing", "two things", "three things"):
        m.remember("p", "t", fact)
    assert [x["fact"] for x in m.mine("p", "t")] == ["three things", "two things"]  # oldest dropped
    m.remember("p", "t", "Two  things")  # the same fact again is not a second memory
    assert len(m.mine("p", "t")) == 2
    m.rows[0]["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    assert m.prune() == 1 and len(m.mine("p", "t")) == 1
    assert scope_of(["b", "a", "a"]) == "a,b" and scope_of(None) == ""
    with pytest.raises(ValueError):
        m.remember("p", "t", "x" * 501)


DSN = os.environ.get("AGENT_TEST_DATABASE_URL", "")


@pytest.mark.skipif(not DSN, reason="set AGENT_TEST_DATABASE_URL to test against Postgres")
def test_row_level_security_keeps_people_and_tenants_apart() -> None:
    """As an ordinary role, as deployed: the table's own policy hides everyone else's rows."""
    role, password = f"agent_mem_{uuid.uuid4().hex[:8]}", uuid.uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f"create role {role} login password '{password}'")
        admin.execute(f"grant create, usage on schema public to {role}")
        admin.execute("drop table if exists agent_memories")
    conninfo = psycopg.conninfo.make_conninfo(DSN, user=role, password=password)
    pool = ConnectionPool(
        conninfo, min_size=1, max_size=2, open=True, kwargs={"autocommit": True, "row_factory": dict_row}
    )
    try:
        mem = PostgresMemory(pool, retention_days=30, max_per_person=2)
        mem.remember("ana", "north", "prefers bullet summaries")
        mem.remember("ana", "south", "south clinic note")
        mem.remember("bo", "north", "bo's own preference")
        assert [r["fact"] for r in mem.recall("ana", "north", "summaries")] == ["prefers bullet summaries"]
        assert [r["fact"] for r in mem.mine("ana", "south")] == ["south clinic note"]
        # Without a person set, a raw query sees nothing: the policy, not the code's where clause.
        with pool.connection() as conn:
            assert conn.execute("select count(*) as n from agent_memories").fetchone()["n"] == 0
        assert mem.forget("bo", "north", mem.mine("ana", "north")[0]["id"]) == 0
        mem.remember("ana", "north", "second")
        mem.remember("ana", "north", "third")
        assert [r["fact"] for r in mem.mine("ana", "north")] == ["third", "second"]
        with pool.connection() as conn, conn.transaction():
            conn.execute(
                "select set_config('agent.memory_owner', 'ana', true),"
                " set_config('agent.memory_tenant', 'north', true)"
            )
            conn.execute(
                "update agent_memories set expires_at = now() - interval '1 second' where fact = 'second'"
            )
        assert mem.prune() == 1
        assert [r["fact"] for r in mem.mine("ana", "north")] == ["third"]
        assert mem.forget("ana", "north") == 1 and mem.mine("bo", "north")
    finally:
        pool.close()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute("drop table if exists agent_memories")
            admin.execute(f"revoke all on schema public from {role}")
            admin.execute(f"drop role {role}")
