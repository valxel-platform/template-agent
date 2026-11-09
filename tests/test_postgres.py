"""Several replicas on one database: a thread executes on one pod at a time, any pod answers for
it, and a pod that dies lets go. Needs Postgres (AGENT_TEST_DATABASE_URL; CI starts one)."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent import locks
from agent.api import create_app
from agent.chat import events
from agent.locks import LeaseLost, PostgresLocks
from agent.model import ScriptedModel
from agent.settings import Settings
from agent.store import PostgresConversations, PostgresProgress
from agent.tools import McpToolHost

from .conftest import CREATE, FIND, reply, text

DSN = os.environ.get("AGENT_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set AGENT_TEST_DATABASE_URL to test against Postgres")

TOKENS = {"nurse": {"sub": "nurse-7", "scope": "agent:run", "organization": ["north-clinic"]}}


def verify(token: str) -> dict[str, Any]:
    return TOKENS[token]


AUTH = {"Authorization": "Bearer nurse"}


@pytest.fixture
def pool() -> Any:
    p = ConnectionPool(DSN, max_size=4, open=True, kwargs={"autocommit": True, "row_factory": dict_row})
    yield p
    p.close()


def pod(tools: McpToolHost, responses: list[Any], name: str) -> tuple[TestClient, ScriptedModel]:
    """One replica: its own process state, the shared database."""
    model = ScriptedModel(responses)
    app = create_app(
        Settings(name=name, database_url=DSN, database_pool=4),
        model=model,
        tools=tools,
        verifier=verify,
        synchronous=True,
    )
    return TestClient(app), model


def test_one_pod_holds_a_thread_and_every_pod_can_see_it(pool: Any) -> None:
    a, b = PostgresLocks(DSN, pool, "agent-x"), PostgresLocks(DSN, pool, "agent-x")
    other_agent = PostgresLocks(DSN, pool, "agent-y")
    thread = f"t-{uuid.uuid4()}"
    lease = a.acquire(thread)
    assert lease is not None and lease.alive()
    assert b.acquire(thread) is None and b.busy(thread) and a.busy(thread)
    assert not b.busy(f"t-{uuid.uuid4()}")
    assert not other_agent.busy(thread)  # agents sharing a database never collide
    lease.release()
    assert not b.busy(thread)
    again = b.acquire(thread)
    assert again is not None
    again.release()


def test_a_pod_that_loses_its_connection_lets_go_and_stops(pool: Any) -> None:
    a, b = PostgresLocks(DSN, pool, "agent-x"), PostgresLocks(DSN, pool, "agent-x")
    thread = f"t-{uuid.uuid4()}"
    lease = a.acquire(thread)
    assert lease is not None
    pid = lease.conn.execute("select pg_backend_pid()").fetchone()[0]  # type: ignore[attr-defined]
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute("select pg_terminate_backend(%s)", (pid,))
    assert not lease.alive()
    assert not b.busy(thread)
    taken = b.acquire(thread)
    assert taken is not None
    with locks.holding(lease), pytest.raises(LeaseLost):
        locks.check()  # the first pod stops before its next step
    with locks.holding(taken):
        locks.check()
    lease.release()  # harmless
    taken.release()


def test_progress_reaches_streams_on_other_pods(pool: Any) -> None:
    here, there = PostgresProgress(pool, DSN), PostgresProgress(pool, DSN)
    thread = f"conv-{uuid.uuid4()}"
    first = here.emit(thread, "turn", {"turn": 1})
    assert there.last(thread) == first and there.read(thread)[0]["data"] == {"turn": 1}
    here.emit(thread, "problem", {"text": "x"})
    assert there.latest(thread, ("problem",))["data"] == {"text": "x"}  # type: ignore[index]

    async def follow() -> list[str]:
        out: list[str] = []

        async def produce() -> None:
            await asyncio.sleep(0.3)
            await asyncio.to_thread(here.emit, thread, "text", {"delta": "hi"})
            await asyncio.to_thread(here.emit, thread, "idle", {"status": "idle"})

        task = asyncio.create_task(produce())
        async for line in events(thread, first + 1, lambda _: True, there):
            out.append(line)
        await task
        return out

    lines = asyncio.run(asyncio.wait_for(follow(), 10))
    assert [ln.split("\n")[1] for ln in lines] == ["event: text", "event: idle"]
    assert here.prune(0) >= 4 and there.read(thread) == []


def test_conversations_are_listed_per_person(pool: Any) -> None:
    store = PostgresConversations(pool)
    owner = f"u-{uuid.uuid4()}"
    store.add("conv-1-" + owner, owner, ["north"], "First")
    store.add("conv-2-" + owner, owner, ["north"], "Second")
    store.touch("conv-1-" + owner)
    assert [c["title"] for c in store.mine(owner)] == ["First", "Second"]


def test_a_conversation_moves_between_pods(tools: McpToolHost) -> None:
    """Pod A takes the first message and pod B the approval; pod C streams; any pod answers."""
    name = f"agent-{uuid.uuid4().hex[:8]}"
    a, _ = pod(tools, [reply(text("Let me check."), FIND), reply(text("I'll create it."), CREATE)], name)
    b, _ = pod(tools, [reply(text("Created the task."))], name)
    c, _ = pod(tools, [], name)
    cid = a.post("/conversations", json={"text": "Book an echo for Ana"}, headers=AUTH).json()[
        "conversation_id"
    ]
    seen = c.get(f"/conversations/{cid}", headers=AUTH).json()
    assert seen["status"] == "awaiting_review" and seen["messages"][1]["text"].startswith("Let me check.")

    # While pod A holds the conversation, the others refuse to move it on and say it is working.
    lease = a.app.state.runner.locks.acquire(cid)  # type: ignore[attr-defined]
    assert c.get(f"/conversations/{cid}", headers=AUTH).json()["status"] == "working"
    busy = b.post(f"/conversations/{cid}/decision", json={"decision": "approve"}, headers=AUTH)
    assert busy.status_code == 409
    lease.release()

    r = b.post(f"/conversations/{cid}/decision", json={"decision": "approve"}, headers=AUTH)
    assert r.status_code == 202
    lines = []
    with c.stream("GET", f"/conversations/{cid}/stream?after={r.json()['after']}", headers=AUTH) as s:
        lines = [ln for ln in s.iter_lines() if ln.startswith("event: ")]
    assert lines[-1] == "event: idle" and "event: reply" in lines
    final = a.get(f"/conversations/{cid}", headers=AUTH).json()
    assert final["status"] == "idle" and final["messages"][-1]["text"].endswith("Created the task.")
    listed = c.get("/conversations", headers=AUTH).json()["conversations"]
    assert listed[0]["conversation_id"] == cid


def test_a_run_started_twice_on_two_pods_runs_once(tools: McpToolHost) -> None:
    name = f"agent-{uuid.uuid4().hex[:8]}"
    a, model_a = pod(tools, [reply(text("Done."))], name)
    b, model_b = pod(tools, [reply(text("Done."))], name)
    body = {"task": {"text": "Ana"}}
    h = {**AUTH, "Idempotency-Key": f"k-{uuid.uuid4()}"}
    first = a.post("/runs", json=body, headers=h).json()
    second = b.post("/runs", json=body, headers=h).json()
    assert first["run_id"] == second["run_id"] and second["status"] == "done"
    assert len(model_a.calls) + len(model_b.calls) == 1
