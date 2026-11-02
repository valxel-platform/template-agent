from __future__ import annotations

import itertools
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import InvalidDecision, add_messages, build_graph, normalize
from agent.model import ScriptedModel
from agent.tools import McpToolHost
from evals.world import World

from .conftest import CREATE, FIND, reply, text, tool_use

CFG = {"configurable": {"thread_id": "r1"}}


def make(model: ScriptedModel, tools: McpToolHost) -> Any:
    return build_graph(model=model, tools=tools, product="EMR", checkpointer=InMemorySaver())


def start(graph: Any, max_steps: int = 8) -> None:
    graph.invoke(
        {
            "run_id": "r1",
            "task": {"type": "referral", "text": "Ana Lopez 1984-03-02"},
            "max_steps": max_steps,
        },
        CFG,
    )


def test_tools_are_filtered_and_classified(tools: McpToolHost) -> None:
    by_name = {t.name: t for t in tools.tools()}
    assert set(by_name) == {"care__find_patient", "care__list_open_tasks", "care__create_task"}
    assert by_name["care__find_patient"].read_only
    assert not by_name["care__create_task"].read_only


def test_reads_run_at_once_and_writes_wait_for_review(world: World, tools: McpToolHost) -> None:
    model = ScriptedModel(
        [reply(FIND), reply(text("Proposing a task."), CREATE), reply(text("Created a task."))]
    )
    graph = make(model, tools)
    start(graph)
    snap = graph.get_state(CFG)
    assert world.called("find_patient") and not world.called("create_task")
    waiting = snap.interrupts[0].value
    assert waiting["type"] == "approval" and waiting["actions"][0]["tool"] == "care__create_task"
    assert waiting["summary"] == "Proposing a task."
    assert snap.values["status"] == "awaiting_review"

    graph.invoke(Command(resume={"decision": "approve", "reviewer": "u-1"}), CFG)
    v = graph.get_state(CFG).values
    assert v["status"] == "done" and v["result"] == "Created a task."
    (call,) = world.called("create_task")
    assert call["idempotency_key"] == "r1:tu_2"  # stable across retries of the same proposal
    assert {"kind": "review", "decision": "approve", "reviewer": "u-1"}.items() <= next(
        e for e in v["trace"] if e["kind"] == "review"
    ).items()
    # The model saw the tool result paired with its tool_use id.
    last_user = model.calls[-1]["messages"][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["tool_use_id"] == "tu_2"


def test_rejection_tells_the_model_and_changes_nothing(world: World, tools: McpToolHost) -> None:
    model = ScriptedModel([reply(CREATE), reply(text("Not created."))])
    graph = make(model, tools)
    start(graph)
    graph.invoke(Command(resume={"decision": "reject", "note": "wrong clinic"}), CFG)
    assert not world.called("create_task")
    result = model.calls[-1]["messages"][-1]["content"][0]
    assert result["is_error"] and "wrong clinic" in result["content"]
    assert graph.get_state(CFG).values["status"] == "done"


def test_partial_approval(world: World, tools: McpToolHost) -> None:
    other = tool_use(
        "tu_3", "care__create_task", {"patient_id": "p-102", "title": "Labs", "priority": "routine"}
    )
    model = ScriptedModel([reply(CREATE, other), reply(text("One created."))])
    graph = make(model, tools)
    start(graph)
    graph.invoke(Command(resume={"decision": "approve", "approved_ids": ["tu_3"]}), CFG)
    assert [c["title"] for c in world.called("create_task")] == ["Labs"]


def test_tools_outside_the_allowlist_are_refused(world: World, tools: McpToolHost) -> None:
    model = ScriptedModel(
        [reply(tool_use("tu_9", "care__delete_task", {"task_id": "t-1"})), reply(text("ok"))]
    )
    graph = make(model, tools)
    start(graph)
    assert not world.called("delete_task")
    assert "not available" in model.calls[-1]["messages"][-1]["content"][0]["content"]


def test_tool_errors_go_back_to_the_model(tools: McpToolHost) -> None:
    bad = tool_use("tu_1", "care__find_patient", {"name": "Ana"})  # missing date_of_birth
    model = ScriptedModel([reply(bad), reply(text("Need the date of birth."))])
    graph = make(model, tools)
    start(graph)
    assert model.calls[-1]["messages"][-1]["content"][0]["is_error"]
    assert graph.get_state(CFG).values["result"] == "Need the date of birth."


def test_step_limit_asks_a_person(tools: McpToolHost) -> None:
    model = ScriptedModel([reply(FIND), reply(FIND), reply(text("Done after help."))])
    graph = make(model, tools)
    start(graph, max_steps=2)
    snap = graph.get_state(CFG)
    assert snap.interrupts[0].value["type"] == "takeover"
    graph.invoke(Command(resume={"decision": "resume", "note": "Use the referral's DOB."}), CFG)
    v = graph.get_state(CFG).values
    assert v["status"] == "done"
    # The note joined the last user turn instead of breaking role alternation.
    msgs = model.calls[-1]["messages"]
    assert all(a["role"] != b["role"] for a, b in itertools.pairwise(msgs))
    assert "Use the referral's DOB." in str(msgs[-1]["content"])


def test_step_limit_stop(tools: McpToolHost) -> None:
    graph = make(ScriptedModel([reply(FIND)]), tools)
    start(graph, max_steps=1)
    graph.invoke(Command(resume={"decision": "stop", "note": "handled by phone"}), CFG)
    v = graph.get_state(CFG).values
    assert v["status"] == "stopped" and v["result"] == "handled by phone"


def test_normalize_and_merge() -> None:
    assert normalize("approve", ("approve", "reject")) == {"decision": "approve"}
    with pytest.raises(InvalidDecision):
        normalize({"decision": "maybe"}, ("approve", "reject"))
    merged = add_messages([{"role": "user", "content": "a"}], [{"role": "user", "content": [text("b")]}])
    assert merged == [{"role": "user", "content": [text("a"), text("b")]}]
