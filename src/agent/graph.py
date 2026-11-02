"""The agent as a LangGraph state machine.

    think ──(read tools)──> think            read-only tools run immediately
      │ (write tools proposed)
      v
    review ── interrupt: a person approves or rejects each proposed change
      v
    act ── runs approved changes (idempotency key per call) ──> think
    think ──(no tools)──> done               the model's summary is the result
    think ──(step limit)──> escalate ── interrupt: a person resumes or stops the run

Runs are durable: with a Postgres checkpointer a run survives restarts and can wait days for a
review. Every model turn, tool call and decision is in the run's trace for audit and evals.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from importlib.resources import files
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from .model import Model
from .tools import Tool, ToolError, ToolHost

APPROVAL_DECISIONS = ("approve", "reject")
TAKEOVER_DECISIONS = ("resume", "stop")


class RunState(TypedDict, total=False):
    run_id: str
    task: dict[str, Any]  # the triggering event
    messages: Annotated[list[dict[str, Any]], add_messages]
    steps: int
    max_steps: int
    pending: list[dict[str, Any]]  # proposed changes awaiting review: tool_use blocks
    read_results: list[dict[str, Any]]  # results of reads made in the same turn as the proposals
    status: str  # running | awaiting_review | needs_attention | done | stopped
    result: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def add_messages(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append messages, joining consecutive turns of the same role (the Messages API alternates
    user and assistant), so a reviewer's note can follow tool results."""
    out = list(left or [])
    for m in right or []:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": _blocks(out[-1]["content"]) + _blocks(m["content"])}
        else:
            out.append(m)
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _event(kind: str, **data: Any) -> dict[str, Any]:
    return {"at": _now(), "kind": kind, **data}


def _tool_result(tool_use_id: str, content: str, error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if error:
        out["is_error"] = True
    return out


def system_prompt(product: str) -> str:
    return files("agent").joinpath("prompts/system.md").read_text().format(product=product)


def build_graph(*, model: Model, tools: ToolHost, product: str, checkpointer: Any = None) -> Any:
    def catalog() -> dict[str, Tool]:
        return {t.name: t for t in tools.tools()}

    def run_reads(
        run_id: str, calls: list[dict[str, Any]], known: dict[str, Tool]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for c in calls:
            try:
                out = tools.call(known[c["name"]], c.get("input") or {}, f"{run_id}:{c['id']}")
                results.append(_tool_result(c["id"], out))
                trace.append(_event("tool_call", tool=c["name"], input=c.get("input"), ok=True))
            except ToolError as exc:
                results.append(_tool_result(c["id"], str(exc), error=True))
                trace.append(
                    _event("tool_call", tool=c["name"], input=c.get("input"), ok=False, error=str(exc))
                )
        return results, trace

    def start(state: RunState) -> dict[str, Any]:
        task = state["task"]
        text = task.get("text") or str(task)
        return {
            "messages": [{"role": "user", "content": text}],
            "steps": 0,
            "status": "running",
            "trace": [_event("started", task_type=task.get("type", ""))],
        }

    def think(state: RunState) -> Command[Literal["think", "review", "escalate", "__end__"]]:
        steps = state.get("steps", 0)
        if steps >= state.get("max_steps", 8):
            return Command(goto="escalate")
        known = catalog()
        reply = model.respond(
            system=system_prompt(product),
            messages=state["messages"],
            tools=[t.definition() for t in known.values()],
            labels={"run": state["run_id"]},
        )
        content = reply.get("content") or []
        assistant = {"role": "assistant", "content": content}
        usage = reply.get("usage") or {}
        trace = [
            _event(
                "model_turn",
                step=steps + 1,
                stop_reason=reply.get("stop_reason"),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_tokens=usage.get("cache_read_input_tokens"),
            )
        ]
        calls = [b for b in content if b.get("type") == "tool_use"]
        unknown = [c for c in calls if c["name"] not in known]
        reads = [c for c in calls if c["name"] in known and known[c["name"]].read_only]
        writes = [c for c in calls if c["name"] in known and not known[c["name"]].read_only]
        base = {"steps": steps + 1, "messages": [assistant]}
        if not calls:
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            return Command(
                goto="__end__",
                update={**base, "status": "done", "result": text, "trace": [*trace, _event("done")]},
            )
        results, read_trace = run_reads(state["run_id"], reads, known)
        results += [
            _tool_result(c["id"], f"Tool {c['name']} is not available to this agent.", error=True)
            for c in unknown
        ]
        trace += read_trace + [_event("tool_refused", tool=c["name"]) for c in unknown]
        if writes:
            return Command(
                goto="review",
                update={
                    **base,
                    "pending": writes,
                    "read_results": results,
                    "status": "awaiting_review",
                    "trace": [*trace, _event("proposed", tools=[w["name"] for w in writes])],
                },
            )
        return Command(
            goto="think",
            update={**base, "messages": [assistant, {"role": "user", "content": results}], "trace": trace},
        )

    def review(state: RunState) -> Command[Literal["act"]]:
        known = catalog()
        decision = interrupt(
            {
                "type": "approval",
                "run_id": state["run_id"],
                "actions": [
                    {
                        "id": w["id"],
                        "tool": w["name"],
                        "input": w.get("input"),
                        "description": known[w["name"]].description if w["name"] in known else "",
                    }
                    for w in state.get("pending", [])
                ],
                "summary": next(
                    (b.get("text", "") for b in state["messages"][-1]["content"] if b.get("type") == "text"),
                    "",
                ),
                "allowed": list(APPROVAL_DECISIONS),
            }
        )
        decision = normalize(decision, APPROVAL_DECISIONS)
        return Command(
            goto="act",
            update={
                "trace": [_event("review", **decision)],
                "status": "running",
                "pending": [
                    {**w, "_approved": _approved(decision, w["id"]), "_note": decision.get("note") or ""}
                    for w in state.get("pending", [])
                ],
            },
        )

    def act(state: RunState) -> Command[Literal["think"]]:
        known = catalog()
        results = list(state.get("read_results") or [])
        trace = []
        for w in state.get("pending", []):
            if not w.get("_approved"):
                reason = f" Reviewer's note: {w['_note']}" if w.get("_note") else ""
                results.append(
                    _tool_result(
                        w["id"], f"A reviewer rejected this change. Do not retry it.{reason}", error=True
                    )
                )
                continue
            try:
                out = tools.call(known[w["name"]], w.get("input") or {}, f"{state['run_id']}:{w['id']}")
                results.append(_tool_result(w["id"], out))
                trace.append(
                    _event("tool_call", tool=w["name"], input=w.get("input"), ok=True, approved=True)
                )
            except ToolError as exc:
                results.append(_tool_result(w["id"], str(exc), error=True))
                trace.append(
                    _event("tool_call", tool=w["name"], input=w.get("input"), ok=False, error=str(exc))
                )
        return Command(
            goto="think",
            update={
                "messages": [{"role": "user", "content": results}],
                "pending": [],
                "read_results": [],
                "trace": trace,
            },
        )

    def escalate(state: RunState) -> Command[Literal["think", "__end__"]]:
        decision = normalize(
            interrupt(
                {
                    "type": "takeover",
                    "run_id": state["run_id"],
                    "reason": f"The agent used its {state.get('max_steps', 8)} steps without finishing.",
                    "allowed": list(TAKEOVER_DECISIONS),
                }
            ),
            TAKEOVER_DECISIONS,
        )
        if decision["decision"] == "stop":
            return Command(
                goto="__end__",
                update={
                    "status": "stopped",
                    "result": decision.get("note", ""),
                    "trace": [_event("stopped", reviewer=decision.get("reviewer"))],
                },
            )
        note = decision.get("note") or ""
        return Command(
            goto="think",
            update={
                "status": "running",
                "max_steps": state.get("steps", 0) + state.get("max_steps", 8),
                "messages": [{"role": "user", "content": f"Note from the reviewer: {note}"}] if note else [],
                "trace": [_event("resumed", reviewer=decision.get("reviewer"))],
            },
        )

    g = StateGraph(RunState)
    g.add_node("start", start)
    g.add_node("think", think)
    g.add_node("review", review)
    g.add_node("act", act)
    g.add_node("escalate", escalate)
    g.add_edge(START, "start")
    g.add_edge("start", "think")
    return g.compile(checkpointer=checkpointer, name="agent")


class InvalidDecision(ValueError):
    pass


def normalize(raw: Any, allowed: tuple[str, ...]) -> dict[str, Any]:
    d = {"decision": raw} if isinstance(raw, str) else dict(raw or {})
    if d.get("decision") not in allowed:
        raise InvalidDecision(f"decision must be one of {allowed}")
    return d


def _approved(decision: dict[str, Any], tool_use_id: str) -> bool:
    if decision["decision"] != "approve":
        return False
    ids = decision.get("approved_ids")
    return ids is None or tool_use_id in ids
