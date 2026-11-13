"""The agent as a LangGraph state machine.

    think ──(read tools)──> think            read-only tools run immediately
      │ (write tools proposed)
      v
    review ── interrupt: a person approves or rejects each proposed change
      v
    act ── runs approved changes (idempotency key per call) ──> think
    think ──(no tools)──> done               the model's summary is the result
    think ──(step limit)──> escalate ── interrupt: a person resumes or stops the run

A conversation is the same machine with `listen` in place of `start`: each message a person
sends enters at `listen`, joins the conversation's history and runs the loop above until the
agent answers (or proposes a change, which the person approves or declines in the conversation).
The model sees the last `history_turns` exchanges, so a long conversation's cost stays bounded.
While a thread executes it reports what it does through `emit` (store.py): the answer's text as
it is written, each tool step, proposals, the answer. Before each step the executing replica
confirms it still holds the thread's lock (locks.py).

Runs are durable: with a Postgres checkpointer a run survives restarts and can wait days for a
review. Every model turn, tool call and decision is in the run's trace (for evals and the API).

The audit trail (audit.py) records what people decide and what the agent changes: the run's
start (who asked), each review (who approved or rejected which changes), each approved change
the agent makes (on whose behalf), and stops and resumes. Reads are audited by the MCP servers
that serve them. Model turns and tool calls are OpenTelemetry spans (GenAI conventions), with
token counts and tool names but never prompt or result text.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.resources import files
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry import trace as otel

from . import locks
from .audit import Actor, Auditor, MemorySink
from .model import Model
from .tools import Tool, ToolError, ToolHost
from .variants import DEFAULT as DEFAULT_VARIANT
from .variants import Variant
from .variants import load as load_variants
from .variants import pick as pick_variant

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
    kind: str  # run | conversation
    message: str  # a conversation: the person's latest message
    turn: int  # a conversation: how many messages the person has sent
    variant: str  # the prompt and model variant, picked when the run starts (variants.py)


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


def system_prompt(product: str, name: str = "system") -> str:
    return files("agent").joinpath(f"prompts/{name}.md").read_text().format(product=product)


Emit = Callable[[str, str, dict[str, Any]], Any]


def window(messages: list[dict[str, Any]], turns: int) -> list[dict[str, Any]]:
    """The last `turns` exchanges of a conversation: from the person's message that began the
    oldest of them (a turn always starts with their text, so tool calls are never split)."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)]
    return messages[starts[-turns] :] if len(starts) > turns else messages


class TextBatcher:
    """Text as the model writes it, in a few lines a second rather than one per token."""

    def __init__(self, emit: Emit, thread: str, every_s: float = 0.15, size: int = 200) -> None:
        self.emit, self.thread, self.every_s, self.size = emit, thread, every_s, size
        self.buf: list[str] = []
        self.at = 0.0

    def __call__(self, chunk: str) -> None:
        import time

        self.buf.append(chunk)
        now = time.monotonic()
        if sum(map(len, self.buf)) >= self.size or now - self.at >= self.every_s:
            self.flush()
            self.at = now

    def flush(self) -> None:
        if self.buf:
            self.emit(self.thread, "text", {"delta": "".join(self.buf)})
            self.buf = []


tracer = otel.get_tracer("agent")


def build_graph(
    *,
    model: Model,
    tools: ToolHost,
    product: str,
    checkpointer: Any = None,
    auditor: Auditor | None = None,
    agent: str = "agent",
    emit: Emit | None = None,
    mode: Literal["run", "conversation"] = "run",
    history_turns: int = 20,
    variants: dict[str, Variant] | None = None,
    variant: str | None = None,
) -> Any:
    """`variants`: the reviewed prompt and model variants (variants.toml); `variant` forces one for
    every run (the evals), else Flipt picks per run."""
    audit = auditor or Auditor(MemorySink(), service=agent)
    me = Actor(id=agent, type="agent")
    known_variants = variants if variants is not None else load_variants()
    if variant is not None and variant not in known_variants:
        raise ValueError(f"no variant {variant!r} in variants.toml")

    def choose(state: RunState) -> str:
        if variant is not None:
            return variant
        task = state.get("task") or {}
        return pick_variant(
            agent, known_variants, user=requester(state) or str(task.get("event_id") or state["run_id"])
        )

    def say(thread: str, kind: str, **data: Any) -> None:
        if emit is not None:
            emit(thread, kind, data)

    def resource(state: RunState) -> tuple[str, list[str]]:
        return ("conversation" if mode == "conversation" else "run", [state["run_id"]])

    def catalog() -> dict[str, Tool]:
        return {t.name: t for t in tools.tools()}

    def call_tool(tool: Tool, args: dict[str, Any], key: str) -> str:
        with tracer.start_as_current_span(f"execute_tool {tool.name}", kind=otel.SpanKind.INTERNAL) as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("gen_ai.tool.call.id", key.rsplit(":", 1)[-1])
            try:
                return tools.call(tool, args, key)
            except ToolError as exc:
                span.set_status(otel.Status(otel.StatusCode.ERROR, type(exc).__name__))
                raise

    def respond(**kw: Any) -> dict[str, Any]:
        name = str(kw.get("model") or getattr(model, "model", "") or "")
        with tracer.start_as_current_span(f"chat {name}".strip(), kind=otel.SpanKind.CLIENT) as span:
            span.set_attribute("valxel.variant", kw["labels"].get("variant", "default"))
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.provider.name", "anthropic")
            span.set_attribute("gen_ai.conversation.id", kw["labels"]["run"])
            if name:
                span.set_attribute("gen_ai.request.model", name)
            if emit is None:
                reply = model.respond(**kw)
            else:
                text = TextBatcher(emit, kw["labels"]["run"])
                try:
                    reply = model.respond(**kw, on_text=text)
                finally:
                    text.flush()
            usage = reply.get("usage") or {}
            for attr, k in (
                ("gen_ai.usage.input_tokens", "input_tokens"),
                ("gen_ai.usage.output_tokens", "output_tokens"),
            ):
                if usage.get(k) is not None:
                    span.set_attribute(attr, int(usage[k]))
            if reply.get("stop_reason"):
                span.set_attribute("gen_ai.response.finish_reasons", [str(reply["stop_reason"])])
            if reply.get("model"):
                span.set_attribute("gen_ai.response.model", str(reply["model"]))
            return reply

    def requester(state: RunState) -> str:
        return str((state.get("task") or {}).get("requested_by") or "")

    def run_reads(
        run_id: str, calls: list[dict[str, Any]], known: dict[str, Tool]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for c in calls:
            try:
                out = call_tool(known[c["name"]], c.get("input") or {}, f"{run_id}:{c['id']}")
                results.append(_tool_result(c["id"], out))
                trace.append(_event("tool_call", tool=c["name"], input=c.get("input"), ok=True))
                say(run_id, "step", tool=c["name"], title=step_title(known[c["name"]]), ok=True)
            except ToolError as exc:
                results.append(_tool_result(c["id"], str(exc), error=True))
                trace.append(
                    _event("tool_call", tool=c["name"], input=c.get("input"), ok=False, error=str(exc))
                )
                say(run_id, "step", tool=c["name"], title=step_title(known[c["name"]]), ok=False)
        return results, trace

    def start(state: RunState) -> dict[str, Any]:
        task = state["task"]
        text = task.get("text") or str(task)
        who = requester(state)
        audit.record(
            "agent.run.start",
            actor=Actor(id=who, type="user")
            if who
            else Actor(id=str(task.get("event_id") or "event"), type="service"),
            resource=resource(state),
            details={"task_type": str(task.get("type", ""))},
        )
        say(state["run_id"], "turn", turn=1)
        chosen = choose(state)
        return {
            "messages": [{"role": "user", "content": text}],
            "steps": 0,
            "status": "running",
            "kind": "run",
            "variant": chosen,
            "trace": [_event("started", task_type=task.get("type", ""), variant=chosen)],
        }

    def listen(state: RunState) -> dict[str, Any]:
        """A person's message in a conversation: the next turn begins."""
        turn = state.get("turn", 0) + 1
        audit.record(
            "agent.chat.message",
            actor=Actor(id=requester(state), type="user"),
            resource=resource(state),
            details={"turn": turn, "chars": len(state.get("message") or "")},
        )
        say(state["run_id"], "turn", turn=turn, text=state.get("message") or "")
        chosen = state.get("variant") or choose(state)  # a conversation keeps its first variant
        return {
            "variant": chosen,
            "messages": [{"role": "user", "content": state.get("message") or ""}],
            "steps": 0,
            "turn": turn,
            "status": "running",
            "result": "",
            "kind": "conversation",
            "trace": [_event("message", turn=turn)],
        }

    def think(state: RunState) -> Command[Literal["think", "review", "escalate", "__end__"]]:
        locks.check()
        steps = state.get("steps", 0)
        if steps >= state.get("max_steps", 8):
            return Command(goto="escalate")
        known = catalog()
        v = known_variants.get(state.get("variant") or DEFAULT_VARIANT) or known_variants[DEFAULT_VARIANT]
        reply = respond(
            system=system_prompt(product, v.prompt(mode)),
            messages=window(state["messages"], history_turns)
            if mode == "conversation"
            else state["messages"],
            tools=[t.definition() for t in known.values()],
            labels={"run": state["run_id"], "variant": v.name},
            **({"model": v.model} if v.model else {}),
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
            say(state["run_id"], "reply", text=text)
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
            say(
                state["run_id"],
                "approval",
                actions=[
                    {
                        "id": w["id"],
                        "tool": w["name"],
                        "input": w.get("input"),
                        "title": step_title(known[w["name"]]),
                    }
                    for w in writes
                ],
            )
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
        pending = state.get("pending", [])
        audit.record(
            f"agent.change.{'approve' if decision['decision'] == 'approve' else 'reject'}",
            actor=Actor(id=str(decision.get("reviewer") or ""), type="user"),
            resource=resource(state),
            on_behalf_of=requester(state),
            details={
                "approved": [w["name"] for w in pending if _approved(decision, w["id"])],
                "rejected": [w["name"] for w in pending if not _approved(decision, w["id"])],
            },
        )
        return Command(
            goto="act",
            update={
                "trace": [_event("review", **decision)],
                "status": "running",
                "pending": [
                    {
                        **w,
                        "_approved": _approved(decision, w["id"]),
                        "_note": decision.get("note") or "",
                        "_by": str(decision.get("reviewer") or ""),
                    }
                    for w in state.get("pending", [])
                ],
            },
        )

    def act(state: RunState) -> Command[Literal["think"]]:
        locks.check()
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
            outcome, reason = "success", ""
            try:
                out = call_tool(known[w["name"]], w.get("input") or {}, f"{state['run_id']}:{w['id']}")
                results.append(_tool_result(w["id"], out))
                trace.append(
                    _event("tool_call", tool=w["name"], input=w.get("input"), ok=True, approved=True)
                )
                say(state["run_id"], "step", tool=w["name"], title=step_title(known[w["name"]]), ok=True)
            except ToolError as exc:
                outcome, reason = "failure", "the tool refused the change"
                results.append(_tool_result(w["id"], str(exc), error=True))
                trace.append(
                    _event("tool_call", tool=w["name"], input=w.get("input"), ok=False, error=str(exc))
                )
                say(state["run_id"], "step", tool=w["name"], title=step_title(known[w["name"]]), ok=False)
            # The MCP server audits the records it changed; this ties the change to the run,
            # the person it was done for and the approval.
            audit.record(
                "agent.change.apply",
                actor=me,
                resource=resource(state),
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
                # The change ran with its approver's delegated token, so it was done for them.
                on_behalf_of=w.get("_by") or requester(state),
                details={"tool": w["name"], "tool_call_id": w["id"], "requested_by": requester(state)},
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
        audit.record(
            f"agent.run.{decision['decision']}",
            actor=Actor(id=str(decision.get("reviewer") or ""), type="user"),
            resource=resource(state),
            on_behalf_of=requester(state),
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
    entry = "start" if mode == "run" else "listen"
    g.add_node(entry, start if mode == "run" else listen)
    g.add_node("think", think)
    g.add_node("review", review)
    g.add_node("act", act)
    g.add_node("escalate", escalate)
    g.add_edge(START, entry)
    g.add_edge(entry, "think")
    return g.compile(checkpointer=checkpointer, name="agent" if mode == "run" else "chat")


def step_title(tool: Tool) -> str:
    """What a person sees for a step: the first sentence of the tool's description."""
    first = (tool.description or tool.remote_name).strip().split("\n", 1)[0]
    return first.split(". ", 1)[0].rstrip(".")[:120]


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
