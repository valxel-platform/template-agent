"""Conversations: a person talks with the agent, which looks things up for them and proposes
changes they approve in the conversation.

  POST /conversations                     start one with a first message
  GET  /conversations                     the caller's conversations, latest first
  GET  /conversations/{id}                its messages, status and what waits for the person
  POST /conversations/{id}/messages       send the next message
  GET  /conversations/{id}/stream         what happens as it happens (Server-Sent Events)
  POST /conversations/{id}/decision       approve or decline proposed changes; continue or stop

Sending a message answers 202 at once with `after`, the position to stream from. The reply comes
on the stream from whichever pod executes it, through Postgres: the stream can be served by any
replica and survives a reconnect (send Last-Event-ID, or ?after=).

  event: turn      {"turn": 3, "text": "..."}         the person's message was taken
  event: text      {"delta": "..."}                   the reply, as it is written
  event: step      {"tool", "title", "ok"}            a tool step, e.g. "Find a patient"
  event: approval  {"actions": [...]}                 changes waiting for the person's decision
  event: reply     {"text": "..."}                    the agent's answer
  event: problem   {"text": "..."}                    the step failed (empty: it didn't)
  event: idle      {"status": "..."}                  the turn is over; the stream ends

A conversation belongs to the person who started it: nobody else sees it (404). One message is
answered at a time (409 while it works), and a proposal must be decided before the next message.
The person approves changes in their own conversation: they asked for them, and the changes run
with their own token. Work that needs someone else's approval goes through runs (POST /runs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from anyio import to_thread
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from .graph import APPROVAL_DECISIONS, TAKEOVER_DECISIONS, step_title
from .store import Conversations, Progress
from .tools import ToolHost

log = logging.getLogger("agent.chat")
CONVERSATION = "conv-"
PING_S = 15.0  # a comment line this often keeps proxies from closing a quiet stream
STREAM_MAX_S = 600.0  # then the client reconnects from where it was


class NewConversation(BaseModel):
    text: str = Field(min_length=1)
    title: str = Field(default="", max_length=120)


class Message(BaseModel):
    text: str = Field(min_length=1)


class ChatDecision(BaseModel):
    decision: str
    approved_ids: list[str] | None = None
    note: str = ""


def history(values: dict[str, Any], titles: dict[str, str]) -> list[dict[str, Any]]:
    """The conversation as the person sees it: their messages, and each answer with its steps."""
    out: list[dict[str, Any]] = []
    for m in values.get("messages") or []:
        content = m["content"]
        if m["role"] == "user" and isinstance(content, str):
            if not content.startswith("Note from the reviewer: "):
                out.append({"role": "user", "text": content})
            continue
        if m["role"] != "assistant":
            continue  # tool results
        if not out or out[-1]["role"] != "assistant":
            out.append({"role": "assistant", "text": "", "steps": []})
        for b in content if isinstance(content, list) else [{"type": "text", "text": content}]:
            if b.get("type") == "text" and b.get("text"):
                sep = "\n\n" if out[-1]["text"] else ""
                out[-1]["text"] += sep + b["text"]
            elif b.get("type") == "tool_use":
                out[-1]["steps"].append({"tool": b["name"], "title": titles.get(b["name"], b["name"])})
    return out


def start_pruning(progress: Progress, retention_s: float, every_s: float = 3600.0) -> None:
    """Drop old progress hourly (on every replica: the delete is the same wherever it runs)."""

    def loop() -> None:
        while True:
            time.sleep(every_s)
            try:
                progress.prune(retention_s)
            except Exception:
                log.exception("pruning progress failed")

    threading.Thread(target=loop, name="progress-prune", daemon=True).start()


def sse(seq: int | None, kind: str, data: dict[str, Any]) -> str:
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def routes(
    app: FastAPI,
    runner: Any,
    conversations: Conversations,
    tools: ToolHost,
    settings: Any,
    can_run: Any,
    synchronous: bool,
) -> None:
    from .api import BEARER, owner_of

    progress: Progress = runner.progress

    def titles() -> dict[str, str]:
        try:
            return {t.name: step_title(t) for t in tools.tools()}
        except Exception:  # noqa: BLE001 - steps show the tool's name instead
            return {}

    def mine(cid: str, claims: dict[str, Any]) -> dict[str, Any]:
        """The conversation's state, if it is the caller's; 404 otherwise."""
        if not cid.startswith(CONVERSATION):
            raise HTTPException(404, "No conversation with this id.")
        values = runner.state(cid).values
        task = (values or {}).get("task") or {}
        if not values or not claims.get("sub") or task.get("requested_by") != claims.get("sub"):
            raise HTTPException(404, "No conversation with this id.")
        return dict(values)

    def view(cid: str, values: dict[str, Any]) -> dict[str, Any]:
        v = runner.view(cid)
        status = "idle" if v["status"] == "done" else v["status"]
        return {
            "conversation_id": cid,
            "status": status,
            "waiting_for": v["waiting_for"],
            "problem": v["problem"],
            "messages": history(values, titles()),
            "after": progress.last(cid),
        }

    def check_text(text: str) -> str:
        text = text.strip()
        if not text:
            raise HTTPException(422, "Write a message.")
        if len(text) > settings.chat_max_chars:
            raise HTTPException(422, f"A message can be at most {settings.chat_max_chars} characters.")
        return text

    def send(
        cid: str, payload: dict[str, Any], claims: dict[str, Any], fresh: bool = False
    ) -> dict[str, Any]:
        after = progress.last(cid)
        if not runner.submit(cid, payload, wait=synchronous, person=claims.get(BEARER), fresh=fresh):
            raise HTTPException(409, "This conversation already exists.")
        runner.messages.inc()
        conversations.touch(cid)
        return {"conversation_id": cid, "after": after, "status": "working"}

    @app.post("/conversations", status_code=202)
    def create(
        body: NewConversation, claims: dict[str, Any] = can_run, idempotency_key: str = Header(default="")
    ) -> dict[str, Any]:
        text = check_text(body.text)
        key = f"{claims.get('sub', '')}:{idempotency_key}" if idempotency_key else str(uuid.uuid4())
        cid = CONVERSATION + str(uuid.uuid5(uuid.NAMESPACE_URL, f"conversation:{key}"))
        if runner.exists(cid):  # a retry of the same start
            mine(cid, claims)
            return {"conversation_id": cid, "after": 0, "status": runner.view(cid)["status"]}
        owner = owner_of(claims, settings.tenant_claim)
        conversations.add(cid, owner["requested_by"], owner["tenants"], body.title or text[:60])
        payload = {"run_id": cid, "task": owner, "message": text, "max_steps": runner.max_steps}
        return send(cid, payload, claims, fresh=True)

    @app.get("/conversations")
    def listing(claims: dict[str, Any] = can_run) -> dict[str, Any]:
        rows = conversations.mine(str(claims.get("sub", "")))
        return {
            "conversations": [{k: r[k] for k in ("conversation_id", "title", "updated_at")} for r in rows]
        }

    @app.get("/conversations/{cid}")
    def get(cid: str, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        return view(cid, mine(cid, claims))

    @app.post("/conversations/{cid}/messages", status_code=202)
    def message(cid: str, body: Message, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        mine(cid, claims)
        text = check_text(body.text)
        if runner.state(cid).interrupts:
            raise HTTPException(409, "Decide on the proposed change first.")
        return send(cid, {"run_id": cid, "message": text, "max_steps": runner.max_steps}, claims)

    @app.post("/conversations/{cid}/decision", status_code=202)
    def decide(cid: str, body: ChatDecision, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        mine(cid, claims)
        waiting = [i.value for i in runner.state(cid).interrupts or ()]
        if not waiting:
            raise HTTPException(409, "Nothing in this conversation is waiting for a decision.")
        allowed = APPROVAL_DECISIONS if waiting[0]["type"] == "approval" else TAKEOVER_DECISIONS
        if body.decision not in allowed:
            raise HTTPException(422, f"Decide one of: {', '.join(allowed)}.")
        if body.approved_ids is not None:
            unknown = set(body.approved_ids) - {a["id"] for a in waiting[0].get("actions", [])}
            if unknown:
                raise HTTPException(422, f"Not proposed here: {', '.join(sorted(unknown))}.")
        runner.decisions.labels(body.decision).inc()
        resume = {**body.model_dump(), "reviewer": claims.get("sub", "")}
        # The changes run for the person who approved them: their token, exchanged per MCP server.
        after = progress.last(cid)
        runner.submit(cid, Command(resume=resume), wait=synchronous, person=claims.get(BEARER))
        return {"conversation_id": cid, "after": after, "status": "working"}

    @app.get("/conversations/{cid}/stream")
    async def stream(
        cid: str,
        claims: dict[str, Any] = can_run,
        after: int = Query(default=0, ge=0),
        last_event_id: str = Header(default=""),
    ) -> StreamingResponse:
        await to_thread.run_sync(mine, cid, claims)
        start = max(after, int(last_event_id) if last_event_id.isdigit() else 0)
        return StreamingResponse(
            events(cid, start, runner.locks.busy, progress),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )


async def events(cid: str, after: int, busy: Callable[[str], bool], progress: Progress) -> AsyncIterator[str]:
    """Progress after `after`, until the turn is over (idle) or the stream has run long enough."""
    watch = progress.watch(cid)
    began, pinged = time.monotonic(), time.monotonic()
    try:
        while time.monotonic() - began < STREAM_MAX_S:
            items = await to_thread.run_sync(progress.read, cid, after)
            for e in items:
                after = e["seq"]
                yield sse(e["seq"], e["kind"], e["data"])
                if e["kind"] == "idle":
                    return
            if not items:
                if not await to_thread.run_sync(busy, cid):
                    # Nothing is happening: read once more (the turn may have just ended) and stop.
                    for e in await to_thread.run_sync(progress.read, cid, after):
                        yield sse(e["seq"], e["kind"], e["data"])
                    yield sse(None, "idle", {"status": "idle"})
                    return
                if time.monotonic() - pinged >= PING_S:
                    pinged = time.monotonic()
                    yield ": ping\n\n"
                await watch.wait(2.0)
    except asyncio.CancelledError:  # the person went away
        raise
    finally:
        progress.unwatch(watch)
