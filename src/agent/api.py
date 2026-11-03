"""HTTP API: start runs, read them, decide reviews.

  POST /runs                   start a run (Idempotency-Key header: a retry returns the same run)
  POST /events                 start a run from an event source (the event id is the idempotency key)
  GET  /runs/{id}              status, result, what is waiting for review, and the trace
  POST /runs/{id}/decision     approve or reject proposed changes; resume or stop a stalled run
  GET  /healthz /readyz /metrics

Callers present OAuth 2.0 bearer tokens from the product's IdP. Starting runs needs the run
scope; deciding needs the review scope, and the reviewer's identity is recorded in the trace.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest
from pydantic import BaseModel, Field

from . import flags
from .graph import APPROVAL_DECISIONS, TAKEOVER_DECISIONS, build_graph
from .model import GatewayModel, Model
from .settings import Settings
from .tools import McpToolHost, ToolHost

log = logging.getLogger("agent")
NAMESPACE = uuid.UUID("6f1c2a52-9b7e-4d3f-a0d4-1c5e8b7f2a10")

Verifier = Callable[[str], dict[str, Any]]


class RunRequest(BaseModel):
    task: dict[str, Any] = Field(description="What to work on, e.g. {'type': 'referral', 'text': '...'}")


class EventRequest(BaseModel):
    id: str
    type: str
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    decision: str
    approved_ids: list[str] | None = None  # approve only some of the proposed changes
    note: str = ""


def jwt_verifier(settings: Settings) -> Verifier:
    import jwt

    if not (settings.oidc_issuer and settings.oidc_audience and settings.oidc_jwks_url):
        raise RuntimeError("Set AGENT_OIDC_ISSUER, AGENT_OIDC_AUDIENCE and AGENT_OIDC_JWKS_URL")
    jwks = jwt.PyJWKClient(settings.oidc_jwks_url, cache_keys=True)

    def verify(token: str) -> dict[str, Any]:
        key = jwks.get_signing_key_from_jwt(token)
        claims: dict[str, Any] = jwt.decode(
            token,
            key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
        )
        return claims

    return verify


def _scopes(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope") or claims.get("scp") or ""
    return set(raw.split()) if isinstance(raw, str) else set(raw)


class Runner:
    """Runs the graph in the background; one run executes at a time per id."""

    def __init__(self, graph: Any, max_steps: int, workers: int = 8) -> None:
        self.graph = graph
        self.max_steps = max_steps
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self.active: set[str] = set()
        self.lock = threading.Lock()
        self.registry = CollectorRegistry()
        self.started = Counter("agent_runs_started", "Runs started", registry=self.registry)
        self.finished = Counter(
            "agent_runs_finished", "Runs that reached an end or a review", ["status"], registry=self.registry
        )
        self.decisions = Counter("agent_decisions", "Review decisions", ["decision"], registry=self.registry)
        self.errors = Counter("agent_run_errors", "Runs that raised", registry=self.registry)

    @staticmethod
    def config(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def exists(self, run_id: str) -> bool:
        return bool(self.graph.get_state(self.config(run_id)).values)

    def _execute(self, run_id: str, payload: Any) -> None:
        try:
            self.graph.invoke(payload, self.config(run_id))
            self.finished.labels(self.view(run_id)["status"]).inc()
        except Exception:
            self.errors.inc()
            log.exception("run %s failed", run_id)
        finally:
            with self.lock:
                self.active.discard(run_id)

    def submit(self, run_id: str, payload: Any, wait: bool = False) -> None:
        with self.lock:
            if run_id in self.active:
                raise HTTPException(409, "This run is already working; try again when it pauses.")
            self.active.add(run_id)
        fut = self.pool.submit(self._execute, run_id, payload)
        if wait:
            fut.result()

    def start(self, run_id: str, task: dict[str, Any], wait: bool = False) -> None:
        self.started.inc()
        self.submit(run_id, {"run_id": run_id, "task": task, "max_steps": self.max_steps}, wait)

    def view(self, run_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self.config(run_id))
        if not snap.values:
            raise HTTPException(404, "No run with this id.")
        v = snap.values
        waiting = [i.value for i in getattr(snap, "interrupts", ()) or ()]
        with self.lock:
            working = run_id in self.active
        status = "working" if working else v.get("status", "running")
        if not working and not waiting and status in ("running", "awaiting_review"):
            status = "failed"  # the last attempt raised; see the logs and the trace
        if waiting and not working:
            status = "needs_attention" if waiting[0].get("type") == "takeover" else "awaiting_review"
        return {
            "run_id": run_id,
            "status": status,
            "result": v.get("result", ""),
            "waiting_for": waiting[0] if waiting else None,
            "steps": v.get("steps", 0),
            "trace": v.get("trace", []),
        }


def make_checkpointer(settings: Settings) -> Any:
    if not settings.database_url:
        from langgraph.checkpoint.memory import InMemorySaver

        log.warning("AGENT_DATABASE_URL is not set: runs are kept in memory and lost on restart")
        return InMemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        settings.database_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool)  # type: ignore[arg-type]
    saver.setup()
    return saver


def create_app(
    settings: Settings | None = None,
    *,
    model: Model | None = None,
    tools: ToolHost | None = None,
    verifier: Verifier | None = None,
    checkpointer: Any = None,
    synchronous: bool = False,
) -> FastAPI:
    """`synchronous` makes requests wait for the run to pause or finish (tests and evals)."""
    settings = settings or Settings()
    flags.configure(settings)
    model = model or GatewayModel(
        base_url=settings.gateway_url,
        api_key=settings.gateway_key,
        model=settings.model,
        max_tokens=settings.max_tokens,
        agent=settings.name,
    )
    tools = tools or McpToolHost.from_settings(settings)
    verify = verifier or jwt_verifier(settings)
    graph = build_graph(
        model=model,
        tools=tools,
        product=settings.name,
        checkpointer=checkpointer or make_checkpointer(settings),
    )
    runner = Runner(graph, settings.max_steps)
    app = FastAPI(title=settings.name, version="0.1.0")
    app.state.runner = runner

    def caller(scope: str) -> Callable[..., dict[str, Any]]:
        def dep(authorization: str = Header(default="")) -> dict[str, Any]:
            if not authorization.lower().startswith("bearer "):
                raise HTTPException(
                    401, "Send an OAuth bearer token.", headers={"WWW-Authenticate": "Bearer"}
                )
            try:
                claims = verify(authorization[7:])
            except Exception as exc:
                raise HTTPException(
                    401, "The token is not valid.", headers={"WWW-Authenticate": "Bearer"}
                ) from exc
            if scope not in _scopes(claims):
                raise HTTPException(403, f"The token needs the {scope} scope.")
            return claims

        return dep

    can_run = Depends(caller(settings.run_scope))
    can_review = Depends(caller(settings.review_scope))

    def start(key: str, task: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid.uuid5(NAMESPACE, key))
        if not runner.exists(run_id):
            runner.start(run_id, task, wait=synchronous)
        return runner.view(run_id) if runner.exists(run_id) else {"run_id": run_id, "status": "working"}

    @app.post("/runs", status_code=202)
    def create_run(
        body: RunRequest, claims: dict[str, Any] = can_run, idempotency_key: str = Header(default="")
    ) -> dict[str, Any]:
        key = f"{claims.get('sub', '')}:{idempotency_key}" if idempotency_key else str(uuid.uuid4())
        return start(key, {**body.task, "requested_by": claims.get("sub", "")})

    @app.post("/events", status_code=202)
    def event(body: EventRequest, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        return start(
            f"event:{body.type}:{body.id}",
            {"type": body.type, "text": body.text, **body.data, "event_id": body.id},
        )

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        return runner.view(run_id)

    @app.post("/runs/{run_id}/decision")
    def decide(run_id: str, body: Decision, claims: dict[str, Any] = can_review) -> dict[str, Any]:
        current = runner.view(run_id)
        waiting = current["waiting_for"]
        if not waiting or current["status"] == "working":
            raise HTTPException(409, "This run is not waiting for a decision.")
        allowed = APPROVAL_DECISIONS if waiting["type"] == "approval" else TAKEOVER_DECISIONS
        if body.decision not in allowed:
            raise HTTPException(422, f"Decide one of: {', '.join(allowed)}.")
        if body.approved_ids is not None:
            unknown = set(body.approved_ids) - {a["id"] for a in waiting.get("actions", [])}
            if unknown:
                raise HTTPException(422, f"Not proposed in this run: {', '.join(sorted(unknown))}.")
        runner.decisions.labels(body.decision).inc()
        resume = {**body.model_dump(), "reviewer": claims.get("sub", "")}
        runner.submit(run_id, Command(resume=resume), wait=synchronous)
        return runner.view(run_id)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        try:
            n = len(tools.tools())
        except Exception as exc:
            raise HTTPException(503, f"MCP servers unreachable: {type(exc).__name__}") from exc
        return {"status": "ready", "tools": n}

    @app.get("/metrics")
    def metrics(request: Request) -> PlainTextResponse:
        return PlainTextResponse(generate_latest(runner.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)  # noqa: S104  (inside the container)
