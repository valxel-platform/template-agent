"""HTTP API: start runs, read them, decide reviews.

  POST /runs                   start a run (Idempotency-Key header: a retry returns the same run)
  POST /events                 start a run from an event source (the event id is the idempotency key)
  GET  /runs/{id}              status, result, what is waiting for review, and the trace
  POST /runs/{id}/decision     approve or reject proposed changes; resume or stop a stalled run
  /conversations ...           a person talking with the agent (chat.py)
  GET  /healthz /readyz /metrics

Callers present OAuth 2.0 bearer tokens from the product's IdP. Starting runs needs the run
scope; deciding needs the review scope, and the reviewer's identity is recorded in the trace.
A run is visible to whoever started it and to reviewers of the same tenant; anyone else gets 404.
Reviewers can't approve changes in runs they started themselves (allow_self_approval).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from langgraph.types import Command
from opentelemetry import context as otel_context
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest
from pydantic import BaseModel, Field

from . import chat, cohorts, delegation, errors, events, flags, locks, telemetry, usage
from .audit import Auditor, AuditUnavailable, make_sink
from .chat import CONVERSATION
from .graph import APPROVAL_DECISIONS, TAKEOVER_DECISIONS, build_graph
from .locks import Lease, LeaseLost, Locks, MemoryLocks, PostgresLocks
from .model import GatewayModel, Model
from .settings import Settings
from .store import (
    Conversations,
    MemoryConversations,
    MemoryProgress,
    PostgresConversations,
    PostgresProgress,
    Progress,
)
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


BEARER = "_bearer"  # the caller's raw token, carried beside its verified claims


def _scopes(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope") or claims.get("scp") or ""
    return set(raw.split()) if isinstance(raw, str) else set(raw)


def tenants(claims: dict[str, Any], claim: str) -> list[str]:
    """The caller's tenants from the token: a name, a list of names, or Keycloak's organization
    claim (a map of organization alias to details)."""
    raw = claims.get(claim)
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        return sorted(str(k) for k in raw)
    return sorted(str(v) for v in raw)


def owner_of(claims: dict[str, Any], claim: str) -> dict[str, Any]:
    """What a run keeps about who started it, to decide who may see and decide it."""
    return {"requested_by": str(claims.get("sub", "")), "tenants": tenants(claims, claim)}


def may_see(claims: dict[str, Any], task: dict[str, Any], *, claim: str, review_scope: str) -> bool:
    """The run's requester, or a reviewer who shares a tenant with it (or both have none)."""
    if claims.get("sub") and claims.get("sub") == task.get("requested_by"):
        return True
    if review_scope not in _scopes(claims):
        return False
    mine, theirs = set(tenants(claims, claim)), set(task.get("tenants") or [])
    return bool(mine & theirs) if (mine or theirs) else True


class Runner:
    """Runs threads (runs and conversations) in the background, one step at a time per thread
    across every replica: the pod that moves a thread on holds its lock (locks.py) and reports
    what happens as it happens (store.py), so any pod can answer for it."""

    def __init__(
        self,
        graph: Any,
        max_steps: int,
        workers: int = 8,
        name: str = "agent",
        *,
        chat: Any = None,
        locks: Locks | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.graph = graph
        self.chat = chat
        self.name = name
        self.max_steps = max_steps
        self.locks: Locks = locks or MemoryLocks()
        self.progress: Progress = progress or MemoryProgress()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self.lock = threading.Lock()
        self.registry = CollectorRegistry()
        events.register(self.registry)  # valxel_events_total: the product events goals count
        self.started = Counter("agent_runs_started", "Runs started", registry=self.registry)
        self.finished = Counter(
            "agent_runs_finished", "Runs that reached an end or a review", ["status"], registry=self.registry
        )
        self.decisions = Counter("agent_decisions", "Review decisions", ["decision"], registry=self.registry)
        self.errors = Counter("agent_run_errors", "Runs that raised", registry=self.registry)
        self.messages = Counter("agent_chat_messages", "Messages people sent", registry=self.registry)
        # thread id -> (token of the person behind its next steps, when it expires). Memory only,
        # on the pod that executes those steps (the one that received the request): never in the
        # checkpoint, so a stolen database holds no one's credentials.
        self.people: dict[str, tuple[str, float | None]] = {}

    # Bounds on what the runner keeps in memory per thread.
    MAX_PEOPLE = 10_000

    def _prune(self) -> None:
        """Drop expired tokens, and the oldest when there are too many."""
        now = time.time()
        for thread, (_, expires) in list(self.people.items()):
            if expires is not None and now >= expires:
                self.people.pop(thread, None)
        while len(self.people) > self.MAX_PEOPLE:
            self.people.pop(next(iter(self.people)))

    def person(self, thread: str) -> str | None:
        held = self.people.get(thread)
        if not held:
            return None
        token, expires = held
        if expires is not None and time.time() >= expires:
            self.people.pop(thread, None)
            return None
        return token

    def graph_for(self, thread: str) -> Any:
        return self.chat if thread.startswith(CONVERSATION) and self.chat is not None else self.graph

    @staticmethod
    def config(thread: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread}}

    def state(self, thread: str) -> Any:
        return self.graph_for(thread).get_state(self.config(thread))

    def exists(self, thread: str) -> bool:
        return bool(self.state(thread).values)

    def _in_context(self, ctx: Any, thread: str, payload: Any, lease: Lease) -> None:
        """Runs on a worker thread, inside the trace of the request that started or resumed it."""
        token = otel_context.attach(ctx)
        try:
            self._execute(thread, payload, lease)
        finally:
            otel_context.detach(token)

    def _problem(self, thread: str, text: str) -> None:
        try:
            self.progress.emit(thread, "problem", {"text": text})
        except Exception:
            log.exception("could not record the outcome of %s", thread)

    def _execute(self, thread: str, payload: Any, lease: Lease) -> None:
        tracer = trace.get_tracer("agent")
        try:
            with tracer.start_as_current_span(
                f"invoke_agent {self.name}", kind=trace.SpanKind.INTERNAL
            ) as span:
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.agent.name", self.name)
                span.set_attribute("gen_ai.conversation.id", thread)
                with delegation.acting_for(self.person(thread)), locks.holding(lease):
                    self.graph_for(thread).invoke(payload, self.config(thread))
            self._problem(thread, "")
            self.finished.labels(self.view(thread, working=False)["status"]).inc()
        except LeaseLost:
            # Another replica may be carrying the thread on; it reports from here.
            log.warning("%s: this replica lost the thread's lock and stopped", thread)
        except AuditUnavailable:
            self.errors.inc()
            self._problem(thread, "The audit trail is unavailable, so the last step was not done. Try again.")
            log.exception("%s stopped: audit trail unavailable", thread)
        except Exception:
            self.errors.inc()
            self._problem(thread, "The last step failed; see the service logs.")
            log.exception("%s failed", thread)
        finally:
            lease.release()
            try:
                self.progress.emit(thread, "idle", {"status": self.view(thread, working=False)["status"]})
            except Exception:
                log.exception("could not report %s idle", thread)

    def submit(
        self, thread: str, payload: Any, wait: bool = False, person: str | None = None, fresh: bool = False
    ) -> bool:
        """Move a thread on, on this pod. 409 when another request is moving it on (here or on
        another pod). `fresh`: only if it doesn't exist yet (a start); False when it does."""
        lease = self.locks.acquire(thread)
        if lease is None:
            raise HTTPException(409, "This is already working; try again when it pauses.")
        try:
            if fresh and self.exists(thread):
                lease.release()
                return False
            with self.lock:
                if person:
                    self.people.pop(thread, None)  # re-inserted last: the oldest entries go first
                    self.people[thread] = (person, delegation.unverified_expiry(person))
                self._prune()
            fut = self.pool.submit(self._in_context, otel_context.get_current(), thread, payload, lease)
        except BaseException:
            lease.release()
            raise
        if wait:
            fut.result()
        return True

    def start(self, run_id: str, task: dict[str, Any], wait: bool = False, person: str | None = None) -> bool:
        started = self.submit(
            run_id, {"run_id": run_id, "task": task, "max_steps": self.max_steps}, wait, person, fresh=True
        )
        if started:
            self.started.inc()
        return started

    def view(self, run_id: str, working: bool | None = None) -> dict[str, Any]:
        snap = self.state(run_id)
        if not snap.values:
            raise HTTPException(404, "No run with this id.")
        v = snap.values
        waiting = [i.value for i in getattr(snap, "interrupts", ()) or ()]
        if working is None:
            working = self.locks.busy(run_id)
        status = "working" if working else v.get("status", "running")
        if not working and not waiting and status in ("running", "awaiting_review"):
            status = "failed"  # the last attempt raised; see the logs and the trace
        if waiting and not working:
            status = "needs_attention" if waiting[0].get("type") == "takeover" else "awaiting_review"
        problem = self.progress.latest(run_id, ("problem",))
        return {
            "run_id": run_id,
            "status": status,
            "result": v.get("result", ""),
            "waiting_for": waiting[0] if waiting else None,
            "steps": v.get("steps", 0),
            "problem": (problem or {}).get("data", {}).get("text", ""),
            "trace": v.get("trace", []),
        }


def make_auditor(settings: Settings) -> Auditor:
    sink = make_sink(
        settings.audit_sink,
        bootstrap=settings.audit_kafka_bootstrap,
        topic=settings.audit_topic,
        config=dict(settings.audit_kafka_config),
        data_class=settings.data_class,
    )
    return Auditor(
        sink,
        service=settings.name,
        product=settings.product,
        environment=settings.environment,
        data_class=settings.data_class,
        required=settings.audit_required,
    )


class Database:
    """Where threads are kept and coordinated: Postgres in deployed environments (any number of
    replicas), memory otherwise (one process, lost on restart)."""

    def __init__(self, settings: Settings, checkpointer: Any = None) -> None:
        self.pool: Any = None
        if not settings.database_url:
            from langgraph.checkpoint.memory import InMemorySaver

            if checkpointer is None:
                log.warning("AGENT_DATABASE_URL is not set: runs are kept in memory and lost on restart")
            self.checkpointer = checkpointer or InMemorySaver()
            self.locks: Locks = MemoryLocks()
            self.progress: Progress = MemoryProgress()
            self.conversations: Conversations = MemoryConversations()
            return
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(
            settings.database_url,
            max_size=settings.database_pool,
            open=True,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        saver = PostgresSaver(self.pool)
        saver.setup()
        self.checkpointer = checkpointer or saver
        self.locks = PostgresLocks(settings.database_url, self.pool, settings.name)
        self.progress = PostgresProgress(self.pool, settings.database_url)
        self.conversations = PostgresConversations(self.pool)


def create_app(
    settings: Settings | None = None,
    *,
    model: Model | None = None,
    tools: ToolHost | None = None,
    verifier: Verifier | None = None,
    checkpointer: Any = None,
    synchronous: bool = False,
    auditor: Auditor | None = None,
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
        timeout_s=settings.model_timeout_s,
    )
    if tools is None and not settings.allowed_tools:
        log.warning("AGENT_ALLOWED_TOOLS is empty: this agent can use no tools")
    tools = tools or McpToolHost.from_settings(settings)
    verify = verifier or jwt_verifier(settings)
    db = Database(settings, checkpointer)
    auditor = auditor or make_auditor(settings)
    common: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "product": settings.name,
        "checkpointer": db.checkpointer,
        "auditor": auditor,
        "agent": settings.name,
        "emit": db.progress.emit,
    }
    graph = build_graph(**common)
    conversation = build_graph(**common, mode="conversation", history_turns=settings.chat_history_turns)
    runner = Runner(
        graph, settings.max_steps, name=settings.name, chat=conversation, locks=db.locks, progress=db.progress
    )
    chat.start_pruning(db.progress, settings.progress_retention_s)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.pool.shutdown(wait=False)
        if db.pool is not None:
            db.pool.close()

    app = FastAPI(title=settings.name, version="0.1.0", lifespan=lifespan)
    # Each response names its trace (traceresponse); errors go to Sentry when SENTRY_DSN is set.
    app.add_middleware(errors.TraceResponseMiddleware)
    # Requests per flag, on and off, for the SDLC tool's gradual releases (logged once a minute).
    app.add_middleware(cohorts.CohortMiddleware)
    # Which clients call which routes, how fast, how it went (usage.py), on /metrics with the runs.
    app.add_middleware(usage.UsageMiddleware, metrics=usage.Metrics(runner.registry))
    cohorts.start_reporting()
    events.start_reporting()
    errors.start_errors()
    telemetry.setup(settings.name, app=app)
    app.state.runner = runner

    def caller(scope: str) -> Callable[..., dict[str, Any]]:
        def dep(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
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
            # Which OAuth client called, for the usage records (usage.py); never the person.
            setattr(request.state, usage.STATE_KEY, usage.client_of(claims))
            if scope not in _scopes(claims):
                raise HTTPException(403, f"The token needs the {scope} scope.")
            # Kept for delegation (the run's next steps act for this person), never logged or stored.
            return {**claims, BEARER: authorization[7:]}

        return dep

    can_run = Depends(caller(settings.run_scope))
    can_review = Depends(caller(settings.review_scope))

    def start(key: str, task: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid.uuid5(NAMESPACE, key))
        if runner.exists(run_id):  # a retry: the same run, if this caller may see it
            return owned(run_id, claims)
        try:
            started = runner.start(run_id, task, wait=synchronous, person=claims.get(BEARER))
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            # The same start is being handled right now (here or on another replica).
            return {"run_id": run_id, "status": "working"}
        if not started:
            return owned(run_id, claims)
        return runner.view(run_id) if runner.exists(run_id) else {"run_id": run_id, "status": "working"}

    def owned(run_id: str, claims: dict[str, Any]) -> dict[str, Any]:
        """The run's view, if this caller may see it; 404 otherwise (never saying it exists)."""
        if not runner.exists(run_id):
            raise HTTPException(404, "No run with this id.")
        task = runner.graph.get_state(runner.config(run_id)).values.get("task") or {}
        if not may_see(claims, task, claim=settings.tenant_claim, review_scope=settings.review_scope):
            raise HTTPException(404, "No run with this id.")
        return runner.view(run_id)

    @app.post("/runs", status_code=202)
    def create_run(
        body: RunRequest, claims: dict[str, Any] = can_run, idempotency_key: str = Header(default="")
    ) -> dict[str, Any]:
        key = f"{claims.get('sub', '')}:{idempotency_key}" if idempotency_key else str(uuid.uuid4())
        return start(key, {**body.task, **owner_of(claims, settings.tenant_claim)}, claims)

    @app.post("/events", status_code=202)
    def event(body: EventRequest, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        return start(
            f"event:{body.type}:{body.id}",
            {
                "type": body.type,
                "text": body.text,
                **body.data,
                "event_id": body.id,
                **owner_of(claims, settings.tenant_claim),
            },
            claims,
        )

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, claims: dict[str, Any] = can_run) -> dict[str, Any]:
        return owned(run_id, claims)

    @app.post("/runs/{run_id}/decision")
    def decide(run_id: str, body: Decision, claims: dict[str, Any] = can_review) -> dict[str, Any]:
        current = owned(run_id, claims)
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
        task = runner.graph.get_state(runner.config(run_id)).values.get("task") or {}
        requester = task.get("requested_by")
        if (
            waiting["type"] == "approval"
            and body.decision == "approve"
            and not settings.allow_self_approval
            and requester
            and requester == claims.get("sub")
        ):
            raise HTTPException(
                403, "Someone other than the person who asked for this change must approve it."
            )
        runner.decisions.labels(body.decision).inc()
        resume = {**body.model_dump(), "reviewer": claims.get("sub", "")}
        # An approved change runs for its approver: their token, exchanged per MCP server.
        runner.submit(run_id, Command(resume=resume), wait=synchronous, person=claims.get(BEARER))
        return runner.view(run_id)

    chat.routes(app, runner, db.conversations, tools, settings, can_run, synchronous)

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

    settings = Settings()
    telemetry.setup_logging(settings.name)
    uvicorn.run(create_app(settings), host="0.0.0.0", port=8080, log_config=None)  # noqa: S104  (inside the container)
