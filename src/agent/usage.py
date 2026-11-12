"""Who calls this agent: which clients start runs, chat, decide and read, how often, how fast, and
how it went.

Every request to the API (not /healthz, /readyz or /metrics) is recorded three ways, none of them
naming a person:

- one log line on the `agent.requests` logger, JSON like every other log line (trace ids included):
      {"message": "POST /runs 202", "http.route": "/runs", "http.request.method": "POST",
       "agent.client.id": "care-web", "agent.client.app": "vrixel-bff", "http.response.status_code": 202,
       "event.duration_ms": 18.2, ...}
- attributes on the request's trace span: agent.client.id, agent.client.app;
- Prometheus metrics on /metrics, beside the agent's run metrics:
      agent_requests_total{route, method, client, app, status}
      agent_request_duration_seconds{route, method, client, app}   (a histogram)

Two names say which client called:

- `client` is the OAuth client the caller's token was issued to (azp or client_id), set when the
  token is verified: the web app's client behind the BFF, another agent, an event bridge. Never
  the person (`sub`); which person started or decided which run is the audit trail's job.
- `app` is the calling software, from its User-Agent (vrixel-bff/1.4 -> vrixel-bff).

`route` is the route's template (/runs/{run_id}), never the path with its ids. Label values are
bounded: at most 100 OAuth clients and 50 apps are named, later ones count as "other". A request
refused before its token is read counts as client "none".
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from opentelemetry import trace
from prometheus_client import CollectorRegistry, Counter, Histogram

log = logging.getLogger("agent.requests")
MAX_CLIENTS = 100
MAX_APPS = 50
SKIP = ("/healthz", "/readyz", "/metrics")
STATE_KEY = "agent_client_id"  # where the verified token's client goes (request.state)
_SAFE = re.compile(r"[^a-z0-9._-]+")


class _Bounded:
    """The first `limit` names seen keep their name; after that, "other"."""

    def __init__(self, limit: int) -> None:
        self.limit, self.seen, self.lock = limit, set[str](), threading.Lock()

    def __call__(self, name: str) -> str:
        with self.lock:
            if name in self.seen:
                return name
            if len(self.seen) < self.limit:
                self.seen.add(name)
                return name
        return "other"

    def reset(self) -> None:
        with self.lock:
            self.seen.clear()


_clients, _apps = _Bounded(MAX_CLIENTS), _Bounded(MAX_APPS)


def clean(name: str, limit: int = 40) -> str:
    return _SAFE.sub("-", (name or "").strip().lower()).strip("-.")[:limit] or "unknown"


def app_from_user_agent(user_agent: str) -> str:
    """The calling software: the first product in its User-Agent (vrixel-bff/1.4 -> vrixel-bff)."""
    first = (user_agent or "").strip().split(" ", 1)[0]
    return clean(first.split("/", 1)[0])


def client_of(claims: dict[str, Any]) -> str:
    return str(claims.get("azp") or claims.get("client_id") or "")


class Metrics:
    """The request metrics, on the agent's registry."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "agent_requests",
            "API requests, by route, client and status",
            ["route", "method", "client", "app", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "agent_request_duration_seconds",
            "How long API requests take",
            ["route", "method", "client", "app"],
            registry=registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
        )


def record(
    metrics: Metrics, *, route: str, method: str, client_id: str, user_agent: str, status: int, seconds: float
) -> None:
    """One request: the log line, the span's attributes, the metrics. Never raises."""
    try:
        client = _clients(clean(client_id, 80)) if client_id else "none"
        app = _apps(app_from_user_agent(user_agent))
        metrics.requests.labels(route, method, client, app, str(status)).inc()
        metrics.duration.labels(route, method, client, app).observe(seconds)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attributes({"agent.client.id": client, "agent.client.app": app})
        log.info(
            "%s %s %s",
            method,
            route,
            status,
            extra={
                "fields": {
                    "http.route": route,
                    "http.request.method": method,
                    "agent.client.id": client,
                    "agent.client.app": app,
                    "http.response.status_code": status,
                    "event.duration_ms": round(seconds * 1000, 1),
                }
            },
        )
    except Exception:
        logging.getLogger(__name__).debug("usage not recorded", exc_info=True)


class UsageMiddleware:
    """Times each request and records it once the response has started (usage.record)."""

    def __init__(self, app: Any, metrics: Metrics) -> None:
        self.app, self.metrics = app, metrics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path", "") in SKIP:
            await self.app(scope, receive, send)
            return
        started, status = time.monotonic(), 500

        async def send_status(message: dict[str, Any]) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            route_obj = scope.get("route")
            route = getattr(route_obj, "path", "") or "unmatched"
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            state = scope.get("state") or {}
            record(
                self.metrics,
                route=route,
                method=str(scope.get("method", "")),
                client_id=str(state.get(STATE_KEY, "") if isinstance(state, dict) else ""),
                user_agent=headers.get("user-agent", ""),
                status=status,
                seconds=time.monotonic() - started,
            )


def reset() -> None:
    """Tests: forget the names seen."""
    _clients.reset()
    _apps.reset()
