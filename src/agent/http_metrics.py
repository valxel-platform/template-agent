"""HTTP request metrics, as the Java templates record them (Spring's http.server.requests): one
histogram, `http_server_requests_seconds`, by method, route, status and outcome, served on the
management port with the rest of /metrics. The data plane's metrics collector scrapes it on every
cloud, so requests, errors and latency per route need nothing from the network (platform-foundation
docs/product-telemetry.md).

`uri` is the route as declared (`/orders/{order_id}`), never the path as requested, so ids don't
become label values. A request no route matched is `NOT_FOUND`, `REDIRECTION` or `UNKNOWN`, as in
Spring. Kubernetes' probes are left out: they would be most of the traffic and say nothing about it.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Histogram

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]

NAME = "http_server_requests_seconds"
# The same boundaries the Java templates publish (management.metrics.distribution.slo), so one
# query reads both.
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
PROBES = frozenset({"/healthz", "/readyz", "/livez"})


def outcome(status: int) -> str:
    if status < 200:
        return "INFORMATIONAL"
    if status < 300:
        return "SUCCESS"
    if status < 400:
        return "REDIRECTION"
    if status < 500:
        return "CLIENT_ERROR"
    return "SERVER_ERROR"


def route(scope: Scope, status: int, asked: Scope) -> str:
    """The matched route's template. Starlette (and so FastAPI) records the innermost route on the
    scope as it routes; a mounted app's prefix is what routing added to root_path."""
    inner = getattr(scope.get("route"), "path", None)
    if inner:
        return str(scope.get("root_path", ""))[len(str(asked.get("root_path", ""))) :] + str(inner)
    if status == 404:
        return "NOT_FOUND"
    if 300 <= status < 400:
        return "REDIRECTION"
    return "UNKNOWN"


class HttpMetrics:
    """ASGI middleware: `app = HttpMetrics(app)`, or `app.add_middleware(HttpMetrics)` in FastAPI."""

    def __init__(
        self, app: App, registry: CollectorRegistry | None = None, skip: Iterable[str] = PROBES
    ) -> None:
        self.app = app
        self.skip = frozenset(skip)
        registry = REGISTRY if registry is None else registry
        existing = getattr(registry, "_names_to_collectors", {}).get(NAME)
        self.histogram = (
            existing
            if existing is not None
            else Histogram(
                NAME.removesuffix("_seconds"),
                "HTTP requests served, by route and status",
                unit="seconds",
                labelnames=("method", "uri", "status", "outcome"),
                buckets=BUCKETS,
                registry=registry,
            )
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.skip:
            await self.app(scope, receive, send)
            return
        status = 500  # what the client gets if the app raises before it responds
        asked = dict(scope)  # routing into a mounted app rewrites the scope's path
        started = time.perf_counter()

        async def sending(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, sending)
        finally:
            self.histogram.labels(
                scope.get("method", ""),
                route(scope, status, asked),
                str(status),
                outcome(status),
            ).observe(time.perf_counter() - started)
