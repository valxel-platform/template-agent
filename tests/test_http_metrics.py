"""HTTP request metrics: one histogram by method, declared route, status and outcome, as Spring's."""

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from agent.http_metrics import BUCKETS, HttpMetrics, outcome, route


def _count(registry: CollectorRegistry, **labels: str) -> float:
    value = registry.get_sample_value("http_server_requests_seconds_count", labels)
    return value or 0.0


def _boom(_request):
    raise RuntimeError("boom")


def _app(registry: CollectorRegistry) -> HttpMetrics:
    inner = Starlette(
        routes=[
            Route("/orders/{order_id}", lambda r: PlainTextResponse(r.path_params["order_id"])),
            Route("/healthz", lambda r: PlainTextResponse("ok")),
            Route("/boom", _boom),
            Mount(
                "/api",
                routes=[
                    Route(
                        "/items/{item}", lambda r: PlainTextResponse("x", status_code=201), methods=["POST"]
                    )
                ],
            ),
        ]
    )
    return HttpMetrics(inner, registry=registry)


def test_requests_are_counted_by_declared_route_not_path() -> None:
    registry = CollectorRegistry()
    client = TestClient(_app(registry))
    for order in ("a1", "b2"):
        assert client.get(f"/orders/{order}").status_code == 200
    assert _count(registry, method="GET", uri="/orders/{order_id}", status="200", outcome="SUCCESS") == 2
    assert client.post("/api/items/7").status_code == 201
    assert _count(registry, method="POST", uri="/api/items/{item}", status="201", outcome="SUCCESS") == 1


def test_unmatched_errors_and_probes() -> None:
    registry = CollectorRegistry()
    client = TestClient(_app(registry), raise_server_exceptions=False)
    client.get("/nothing/here/123")
    assert _count(registry, method="GET", uri="NOT_FOUND", status="404", outcome="CLIENT_ERROR") == 1
    assert client.get("/boom").status_code == 500
    assert _count(registry, method="GET", uri="/boom", status="500", outcome="SERVER_ERROR") == 1
    client.get("/healthz")  # the probes would drown everything else out
    assert b'uri="/healthz"' not in generate_latest(registry)


def test_buckets_match_the_java_templates_and_wrapping_twice_shares_one_histogram() -> None:
    registry = CollectorRegistry()
    first, second = _app(registry), _app(registry)
    assert first.histogram is second.histogram
    TestClient(first).get("/orders/x")
    text = generate_latest(registry).decode()
    assert all(f'le="{b}"' in text for b in ("0.005", "0.25", "10.0"))
    assert len(BUCKETS) == 11


def test_fastapi_routes_come_from_the_scope() -> None:
    fastapi = pytest.importorskip("fastapi")
    registry = CollectorRegistry()
    app = fastapi.FastAPI()

    @app.get("/things/{thing_id}")
    def thing(thing_id: str) -> dict[str, str]:
        return {"id": thing_id}

    app.add_middleware(HttpMetrics, registry=registry)
    TestClient(app).get("/things/42")
    assert _count(registry, method="GET", uri="/things/{thing_id}", status="200", outcome="SUCCESS") == 1


def test_outcomes_and_unmatched_routes_follow_spring() -> None:
    assert [outcome(s) for s in (101, 204, 302, 404, 503)] == [
        "INFORMATIONAL",
        "SUCCESS",
        "REDIRECTION",
        "CLIENT_ERROR",
        "SERVER_ERROR",
    ]
    assert [route({}, s, {}) for s in (404, 301, 500)] == ["NOT_FOUND", "REDIRECTION", "UNKNOWN"]


def test_websockets_pass_through_uncounted() -> None:
    registry = CollectorRegistry()
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    import asyncio

    asyncio.run(HttpMetrics(inner, registry=registry)({"type": "websocket", "path": "/ws"}, None, None))
    assert (
        seen == ["websocket"]
        and _count(registry, method="", uri="UNKNOWN", status="500", outcome="SERVER_ERROR") == 0
    )
