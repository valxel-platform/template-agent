from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent import errors, telemetry


def _run(app: Any) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(errors.TraceResponseMiddleware(app)({"type": "http", "path": "/x"}, receive, send))
    return sent


async def _ok(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": b"ok"})


def test_every_response_names_its_trace() -> None:
    telemetry.setup("svc", exporter=InMemorySpanExporter())
    with trace.get_tracer("t").start_as_current_span("req") as span:
        sent = _run(_ok)
    ctx = span.get_span_context()
    headers = dict(sent[0]["headers"])
    assert headers[b"traceresponse"] == f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-01".encode()
    assert dict(_run(_ok)[0]["headers"]).get(b"traceresponse") is None  # no trace, no header


def test_an_error_report_keeps_type_stack_path_and_trace_only() -> None:
    event: dict[str, Any] = {
        "user": {"email": "jane@example.com"},
        "extra": {"form": "Jane"},
        "breadcrumbs": {"values": [{"message": "Jane"}]},
        "server_name": "pod-1",
        "message": "Saved Jane",
        "logentry": {"message": "Couldn't save %s", "params": ["Jane"], "formatted": "Couldn't save Jane"},
        "request": {
            "method": "POST",
            "url": "https://svc/runs?name=Jane#x",
            "headers": {"Authorization": "t"},
            "data": {"name": "Jane"},
            "cookies": "s=1",
            "query_string": "name=Jane",
        },
        "exception": {
            "values": [
                {
                    "type": "KeyError",
                    "value": "'Jane'",
                    "stacktrace": {"frames": [{"function": "run", "vars": {"name": "Jane"}}]},
                }
            ]
        },
    }
    telemetry.setup("svc", exporter=InMemorySpanExporter())
    with trace.get_tracer("t").start_as_current_span("req") as span:
        out = errors.scrub(event)
    assert "Jane" not in repr(out)
    assert out["request"] == {"method": "POST", "url": "https://svc/runs"}
    assert out["exception"]["values"][0] == {
        "type": "KeyError",
        "stacktrace": {"frames": [{"function": "run"}]},
    }
    assert out["logentry"] == {"message": "Couldn't save %s"}
    assert out["tags"]["trace_id"] == f"{span.get_span_context().trace_id:032x}"
    assert errors.scrub({"request": "odd"})["request"] == "odd"


def test_sentry_starts_only_with_a_dsn(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    import sentry_sdk

    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    assert errors.start_errors({}) is False and calls == []
    assert errors.start_errors(
        {"SENTRY_DSN": "https://k@o1.ingest.sentry.io/42", "ENVIRONMENT": "production"}
    )
    [kw] = calls
    assert kw["environment"] == "production" and kw["send_default_pii"] is False
    assert kw["include_local_variables"] is False and kw["before_send"] is errors.scrub  # cast keeps it
    assert kw["before_breadcrumb"]({"message": "x"}, None) is None
