from __future__ import annotations

import json
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent import telemetry
from agent.audit import Actor, Auditor, MemorySink


def test_logs_are_json_with_the_trace_they_belong_to() -> None:
    exporter = InMemorySpanExporter()
    telemetry.setup("svc", exporter=exporter)
    fmt = telemetry.JsonFormatter("svc")
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("work") as span:
        line = json.loads(
            fmt.format(logging.LogRecord("app", logging.INFO, __file__, 1, "hello %s", ("x",), None))
        )
        audited = Auditor(MemorySink(), service="svc").record("x", actor=Actor(id="u"))
    ctx = span.get_span_context()
    assert line["message"] == "hello x" and line["service.name"] == "svc" and line["log.level"] == "INFO"
    assert line["trace.id"] == f"{ctx.trace_id:032x}" and line["span.id"] == f"{ctx.span_id:016x}"
    assert audited["data"]["trace_id"] == line["trace.id"]  # audit records point at the trace too
    assert [s.name for s in exporter.get_finished_spans()][-1] == "work"
    outside = json.loads(fmt.format(logging.LogRecord("app", logging.INFO, __file__, 1, "m", (), None)))
    assert "trace.id" not in outside


def test_errors_are_logged_with_their_type() -> None:
    fmt = telemetry.JsonFormatter("svc")
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord("app", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    line = json.loads(fmt.format(rec))
    assert line["error.type"] == "ValueError" and "boom" in line["error.stack_trace"]


def test_logging_goes_to_stdout_as_json(capsys: object) -> None:
    root = logging.getLogger()
    before = root.handlers[:]
    try:
        telemetry.setup_logging("svc")
        logging.getLogger("app").info("ready")
        out = capsys.readouterr().out.strip().splitlines()[-1]  # type: ignore[attr-defined]
        assert json.loads(out)["message"] == "ready"
    finally:
        root.handlers[:] = before
