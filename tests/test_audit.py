from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from agent.audit import Actor, Auditor, AuditUnavailable, KafkaSink, LogSink, MemorySink, make_sink

NURSE = Actor(id="nurse-7", type="user", client="web")


class FakeProducer:
    """confluent_kafka.Producer as KafkaSink uses it."""

    def __init__(self, error: Any = None, stuck: bool = False, full: bool = False) -> None:
        self.error, self.stuck, self.full = error, stuck, full
        self.sent: list[dict[str, Any]] = []
        self.pending: list[Any] = []

    def produce(
        self, topic: str, *, key: bytes, value: bytes, headers: dict[str, str], on_delivery: Any
    ) -> None:
        if self.full:
            raise BufferError("queue full")
        self.sent.append({"topic": topic, "key": key, "value": json.loads(value), "headers": headers})
        self.pending.append(on_delivery)

    def _deliver(self) -> None:
        for cb in self.pending:
            cb(self.error, None)
        self.pending = []

    def poll(self, timeout: float) -> int:
        self._deliver()
        return 0

    def flush(self, timeout: float) -> int:
        if self.stuck:
            return len(self.pending)
        self._deliver()
        return 0


def auditor(sink: Any, required: bool = True) -> Auditor:
    fixed = datetime(2026, 11, 4, 6, 0, tzinfo=UTC)
    return Auditor(
        sink,
        service="svc",
        product="emr",
        environment="staging",
        data_class="phi",
        required=required,
        now=lambda: fixed,
    )


def test_a_record_names_who_did_what_to_which_records() -> None:
    sink = MemorySink()
    e = auditor(sink).record(
        "patient.read",
        actor=NURSE,
        resource=("patient", ["p-2", "p-1", "p-1", ""]),
        on_behalf_of="dr-1",
        details={"count": 2},
    )
    assert sink.events == [e]
    assert e["specversion"] == "1.0" and e["type"] == "vrixel.audit.v1" and e["source"] == "emr/svc"
    assert e["subject"] == "patient/p-1,p-2" and e["time"] == "2026-11-04T06:00:00+00:00"
    d = e["data"]
    assert d["actor"] == {"id": "nurse-7", "type": "user", "client": "web"} and d["on_behalf_of"] == "dr-1"
    assert d["resource"] == {"type": "patient", "ids": ["p-1", "p-2"]} and d["outcome"] == "success"
    assert d["data_class"] == "phi" and d["environment"] == "staging" and d["details"] == {"count": 2}
    assert auditor(MemorySink()).record("login", actor=NURSE)["subject"] == "login"


def test_kafka_records_are_keyed_by_subject_and_acknowledged() -> None:
    p = FakeProducer()
    e = auditor(KafkaSink("k:9092", "emr.audit.v1", producer=p)).record(
        "task.create", actor=NURSE, resource=("task", ["t-1"])
    )
    [sent] = p.sent
    assert sent["topic"] == "emr.audit.v1" and sent["key"] == b"task/t-1" and sent["value"] == e
    assert sent["headers"] == {"ce_type": "vrixel.audit.v1", "ce_id": e["id"]}


@pytest.mark.parametrize(
    "producer", [FakeProducer(error="broker down"), FakeProducer(stuck=True), FakeProducer(full=True)]
)
def test_a_record_kafka_doesnt_acknowledge_fails_the_action(producer: FakeProducer) -> None:
    with pytest.raises(AuditUnavailable):
        auditor(KafkaSink("k:9092", "t", producer=producer)).record("patient.read", actor=NURSE)


def test_best_effort_records_are_logged_when_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="audit")
    sink = MemorySink(fail=True)
    auditor(sink, required=False).record("patient.read", actor=NURSE, resource=("patient", ["p-1"]))
    assert "audit record dropped" in caplog.text and "patient/p-1" in caplog.text
    p = FakeProducer()
    auditor(KafkaSink("k:9092", "t", producer=p, wait=False), required=False).record("x", actor=NURSE)
    assert len(p.sent) == 1 and not p.pending  # delivered on poll


def test_an_unreachable_broker_fails_within_the_timeout() -> None:
    sink = KafkaSink("127.0.0.1:9", "t", timeout=1.0)
    started = time.monotonic()
    with pytest.raises(AuditUnavailable):
        auditor(sink).record("patient.read", actor=NURSE)
    assert time.monotonic() - started < 5
    sink.close()


def test_sinks_are_chosen_by_name(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="audit")
    assert isinstance(make_sink("memory"), MemorySink)
    log_sink = make_sink("log")
    assert isinstance(log_sink, LogSink)
    auditor(log_sink).record("patient.read", actor=NURSE, resource=("patient", ["p-1"]))
    assert json.loads(caplog.records[-1].message)["subject"] == "patient/p-1"
    with pytest.raises(RuntimeError, match="bootstrap"):
        make_sink("kafka")
    with pytest.raises(RuntimeError, match="Unknown"):
        make_sink("syslog")
    assert isinstance(make_sink("kafka", bootstrap="127.0.0.1:9", topic="t"), KafkaSink)
