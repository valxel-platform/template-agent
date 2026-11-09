"""The audit trail: who did what to which records, and how it turned out.

Every record is one event (CloudEvents 1.0, type `vrixel.audit.v1`) published to the product's
audit topic on Kafka. The platform's audit sink (vrixel-shared-services/ss-audit-sink) copies the
topic into an append-only, hash-chained Postgres table and archives it to object storage.

Records name people and records by id only. Never put field values in them (names, notes, dates
of birth): they may be PHI, and the audit trail is read by more people than the data is.

    auditor.record("patient.read", actor=Actor(id=sub, type="user"), resource=("patient", [pid]))

With `required` set (the default for PHI and payments products), a record that can't be
published raises `AuditUnavailable`; callers must not return data or report success then.
Sinks: `kafka` for deployed environments, `log` (one JSON line on the `audit` logger) for local
work, `memory` for tests.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

log = logging.getLogger("audit")

EVENT_TYPE = "vrixel.audit.v1"
Outcome = Literal["success", "denied", "failure"]
ActorType = Literal["user", "service", "agent"]


class AuditUnavailable(RuntimeError):
    """The audit record could not be published, so the action must not be reported as done."""


@dataclass(frozen=True)
class Actor:
    id: str
    type: ActorType = "user"
    client: str = ""  # the OAuth client the caller used, if any


class Sink(Protocol):
    def publish(self, key: str, event: dict[str, Any]) -> None: ...


class LogSink:
    """One JSON line per record on the `audit` logger (local development)."""

    def publish(self, key: str, event: dict[str, Any]) -> None:
        log.info(json.dumps(event, default=str, separators=(",", ":")))


@dataclass
class MemorySink:
    events: list[dict[str, Any]] = field(default_factory=list)
    fail: bool = False

    def publish(self, key: str, event: dict[str, Any]) -> None:
        if self.fail:
            raise AuditUnavailable("audit sink unavailable")
        self.events.append(event)


class KafkaSink:
    """Publishes to Kafka with an idempotent producer (acks=all). `wait` blocks until the broker
    has the record, so a failure is reported to the caller instead of lost."""

    def __init__(
        self,
        bootstrap: str,
        topic: str,
        *,
        config: dict[str, Any] | None = None,
        wait: bool = True,
        timeout: float = 5.0,
        producer: Any = None,
    ) -> None:
        if producer is None:
            from confluent_kafka import Producer

            producer = Producer(
                {
                    "bootstrap.servers": bootstrap,
                    "enable.idempotence": True,
                    "acks": "all",
                    "compression.type": "zstd",
                    "message.timeout.ms": int(timeout * 1000),
                    "client.id": "audit",
                    **(config or {}),
                }
            )
        self.producer, self.topic, self.wait, self.timeout = producer, topic, wait, timeout
        self._lock = threading.Lock()

    def publish(self, key: str, event: dict[str, Any]) -> None:
        errors: list[Any] = []

        def delivered(err: Any, msg: Any) -> None:
            if err is not None:
                errors.append(err)

        value = json.dumps(event, default=str, separators=(",", ":")).encode()
        try:
            self.producer.produce(
                self.topic,
                key=key.encode(),
                value=value,
                headers={"ce_type": EVENT_TYPE, "ce_id": event["id"]},
                on_delivery=delivered,
            )
        except BufferError as exc:
            raise AuditUnavailable("audit producer queue is full") from exc
        if not self.wait:
            self.producer.poll(0)
            return
        with self._lock:
            left = self.producer.flush(self.timeout)
        if errors or left:
            raise AuditUnavailable(
                f"audit record not acknowledged by Kafka: {errors[0] if errors else 'timeout'}"
            )

    def close(self) -> None:
        self.producer.flush(self.timeout)


def _trace_ids() -> tuple[str, str]:
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    except ImportError:  # pragma: no cover - OpenTelemetry is a dependency of every template
        pass
    return "", ""


class Auditor:
    def __init__(
        self,
        sink: Sink,
        *,
        service: str,
        product: str = "",
        environment: str = "",
        data_class: str = "general",
        required: bool = True,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.sink, self.service, self.product = sink, service, product
        self.environment, self.data_class, self.required, self.now = environment, data_class, required, now

    def record(
        self,
        action: str,
        *,
        actor: Actor,
        resource: tuple[str, Iterable[str]] = ("", ()),
        outcome: Outcome = "success",
        reason: str = "",
        on_behalf_of: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish one audit record. `details` may hold counts, codes and ids, never field values."""
        rtype, ids = resource[0], sorted({str(i) for i in resource[1] if i})
        trace_id, span_id = _trace_ids()
        event: dict[str, Any] = {
            "specversion": "1.0",
            "id": str(uuid.uuid4()),
            "source": f"{self.product}/{self.service}" if self.product else self.service,
            "type": EVENT_TYPE,
            "time": self.now().isoformat(),
            "subject": f"{rtype}/{','.join(ids)}" if rtype else action,
            "datacontenttype": "application/json",
            "data": {
                "product": self.product,
                "service": self.service,
                "environment": self.environment,
                "data_class": self.data_class,
                "action": action,
                "actor": {"id": actor.id, "type": actor.type, "client": actor.client},
                "on_behalf_of": on_behalf_of,
                "resource": {"type": rtype, "ids": ids},
                "outcome": outcome,
                "reason": reason[:300],
                "trace_id": trace_id,
                "span_id": span_id,
                "details": details or {},
            },
        }
        try:
            # Records about one resource stay in order: they share a partition.
            self.sink.publish(event["subject"], event)
        except Exception as exc:
            if self.required:
                raise exc if isinstance(exc, AuditUnavailable) else AuditUnavailable(str(exc)) from exc
            log.error("audit record dropped (%s): %s %s", type(exc).__name__, action, event["subject"])
        return event


# Data classes whose audit records must travel encrypted (they name people and records).
ENCRYPTED_CLASSES = ("phi", "payments")


def require_encryption(config: dict[str, Any] | None, data_class: str, what: str = "audit") -> None:
    """PHI and payments products never talk to Kafka in plain text: refuse to start instead."""
    protocol = str((config or {}).get("security.protocol", "PLAINTEXT")).upper()
    if data_class in ENCRYPTED_CLASSES and protocol not in ("SSL", "SASL_SSL"):
        raise RuntimeError(
            f"The {what} Kafka connection of a {data_class} product must use TLS: set security.protocol "
            f"to SSL or SASL_SSL (it is {protocol})"
        )


def make_sink(
    kind: str,
    *,
    bootstrap: str = "",
    topic: str = "",
    config: dict[str, Any] | None = None,
    data_class: str = "general",
) -> Sink:
    if kind == "kafka":
        if not (bootstrap and topic):
            raise RuntimeError("The kafka audit sink needs a bootstrap server and a topic")
        require_encryption(config, data_class)
        return KafkaSink(bootstrap, topic, config=config)
    if kind == "log":
        return LogSink()
    if kind == "memory":
        return MemorySink()
    raise RuntimeError(f"Unknown audit sink {kind!r}: use kafka, log or memory")
