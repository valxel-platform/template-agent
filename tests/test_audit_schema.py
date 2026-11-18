"""The audit events this service builds meet the shared contract: valxel.audit.v1's JSON Schema,
owned by valxel-shared-services/ss-audit-sink and kept here as a byte-identical copy (the Valxel
SDLC checks daily that it still is). Update the copy from the sink; never edit it here."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from agent.audit import Actor, Auditor, MemorySink

SCHEMA = Path(__file__).resolve().parents[1] / "contracts" / "audit" / "valxel.audit.v1.schema.json"


def test_audit_events_meet_the_shared_schema() -> None:
    sink = MemorySink()
    auditor = Auditor(sink, service="svc", product="acme", environment="test", data_class="phi")
    auditor.record(
        "patient.read",
        actor=Actor(id="u-1", type="user", client="web"),
        resource=("patient", ["p-2", "p-1"]),
        details={"count": 2},
    )
    auditor.record(
        "patient.update",
        actor=Actor(id="agent-1", type="agent"),
        resource=("patient", ["p-1"]),
        outcome="denied",
        reason="missing scope",
        on_behalf_of="u-1",
    )
    auditor.record("token.issue", actor=Actor(id="svc-2", type="service"))
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert len(sink.events) == 3
    for event in sink.events:
        assert [e.message for e in validator.iter_errors(event)] == []
