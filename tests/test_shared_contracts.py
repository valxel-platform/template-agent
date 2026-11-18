"""The summary lines this service logs meet the shared contracts the SDLC tool reads them by:
contracts/events/, byte-identical copies of the platform's (the Valxel SDLC keeps them in step;
update them from there, never here)."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from agent import cohorts, events

KEY = "acme.acme-feat-1.triage"
HERE = Path(__file__).resolve().parents[1] / "contracts" / "events"


def _check(name: str, line: str, marker: str) -> None:
    schema = json.loads((HERE / f"{name}.schema.json").read_text(encoding="utf-8"))
    payload = json.loads(line.split(marker, 1)[1])
    assert [e.message for e in Draft202012Validator(schema).iter_errors(payload)] == []


def _request(on: bool) -> None:
    token = cohorts.begin()
    cohorts.seen(KEY, on)
    events.record("referral_triaged")
    cohorts.end(token, failed=not on, elapsed_ms=10.0)


def test_event_and_cohort_lines_meet_the_shared_contracts() -> None:
    events.drain()
    cohorts.drain()
    _request(True)
    _request(False)
    [line] = events.drain()
    _check("valxel.events.v1", line, events.MARKER)
    lines = cohorts.drain()
    assert len(lines) == 1
    _check("valxel.flag_cohorts.v1", lines[0], cohorts.MARKER)
