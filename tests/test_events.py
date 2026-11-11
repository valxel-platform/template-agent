"""Product events: counted for the SDLC tool's outcome loop, never about people."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider
from prometheus_client import CollectorRegistry, generate_latest

from agent import cohorts, events, flags

KEY = "emr.emr-feat-3.referral-triage"


@pytest.fixture(autouse=True)
def fresh():
    events.reset()
    cohorts.drain()
    yield
    events.reset()


def test_events_are_counted_into_one_summary_line_a_minute():
    events.record("referral_triaged")
    events.record("referral_triaged", urgency="routine")
    events.record("referral.escalated")
    [line] = events.drain()
    # The SDLC tool reads exactly this: the marker, then a JSON object of counts.
    assert line.startswith(events.MARKER + " ")
    assert json.loads(line[len(events.MARKER) + 1 :]) == {"referral.escalated": 1, "referral_triaged": 2}
    assert events.drain() == []  # counts start again; nothing happened, no line


def test_the_counter_is_on_metrics_with_its_categories():
    registry = CollectorRegistry()
    events.register(registry)
    events.record("referral_triaged", urgency="routine")
    events.record("referral_triaged", urgency="routine")
    events.record("referral_triaged")
    events.drain()  # the summary line resets; the counter never does
    text = generate_latest(registry).decode()
    assert 'vrixel_events_total{event="referral_triaged",urgency="routine"} 2.0' in text
    assert 'vrixel_events_total{event="referral_triaged"} 1.0' in text


@pytest.mark.parametrize(
    "attribute",
    [
        "email",
        "patient_id",
        "first_name",
        "user_id",
        "patient_email",
        "account_number",
        "member_name",
        "phone",
    ],
)
def test_attributes_never_reach_a_person(attribute):
    with pytest.raises(ValueError, match="person"):
        events.record("referral_triaged", **{attribute: "x"})


def test_names_are_lowercase_words_and_values_short_categories():
    with pytest.raises(ValueError, match="lowercase"):
        events.record("ReferralTriaged")
    with pytest.raises(ValueError):
        events.record("referral triaged")
    with pytest.raises(ValueError, match="short category"):
        events.record("referral_triaged", note="x" * 65)
    events.record("referral_triaged", clinic_name="north", urgency="routine")  # a place, not a person


def test_a_requests_events_count_on_the_side_of_each_flag_it_evaluated():
    flags.use_provider(InMemoryProvider({KEY: InMemoryFlag("on", {"on": True, "off": False})}))
    app = FastAPI()
    app.add_middleware(cohorts.CohortMiddleware)

    @app.get("/triage")
    def triage(user: str = "") -> dict:
        if flags.enabled(KEY, user=user):
            events.record("referral_triaged")
            events.record("referral_triaged")
        return {}

    @app.get("/plain")
    def plain(user: str = "") -> dict:
        flags.enabled(KEY, user=user)
        return {}

    client = TestClient(app)
    client.get("/triage", params={"user": "nurse-1"})
    client.get("/plain", params={"user": "nurse-2"})
    [line] = cohorts.drain()
    got = json.loads(line[len(cohorts.MARKER) + 1 :])
    assert got["on"] == {"requests": 2, "errors": 0, "slow": 0, "events": {"referral_triaged": 2}}
    events.record("referral_triaged")  # outside a request: the summary line counts it, no cohort does
    assert cohorts.drain() == []
