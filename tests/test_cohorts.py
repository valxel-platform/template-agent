"""Requests per flag, on and off, for the SDLC tool's gradual releases."""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider

from agent import cohorts, flags

KEY = "emr.emr-feat-3.referral-triage"


@pytest.fixture
def app():
    flags.use_provider(InMemoryProvider({KEY: InMemoryFlag("on", {"on": True, "off": False})}))
    cohorts.drain()
    a = FastAPI()
    a.add_middleware(cohorts.CohortMiddleware)

    @a.get("/ok")
    def ok(user: str = "") -> dict:
        flags.enabled(KEY, user=user)
        flags.enabled(KEY, user=user)  # twice in one request: still one request
        return {}

    @a.get("/fail")
    def fail() -> dict:
        flags.enabled(KEY)
        raise RuntimeError("boom")

    @a.get("/plain")
    def plain() -> dict:
        return {}

    yield a
    flags.use_provider(InMemoryProvider({}))


def test_each_request_counts_once_on_the_side_the_caller_got(app):
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/ok?user=nurse-1").status_code == 200
    assert client.get("/fail").status_code == 500
    assert client.get("/plain").status_code == 200
    flags.use_provider(InMemoryProvider({KEY: InMemoryFlag("off", {"on": True, "off": False})}))
    client.get("/ok?user=nurse-2")
    [line] = cohorts.drain()
    assert line.startswith(cohorts.MARKER + " ")
    assert json.loads(line[len(cohorts.MARKER) :]) == {
        "flag": KEY,
        "on": {"requests": 2, "errors": 1, "slow": 0},
        "off": {"requests": 1, "errors": 0, "slow": 0},
    }
    assert cohorts.drain() == []  # counts start again


def test_outside_a_request_nothing_is_counted_and_reports_are_logged(caplog):
    flags.use_provider(InMemoryProvider({KEY: InMemoryFlag("on", {"on": True, "off": False})}))
    flags.enabled(KEY)
    assert cohorts.drain() == []
    token = cohorts.begin()
    cohorts.seen(KEY, True)
    cohorts.seen("not a key!", True)
    cohorts.end(token, failed=False, elapsed_ms=cohorts.SLOW_MS + 1)
    with caplog.at_level(logging.INFO, logger="agent.cohorts"):
        cohorts.report()
    assert [r.getMessage() for r in caplog.records] == [
        cohorts.MARKER
        + ' {"flag":"'
        + KEY
        + '","on":{"requests":1,"errors":0,"slow":1},"off":{"requests":0,"errors":0,"slow":0}}'
    ]
    cohorts.start_reporting(period_s=3600)
    cohorts.start_reporting(period_s=3600)  # once only
    flags.use_provider(InMemoryProvider({}))
