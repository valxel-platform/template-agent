"""Who calls this agent: per request, a log line, span attributes and metrics naming the OAuth client
and the calling app, never the person; and how the agent names itself to the MCP servers it calls."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent import usage
from agent.api import create_app
from agent.model import ScriptedModel
from agent.settings import Settings
from agent.telemetry import JsonFormatter
from agent.tools import McpToolHost

from .conftest import CREATE, FIND, reply, text

TOKENS = {
    "web": {
        "sub": "nurse-7",
        "azp": "care-web",
        "scope": "agent:run agent:review",
        "organization": "north-clinic",
    },
    "bridge": {
        "sub": "svc-events",
        "client_id": "event-bridge",
        "scope": "agent:run",
        "organization": "north-clinic",
    },
    "no-scope": {"sub": "svc-x", "azp": "other-app", "scope": "", "organization": "north-clinic"},
}


def verify(token: str) -> dict[str, Any]:
    if token not in TOKENS:
        raise ValueError("bad token")
    return TOKENS[token]


@pytest.fixture(autouse=True)
def fresh() -> None:
    usage.reset()


@pytest.fixture
def client(tools: McpToolHost) -> TestClient:
    model = ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))])
    app = create_app(
        Settings(name="intake-agent"),
        model=model,
        tools=tools,
        verifier=verify,
        checkpointer=InMemorySaver(),
        synchronous=True,
    )
    return TestClient(app)


def sample(text_: str, name: str, **labels: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text_.splitlines()
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items())
    )


def test_each_request_counts_by_route_client_app_and_status(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="agent.requests")
    ua = {"User-Agent": "valxel-bff/1.4.0 (spring)"}
    run = client.post(
        "/runs", json={"task": {"text": "Ana Lopez"}}, headers={"Authorization": "Bearer web", **ua}
    ).json()
    client.get(f"/runs/{run['run_id']}", headers={"Authorization": "Bearer web", **ua})
    client.post("/events", json={"id": "e-1", "type": "referral"}, headers={"Authorization": "Bearer bridge"})
    client.post("/runs", json={"task": {}}, headers={"Authorization": "Bearer no-scope"})  # 403
    client.post("/runs", json={"task": {}})  # no token: 401
    client.get("/healthz")
    metrics = client.get("/metrics").text
    assert (
        sample(
            metrics,
            "agent_requests_total",
            route="/runs",
            method="POST",
            client="care-web",
            app="valxel-bff",
            status="202",
        )
        == 1
    )
    # The route's template, never the run id in the path.
    assert sample(metrics, "agent_requests_total", route="/runs/{run_id}", client="care-web") == 1
    assert run["run_id"] not in metrics
    assert sample(metrics, "agent_requests_total", route="/events", client="event-bridge", status="202") == 1
    assert sample(metrics, "agent_requests_total", route="/runs", client="other-app", status="403") == 1
    assert sample(metrics, "agent_requests_total", route="/runs", client="none", status="401") == 1
    assert sample(metrics, "agent_request_duration_seconds_count", route="/runs", client="care-web") == 1
    assert "/healthz" not in metrics and "/metrics" not in metrics
    first = next(r for r in caplog.records if r.name == "agent.requests")
    assert first.getMessage() == "POST /runs 202"
    line = json.loads(JsonFormatter("intake-agent").format(first))
    assert line["agent.client.id"] == "care-web" and line["agent.client.app"] == "valxel-bff"
    # Never the person or the request: no subject, no task text.
    assert "nurse-7" not in json.dumps(line) and "Ana" not in json.dumps(line)


def test_names_are_cleaned_and_bounded() -> None:
    from prometheus_client import CollectorRegistry, generate_latest

    reg = CollectorRegistry()
    m = usage.Metrics(reg)
    for n in range(usage.MAX_CLIENTS):
        usage.record(
            m, route="/runs", method="POST", client_id=f"c-{n}", user_agent="", status=202, seconds=0.01
        )
    usage.record(
        m, route="/runs", method="POST", client_id="one-too-many", user_agent="", status=202, seconds=0.01
    )
    out = generate_latest(reg).decode()
    assert 'client="other"' in out and "one-too-many" not in out and 'app="unknown"' in out
    assert usage.app_from_user_agent("Mozilla/5.0 (X11)") == "mozilla"
    assert usage.client_of({"azp": "care-web", "client_id": "x"}) == "care-web"


def test_the_agent_names_itself_to_the_mcp_servers_it_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp.client.client as mc

    seen: dict[str, Any] = {}

    class Fake:
        def __init__(self, target: Any, **kw: Any) -> None:
            seen.update(kw)

        async def __aenter__(self) -> Fake:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(mc, "Client", Fake)
    host = McpToolHost.from_settings(Settings(name="intake-agent"))
    host.servers = {"care": object()}  # in-process: no token needed

    async def nothing(c: Any) -> None:
        return None

    import anyio

    anyio.run(host._session, "care", nothing)
    assert seen["client_info"].name == "intake-agent"
