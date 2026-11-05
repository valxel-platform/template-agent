"""The agent acts for a person at every MCP server: the run's starter for its steps, the approver
for the changes they approved; never with a fixed token, never beyond the person."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent import delegation
from agent.api import create_app
from agent.model import ScriptedModel
from agent.settings import McpServer, Settings
from agent.tools import McpToolHost, Tool, ToolError

from .conftest import CREATE, FIND, reply, text


class Keycloak:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.body: dict[str, Any] = {"access_token": "exchanged-1", "expires_in": 300}

    def handler(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.requests.append({"form": form, "auth": request.headers.get("authorization")})
        return httpx.Response(self.status, json=self.body)


@pytest.fixture
def keycloak() -> Keycloak:
    return Keycloak()


def exchange(kc: Keycloak, **kw: Any) -> delegation.TokenExchange:
    clock = kw.pop("clock", lambda: 1000.0)
    return delegation.TokenExchange(
        "https://id.example.com/realms/emr/protocol/openid-connect/token",
        "emr-intake-agent",
        http=httpx.Client(transport=httpx.MockTransport(kc.handler)),
        clock=clock,
        **({"client_secret": "s3cret"} | kw),
    )


def test_a_persons_token_is_exchanged_for_the_server_and_cached(keycloak: Keycloak) -> None:
    now = [1000.0]
    ex = exchange(keycloak, clock=lambda: now[0])
    assert ex.for_person("nurse-token", "emr-care-mcp", "tasks:write") == "exchanged-1"
    assert ex.for_person("nurse-token", "emr-care-mcp", "tasks:write") == "exchanged-1"
    assert len(keycloak.requests) == 1
    form = keycloak.requests[0]["form"]
    assert form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": "nurse-token",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "audience": "emr-care-mcp",
        "scope": "tasks:write",
    }
    assert keycloak.requests[0]["auth"] == "Basic " + base64.b64encode(b"emr-intake-agent:s3cret").decode()
    ex.for_person("doctor-token", "emr-care-mcp", "tasks:write")
    assert len(keycloak.requests) == 2  # never another person's token
    now[0] += 271  # within 30s of expiry
    ex.for_person("nurse-token", "emr-care-mcp", "tasks:write")
    assert len(keycloak.requests) == 3


def test_the_agents_own_identity_is_for_tool_metadata(keycloak: Keycloak) -> None:
    exchange(keycloak).for_agent("emr-care-mcp")
    assert keycloak.requests[0]["form"] == {"grant_type": "client_credentials", "audience": "emr-care-mcp"}


def test_a_signed_assertion_replaces_the_secret(keycloak: Keycloak, tmp_path: Any) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "key.pem"
    pem.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    import time

    exchange(
        keycloak, client_secret="", client_key_file=str(pem), client_key_id="k1", clock=time.time
    ).for_person("t", "aud")
    req = keycloak.requests[0]
    assert req["auth"] is None
    assertion = req["form"]["client_assertion"]
    claims = jwt.decode(
        assertion,
        key.public_key(),
        algorithms=["RS256"],
        audience="https://id.example.com/realms/emr/protocol/openid-connect/token",
    )
    assert claims["iss"] == claims["sub"] == "emr-intake-agent"
    assert jwt.get_unverified_header(assertion)["kid"] == "k1"


def test_refusals_name_the_error_never_the_token(keycloak: Keycloak) -> None:
    keycloak.status, keycloak.body = 400, {"error": "invalid_target", "error_description": "nurse-token"}
    with pytest.raises(delegation.DelegationError, match=r"refused \(400, invalid_target\)$"):
        exchange(keycloak).for_person("nurse-token", "aud")
    keycloak.status, keycloak.body = 200, {}
    with pytest.raises(delegation.DelegationError, match="no access_token"):
        exchange(keycloak).for_person("nurse-token", "aud")
    with pytest.raises(ValueError, match="AGENT_CLIENT_KEY_FILE"):
        delegation.TokenExchange("u", "c")
    with pytest.raises(ValueError, match="AGENT_TOKEN_URL"):
        delegation.TokenExchange("", "c", client_secret="s")
    assert delegation.from_settings(Settings()) is None


def test_every_call_to_a_server_acts_for_the_person_behind_the_step(keycloak: Keycloak) -> None:
    host = McpToolHost(
        {"care": "https://care"}, ["*"], {"care": ("emr-care-mcp", "tasks:write")}, exchange(keycloak)
    )
    with pytest.raises(ToolError, match="No person to act for at care"):
        host.bearer("care")
    with delegation.acting_for("nurse-token"):
        assert host.bearer("care") == "exchanged-1"
    assert keycloak.requests[-1]["form"]["subject_token"] == "nurse-token"
    host.bearer("care", listing=True)
    assert keycloak.requests[-1]["form"]["grant_type"] == "client_credentials"
    assert McpToolHost({"local": "http://x"}, ["*"]).bearer("local") is None  # outside the realm
    no_exchange = McpToolHost({"care": "https://care"}, ["*"], {"care": ("emr-care-mcp", "")})
    with pytest.raises(delegation.DelegationError, match="AGENT_TOKEN_URL"):
        no_exchange.bearer("care")
    s = Settings(
        mcp_servers=[McpServer(name="care", url="https://care", audience="emr-care-mcp", scope="tasks:write")]
    )
    assert McpToolHost.from_settings(s).audiences == {"care": ("emr-care-mcp", "tasks:write")}


class RecordingTools:
    """Records whose token each tool call would act with."""

    def __init__(self, inner: McpToolHost) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str | None]] = []

    def tools(self) -> list[Tool]:
        return self.inner.tools()

    def call(self, tool: Tool, args: dict[str, Any], key: str) -> str:
        self.calls.append((tool.name, delegation.current()))
        return self.inner.call(tool, args, key)


def test_reads_act_for_the_starter_and_changes_for_the_approver(tools: McpToolHost) -> None:
    def token(sub: str, scope: str) -> str:
        return jwt.encode({"sub": sub, "scope": scope, "exp": 4_000_000_000}, "k" * 32, algorithm="HS256")

    starter, approver = token("svc-intake", "agent:run"), token("nurse-7", "agent:run agent:review")
    recorder = RecordingTools(tools)
    app = create_app(
        Settings(name="intake-agent"),
        model=ScriptedModel([reply(FIND), reply(text("Proposing."), CREATE), reply(text("Created."))]),
        tools=recorder,  # type: ignore[arg-type]
        verifier=lambda t: jwt.decode(t, "k" * 32, algorithms=["HS256"]),
        checkpointer=InMemorySaver(),
        synchronous=True,
    )
    client = TestClient(app)
    run = client.post(
        "/runs", json={"task": {"text": "Ana Lopez"}}, headers={"Authorization": f"Bearer {starter}"}
    ).json()
    client.post(
        f"/runs/{run['run_id']}/decision",
        json={"decision": "approve"},
        headers={"Authorization": f"Bearer {approver}"},
    )
    assert recorder.calls == [("care__find_patient", starter), ("care__create_task", approver)]
    # Tokens live in memory only: nothing in the run's state holds them.
    state = app.state.runner.graph.get_state(app.state.runner.config(run["run_id"])).values
    assert starter not in json.dumps(state, default=str) and approver not in json.dumps(state, default=str)


def test_expired_tokens_are_forgotten() -> None:
    from agent.api import Runner

    r = Runner(graph=None, max_steps=1)
    r.people["run-1"] = ("t", 1.0)
    assert r.person("run-1") is None and "run-1" not in r.people
    r.people["run-2"] = ("t", None)
    assert r.person("run-2") == "t"
    assert r.person("run-3") is None
    assert delegation.unverified_expiry("not a jwt") is None
    assert delegation.unverified_expiry(jwt.encode({"exp": 5}, "k" * 32, algorithm="HS256")) == 5.0
    assert delegation.unverified_expiry(jwt.encode({}, "k" * 32, algorithm="HS256")) is None
