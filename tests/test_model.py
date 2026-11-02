from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

from agent.model import GatewayModel, ScriptedModel


def test_gateway_model_sends_cacheable_requests_with_labels() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    model = GatewayModel(
        base_url="http://gw.test",
        api_key="gw-key",
        model="claude-sonnet-5-5",
        max_tokens=64,
        agent="intake-agent",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = model.respond(
        system="sys",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "a", "description": "", "input_schema": {"type": "object"}}],
        labels={"run": "r1"},
    )
    assert out["content"] == [{"type": "text", "text": "hi"}]
    assert seen["url"] == "http://gw.test/v1/messages"
    assert seen["headers"]["authorization"] == "Bearer gw-key"  # the gateway accepts this or x-api-key
    assert seen["headers"]["x-vrixel-agent"] == "intake-agent" and seen["headers"]["x-vrixel-run"] == "r1"
    assert seen["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen["body"]["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_gateway_key_is_required() -> None:
    with pytest.raises(ValueError, match="GATEWAY_KEY"):
        GatewayModel(base_url="http://gw", api_key="", model="m", max_tokens=1, agent="a")


def test_scripted_model_runs_out() -> None:
    m = ScriptedModel([lambda msgs: {"content": [{"type": "text", "text": str(len(msgs))}]}])
    assert m.respond(system="", messages=[{}], tools=[], labels={})["content"][0]["text"] == "1"
    with pytest.raises(RuntimeError):
        m.respond(system="", messages=[], tools=[], labels={})
