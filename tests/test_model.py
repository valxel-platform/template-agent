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


def test_gateway_model_streams_text_as_it_is_written() -> None:
    def sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    body = "".join(
        [
            sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "m",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 3, "output_tokens": 0},
                    },
                },
            ),
            sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            ),
            sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            ),
            sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ),
            sse("message_stop", {"type": "message_stop"}),
        ]
    )
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    model = GatewayModel(
        base_url="http://gw.test",
        api_key="k",
        model="m",
        max_tokens=8,
        agent="a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    chunks: list[str] = []
    out = model.respond(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        labels={"run": "r"},
        on_text=chunks.append,
    )
    assert chunks == ["Hel", "lo"] and seen["body"]["stream"] is True
    assert out["content"] == [{"type": "text", "text": "Hello"}] and out["stop_reason"] == "end_turn"
