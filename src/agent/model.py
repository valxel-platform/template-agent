"""Model access. Always through the product's LLM gateway with the standard Anthropic SDK: the
gateway decides which providers are allowed, redacts what must not leave, and meters spend.
The system prompt and tool definitions are marked for prompt caching, so repeated turns of a run
pay for them once.

`on_text`, when given, receives the reply's text as the model writes it (a conversation shows it
to the person while it is written); the full reply is returned either way."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import anthropic


class Model(Protocol):
    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        labels: dict[str, str],
        on_text: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]: ...


class GatewayModel:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        agent: str,
        http_client: Any = None,
        timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("AGENT_GATEWAY_KEY is required")
        self.client = anthropic.Anthropic(
            base_url=base_url, api_key=api_key, max_retries=3, http_client=http_client, timeout=timeout_s
        )
        self.model = model
        self.max_tokens = max_tokens
        self.agent = agent

    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        labels: dict[str, str],
        on_text: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        cached_tools = [dict(t) for t in tools]
        if cached_tools:
            cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
        kw: dict[str, Any] = {
            "model": model or self.model,  # a variant's model (variants.py), else the agent's
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": cached_tools,
            "extra_headers": {
                "x-vrixel-agent": self.agent,
                **{f"x-vrixel-{k}": v for k, v in labels.items()},
            },
        }
        if on_text is None:
            msg = self.client.messages.create(**kw)
        else:
            with self.client.messages.stream(**kw) as stream:
                for chunk in stream.text_stream:
                    on_text(chunk)
                msg = stream.get_final_message()
        out: dict[str, Any] = msg.model_dump(mode="json", exclude_none=True)
        return out


class ScriptedModel:
    """Offline model for tests and recorded evals: returns prepared responses in order, or calls a
    function of the conversation so far."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        labels: dict[str, str],
        on_text: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system": system,
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "model": model,
                "labels": labels,
            }
        )
        if not self.responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        nxt = self.responses.pop(0)
        out: dict[str, Any] = nxt(messages) if callable(nxt) else nxt
        if on_text is not None:
            for block in out.get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    on_text(block["text"])
        return out
