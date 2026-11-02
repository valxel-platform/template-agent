"""Model access. Always through the product's LLM gateway with the standard Anthropic SDK: the
gateway decides which providers are allowed, redacts what must not leave, and meters spend.
The system prompt and tool definitions are marked for prompt caching, so repeated turns of a run
pay for them once."""

from __future__ import annotations

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
    ) -> dict[str, Any]: ...


class GatewayModel:
    def __init__(
        self, *, base_url: str, api_key: str, model: str, max_tokens: int, agent: str, http_client: Any = None
    ) -> None:
        if not api_key:
            raise ValueError("AGENT_GATEWAY_KEY is required")
        self.client = anthropic.Anthropic(
            base_url=base_url, api_key=api_key, max_retries=3, http_client=http_client
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
    ) -> dict[str, Any]:
        cached_tools = [dict(t) for t in tools]
        if cached_tools:
            cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,  # type: ignore[arg-type]
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=cached_tools,  # type: ignore[arg-type]
            extra_headers={"x-vrixel-agent": self.agent, **{f"x-vrixel-{k}": v for k, v in labels.items()}},
        )
        return msg.model_dump(mode="json", exclude_none=True)


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
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        nxt = self.responses.pop(0)
        out: dict[str, Any] = nxt(messages) if callable(nxt) else nxt
        return out
