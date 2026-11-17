"""Tools come from the product's MCP servers. The agent sees only tools on its allowlist, and it
treats them by what they do: tools on its read-only list run as soon as the model asks; anything
else is proposed, shown to a person, and runs only after approval, with an idempotency key so a
retry never repeats it.

Which tools are read-only is this agent's own setting (AGENT_READ_ONLY_TOOLS), not the server's:
MCP annotations are hints, and a server that mislabels a write as readOnlyHint would otherwise get
it run with no approval. A server's hint can only make a tool stricter: one it marks as writing or
destructive needs approval even if the list names it. Reads run with a read-scope token and
approved changes with a write-scope one, so the server itself refuses a change made unapproved."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from typing import Any, Protocol

import anyio

from . import delegation


@dataclass(frozen=True)
class Tool:
    name: str  # as the model sees it: "<server>__<tool>"
    server: str
    remote_name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolHost(Protocol):
    def tools(self) -> list[Tool]: ...
    def call(self, tool: Tool, args: dict[str, Any], idempotency_key: str) -> str: ...


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("agent")
    except PackageNotFoundError:
        return ""


class McpToolHost:
    """MCP servers over Streamable HTTP (or in-process servers in tests)."""

    def __init__(
        self,
        servers: dict[str, Any],
        allowed: list[str],
        audiences: dict[str, tuple[str, str] | tuple[str, str, str]] | None = None,
        exchange: delegation.TokenExchange | None = None,
        client_name: str = "valxel-agent",
        client_version: str = "",
        read_only: list[str] | None = None,
    ) -> None:
        self.servers = servers  # name -> URL or in-process server
        # How this agent names itself to the MCP servers it calls (clientInfo, sent with every
        # request): their usage records show which agent called, not just "mcp".
        self.client_name, self.client_version = client_name, client_version
        self.allowed = allowed
        self.read_only = list(read_only or [])  # patterns: tools that run without approval
        # name -> (audience, read scope, write scope); a two-item (audience, scope) uses it for both
        self.audiences = {
            k: (v[0], v[1], v[1]) if len(v) == 2 else (v[0], v[1], v[2]) for k, v in (audiences or {}).items()
        }
        self.exchange = exchange
        self._tools: list[Tool] | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> McpToolHost:
        return cls(
            {s.name: s.url for s in settings.mcp_servers},
            settings.allowed_tools,
            {s.name: (s.audience, *s.scopes()) for s in settings.mcp_servers},
            delegation.from_settings(settings),
            client_name=settings.name,
            client_version=_version(),
            read_only=settings.read_only_tools,
        )

    def bearer(self, server: str, *, listing: bool = False, write: bool = False) -> str | None:
        """The token for a call to `server`: for the person behind this step, exchanged for that
        server only, with its write scope for an approved change and its read scope otherwise.
        Listing tools uses the agent's own identity (metadata, no records)."""
        audience, read_scope, write_scope = self.audiences.get(server, ("", "", ""))
        scope = write_scope if write else read_scope
        if not audience:
            return None  # a server outside the realm (local development, in-process tests)
        if self.exchange is None:
            raise delegation.DelegationError(
                f"{server} needs a token: set AGENT_TOKEN_URL and AGENT_CLIENT_ID"
            )
        if listing:
            return self.exchange.for_agent(audience, scope)
        person = delegation.current()
        if not person:
            raise ToolError(f"No person to act for at {server}: a person needs to approve again")
        return self.exchange.for_person(person, audience, scope)

    async def _session(self, server: str, fn: Any, listing: bool = False, write: bool = False) -> Any:
        from mcp.client.client import Client
        from mcp.client.streamable_http import (  # type: ignore[attr-defined]
            create_mcp_http_client,
            streamable_http_client,
        )

        target = self.servers[server]
        if isinstance(target, str):
            token = self.bearer(server, listing=listing, write=write)
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            target = streamable_http_client(target, http_client=create_mcp_http_client(headers=headers))
        from mcp.types import Implementation

        info = Implementation(name=self.client_name, version=self.client_version or "0")
        async with Client(target, client_info=info) as client:
            return await fn(client)

    def tools(self) -> list[Tool]:
        if self._tools is None:
            found: list[Tool] = []
            for server in self.servers:

                async def list_tools(c: Any) -> Any:
                    return (await c.list_tools()).tools

                for t in anyio.run(self._session, server, list_tools, True):
                    name = f"{server}__{t.name}"
                    if not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(t.name, p) for p in self.allowed):
                        continue
                    ann = t.annotations
                    listed = any(
                        fnmatch.fnmatch(name, p) or fnmatch.fnmatch(t.name, p) for p in self.read_only
                    )
                    # The server can only make it stricter: a tool it says writes or destroys needs approval.
                    writes = bool(ann and (ann.read_only_hint is False or ann.destructive_hint))
                    read_only = listed and not writes
                    schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                    found.append(Tool(name, server, t.name, t.description or "", dict(schema), read_only))
            self._tools = found
        return list(self._tools)

    def call(self, tool: Tool, args: dict[str, Any], idempotency_key: str) -> str:
        payload = dict(args)
        if not tool.read_only and "idempotency_key" in tool.input_schema.get("properties", {}):
            payload["idempotency_key"] = idempotency_key

        async def go(c: Any) -> Any:
            return await c.call_tool(tool.remote_name, payload)

        res = anyio.run(self._session, tool.server, go, False, not tool.read_only)
        text = "".join(getattr(b, "text", "") for b in getattr(res, "content", []) or [])
        if getattr(res, "is_error", None) or getattr(res, "isError", None):
            raise ToolError(text or f"{tool.name} failed")
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        return json.dumps(structured) if structured and not text else text


class ToolError(RuntimeError):
    pass
