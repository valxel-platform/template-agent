"""Configuration from the environment (OpenChoreo config and secret references). Nothing secret
has a default, so a misconfigured agent fails at start rather than running half-protected."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpServer(BaseModel):
    name: str
    url: str
    token_env: str = ""  # env var holding this agent's bearer token for the server


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_nested_delimiter="__")

    name: str = "intake-agent"
    # Models, through the product's LLM gateway (Anthropic Messages API).
    gateway_url: str = "http://llm-gateway:8080"
    gateway_key: str = ""
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    # Tools: MCP servers this agent may use, and which of their tools (fnmatch patterns).
    mcp_servers: list[McpServer] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=lambda: ["*"])
    # Guardrails
    max_steps: int = 8  # model turns per run before a person is asked to take over
    # Durable runs: Postgres in deployed environments, in memory otherwise.
    database_url: str = ""
    # Who may start runs and decide approvals: OAuth 2.0 bearer tokens from the product's IdP.
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    # Audit trail (audit.py): "kafka" in deployed environments, "log" locally.
    product: str = ""  # e.g. emr
    environment: str = ""  # e.g. production
    data_class: str = "general"  # general | phi | payments
    audit_sink: str = "log"
    audit_kafka_bootstrap: str = ""
    audit_topic: str = "audit.v1"
    audit_kafka_config: dict[str, str] = Field(default_factory=dict)  # security.protocol, sasl.* ...
    audit_required: bool = True  # a decision or change whose record can't be published fails
    # Feature flags: the product's Flipt over OFREP. Unset, every flag reads as its default (off).
    flags_url: str = ""
    flags_environment: str = ""  # Flipt environment, e.g. production (header x-flipt-environment)
    flags_namespace: str = ""  # the product's Flipt namespace, e.g. emr
    flags_token: str = ""  # a Flipt client token, if the product's Flipt requires authentication
    run_scope: str = "agent:run"
    review_scope: str = "agent:review"
