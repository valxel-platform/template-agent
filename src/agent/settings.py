"""Configuration from the environment (OpenChoreo config and secret references). Nothing secret
has a default, so a misconfigured agent fails at start rather than running half-protected."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpServer(BaseModel):
    name: str
    url: str
    # The server's audience in the product's Keycloak realm (its client id), and the scopes this
    # agent may request there. Calls carry a token exchanged for the person behind the step
    # (delegation.py); there are no fixed tokens. Reads, which run before anyone approves, get
    # read_scope; approved changes get write_scope, so the server refuses a change made with a
    # read token. `scope` is the older single setting, used for both when they're unset.
    audience: str = ""
    read_scope: str = ""
    write_scope: str = ""
    scope: str = ""

    def scopes(self) -> tuple[str, str]:
        return (self.read_scope or self.scope, self.write_scope or self.scope)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_nested_delimiter="__")

    name: str = "intake-agent"
    # Models, through the product's LLM gateway (Anthropic Messages API).
    gateway_url: str = "http://llm-gateway:8080"
    gateway_key: str = ""
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    model_timeout_s: float = 120.0  # one model call; the SDK's default is 10 minutes
    # Tools: MCP servers this agent may use, and which of their tools (fnmatch patterns).
    mcp_servers: list[McpServer] = Field(default_factory=list)
    # Deny by default: list each tool (or a server's tools, "care__*") this agent may use. Read
    # tools run by themselves; anything else waits for a person's approval.
    allowed_tools: list[str] = Field(default_factory=list)
    # Which of those run without a person's approval: this agent's own list, never the server's
    # word for it (an MCP readOnlyHint is a hint). Everything else is proposed and approved first.
    read_only_tools: list[str] = Field(default_factory=list)
    # Guardrails
    max_steps: int = 8  # model turns per run before a person is asked to take over
    # Durable runs: Postgres in deployed environments, in memory otherwise.
    database_url: str = ""
    database_pool: int = 10  # connections per replica (each executing thread holds one more)
    # Conversations: how many recent exchanges the model sees, and how long progress is kept.
    chat_history_turns: int = 20
    chat_max_chars: int = 8000
    progress_retention_s: float = 86400.0
    # Long-term memory (memory.py): off unless set. A conversation's person may ask the agent to
    # remember things across conversations, kept per person and tenant, for this many days.
    memory: bool = False
    memory_retention_days: float = 90.0
    memory_max_per_person: int = 200
    # Who may start runs and decide approvals: OAuth 2.0 bearer tokens from the product's IdP.
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    # Acting for people at MCP servers: token exchange at the realm's token endpoint, as this
    # agent's client. private_key_jwt (a key file) rather than a shared secret where possible.
    token_url: str = ""  # https://id.<domain>/realms/<product>/protocol/openid-connect/token
    client_id: str = ""
    client_secret: str = ""
    client_key_file: str = ""
    client_key_id: str = ""
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
    flags_timeout_s: float = 0.25  # a slow Flipt never slows a step down by more than this
    flags_cache_s: float = 30.0  # how long a flag's answer for a caller is reused
    run_scope: str = "agent:run"
    review_scope: str = "agent:review"
    # Who may see and decide a run: the person or service that started it, and reviewers of the
    # same tenant (the token claim naming the caller's organization; Keycloak organizations).
    tenant_claim: str = "organization"
    # Separation of duties: the person who asked for a change can't approve it themselves.
    allow_self_approval: bool = False
