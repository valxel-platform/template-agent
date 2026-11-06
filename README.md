# template-agent

The Vrixel template for a **workflow agent**: a durable LangGraph agent that works on a task
using the product's MCP tools, with every change approved by a person and every model call going
through the product's LLM gateway.

The example agent turns a referral into a care-team task. Replace the prompt, the tools it may
use and the eval cases with your own; keep the structure.

```
event / POST /runs
      │
   think ──── read-only tools run at once ───┐
      │  proposes changes (write tools)      │
      v                                      │
   review ── a person approves or rejects ── act ── runs approved changes (idempotent) ──> think
      │
   think ── no more tools ──> done (the model's summary is the result)
   think ── step limit ──> a person resumes or stops the run
```

## What the template decides for you

| Concern | How |
| --- | --- |
| Models | Anthropic SDK pointed at the product's LLM gateway (`AGENT_GATEWAY_URL`). The gateway enforces the data class (PHI only to BAA providers), redacts, meters spend and holds the provider keys; the agent holds only a gateway key. System prompt and tools are prompt-cached. |
| Tools | The product's MCP servers (`AGENT_MCP_SERVERS`), filtered by an allowlist (`AGENT_ALLOWED_TOOLS`). Tools annotated `readOnlyHint` run immediately; everything else, including unannotated tools, needs approval. |
| Acting for people | Every MCP call carries a token Keycloak exchanged (RFC 8693) for that server only, on behalf of the person behind the step: the run's starter for its steps, the approver for the changes they approved. No fixed tokens; people's tokens stay in memory, never in checkpoints (`delegation.py`). |
| Human approval | A LangGraph interrupt. Reviewers can approve all, approve some (`approved_ids`) or reject with a note the model sees. The reviewer's identity is in the trace. |
| Safe retries | Each approved call carries `idempotency_key = <run>:<tool_use_id>`; `POST /runs` takes an `Idempotency-Key`; events are keyed by their id. |
| Durability | Runs are checkpointed in Postgres (`AGENT_DATABASE_URL`), so a run can wait days for review and survives restarts. |
| Runaway runs | `AGENT_MAX_STEPS` model turns, then a person decides to resume or stop. |
| Auth | OAuth 2.0 bearer tokens from the product's IdP: `agent:run` to start and read runs, `agent:review` to decide. |
| Audit | Who started each run, who approved or rejected which changes, each change the agent applied (on whose behalf) and stops and resumes are published as audit records (CloudEvents, `vrixel.audit.v1`) to the product's Kafka audit topic; the platform's audit sink keeps them in an append-only, hash-chained table and archives them to object storage. If a decision can't be recorded it isn't applied, and the run says so. The run's own trace (`GET /runs/{id}`) still shows every model turn, tool call and refusal. |
| Tracing | OpenTelemetry: a span per run, model turn (`chat`, with model and token counts) and tool call (`execute_tool`), following the GenAI conventions and never carrying prompt or result text. Runs join the trace of the request that started them, and `traceparent` goes on to the gateway and MCP servers. Logs are JSON with `trace.id`. |
| Observability | `/metrics` (runs started/finished, decisions, errors), `/healthz`, `/readyz` (MCP servers reachable). |
| Data minimization | The prompt tells the model to repeat only what the next step needs; evals check it. |
| Feature flags | OpenFeature with the OFREP provider, pointed at the product's Flipt (`AGENT_FLAGS_URL`). The SDLC tool creates each feature's flag; put new behavior behind `flags.enabled("<key>")`. Unset, every flag is off. |

## API

```
POST /runs                  {"task": {...}}           Idempotency-Key header   (agent:run)
POST /events                {"id", "type", "text"}                             (agent:run)
GET  /runs/{id}             status, result, waiting_for, trace                 (agent:run)
POST /runs/{id}/decision    {"decision": "approve"|"reject"|"resume"|"stop",
                             "approved_ids": [...], "note": "..."}             (agent:review)
```

Status is `working`, `awaiting_review`, `needs_attention`, `done`, `stopped` or `failed`.

## Evals

Agents change behavior when the model, the prompt or a tool changes, so they are tested on what
they do, not only on code paths.

- `evals/cases.jsonl`: tasks with expectations: tools that must or must not be called, their
  arguments, status, text the result must contain, identifiers it must not repeat, a step budget.
- `evals/world.py`: an in-memory, seeded stand-in for the MCP servers.
- `evals/cassettes/`: recorded model responses.

| When | Command | Model | Status |
| --- | --- | --- | --- |
| Every PR (required check) | `uv run python -m evals.run` | recorded | `evals / report` |
| After deploy to dev | `uv run python -m evals.run --live` | real, through the dev gateway | `integration / report` |
| Re-record after a prompt change | `uv run python -m evals.run --live --record` | real | |

The pipeline's quality policy reads these statuses like any other suite.

## Develop

```
uv sync
uv run pytest -m 'not evals'       # unit tests with coverage
uv run python -m evals.run         # recorded evals
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Run locally against a gateway:

```
export AGENT_GATEWAY_URL=http://localhost:8080 AGENT_GATEWAY_KEY=...
export AGENT_MCP_SERVERS='[{"name":"care","url":"http://localhost:9000/mcp"}]'   # no audience: no token locally
export AGENT_OIDC_ISSUER=... AGENT_OIDC_AUDIENCE=... AGENT_OIDC_JWKS_URL=...
uv run agent
```

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `AGENT_NAME` | `intake-agent` | Sent to the gateway as `x-vrixel-agent` for budgets and usage |
| `AGENT_GATEWAY_URL`, `AGENT_GATEWAY_KEY` | | The product's LLM gateway and this agent's key |
| `AGENT_MODEL`, `AGENT_MAX_TOKENS` | `claude-sonnet-5-5`, `2048` | The gateway's model allowlist applies |
| `AGENT_MCP_SERVERS` | `[]` | JSON list of `{name, url, audience, scope}`: `audience` is the server's client in the realm |
| `AGENT_TOKEN_URL`, `AGENT_CLIENT_ID` | | The realm's token endpoint and this agent's client, for token exchange |
| `AGENT_CLIENT_KEY_FILE`, `AGENT_CLIENT_KEY_ID` | | `private_key_jwt` client authentication (preferred), or `AGENT_CLIENT_SECRET` |
| `AGENT_ALLOWED_TOOLS` | `["*"]` | fnmatch patterns on `<server>__<tool>`; list tools explicitly in production |
| `AGENT_MAX_STEPS` | `8` | |
| `AGENT_DATABASE_URL` | | Postgres; in memory if unset (development only) |
| `AGENT_OIDC_ISSUER`, `AGENT_OIDC_AUDIENCE`, `AGENT_OIDC_JWKS_URL` | | Required |
| `AGENT_RUN_SCOPE`, `AGENT_REVIEW_SCOPE` | `agent:run`, `agent:review` | |
| `AGENT_FLAGS_URL` | | The product's Flipt; unset, every feature flag reads as off |
| `AGENT_FLAGS_ENVIRONMENT`, `AGENT_FLAGS_NAMESPACE` | | Flipt environment (e.g. `production`) and the product's namespace (e.g. `emr`) |
| `AGENT_FLAGS_TOKEN` | | A Flipt client token, if the product's Flipt requires one |
| `AGENT_PRODUCT`, `AGENT_ENVIRONMENT`, `AGENT_DATA_CLASS` | | Stamped on every audit record (`data_class`: `general`, `phi` or `payments`) |
| `AGENT_AUDIT_SINK` | `log` | `kafka` in deployed environments; `log` writes JSON lines locally |
| `AGENT_AUDIT_KAFKA_BOOTSTRAP`, `AGENT_AUDIT_TOPIC` | `audit.v1` | The product's Kafka and its audit topic |
| `AGENT_AUDIT_KAFKA_CONFIG` | `{}` | JSON of extra client settings (`security.protocol`, `sasl.*`) |
| `AGENT_AUDIT_REQUIRED` | `true` | A decision or change whose record isn't acknowledged by Kafka is not applied |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | | Standard OpenTelemetry settings; point the endpoint at the data plane's collector |

## Delivery

CI (`ci.yml`) runs `unit` and `evals`, dependency review and an image build; CodeQL comes from
the repo's default setup. These are the required checks the Vrixel SDLC sets on repos with the
`agent` role. The SDLC deploys through OpenChoreo and then sends `vrixel-deployed`, which runs
`post-deploy.yml`: the smoke check at every stage, and live evals in dev.
