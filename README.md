# template-agent

The Valxel template for an **agent**: a durable LangGraph agent that works on tasks (runs) and
talks with people (conversations), using the product's MCP tools, with every change approved by a
person and every model call going through the product's LLM gateway. It runs on any number of
replicas.

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

A conversation is the same machine started by each message a person sends: the agent looks
things up for them as them, answers (streamed as it writes), and proposes changes they approve or
decline in the conversation.

## What the template decides for you

| Concern | How |
| --- | --- |
| Models | Anthropic SDK pointed at the product's LLM gateway (`AGENT_GATEWAY_URL`). The gateway enforces the data class (PHI only to BAA providers), redacts, meters spend and holds the provider keys; the agent holds only a gateway key. System prompt and tools are prompt-cached. |
| Tools | The product's MCP servers (`AGENT_MCP_SERVERS`), filtered by an allowlist (`AGENT_ALLOWED_TOOLS`). Tools annotated `readOnlyHint` run immediately; everything else, including unannotated tools, needs approval. |
| Acting for people | Every MCP call carries a token Keycloak exchanged (RFC 8693) for that server only, on behalf of the person behind the step: the run's starter for its steps, the approver for the changes they approved. No fixed tokens; people's tokens stay in memory, never in checkpoints (`delegation.py`). |
| Human approval | A LangGraph interrupt. Reviewers can approve all, approve some (`approved_ids`) or reject with a note the model sees. The reviewer's identity is in the trace. |
| Safe retries | Each approved call carries `idempotency_key = <run>:<tool_use_id>`; `POST /runs` takes an `Idempotency-Key`; events are keyed by their id. |
| Durability | Runs and conversations are checkpointed in Postgres (`AGENT_DATABASE_URL`), so a run can wait days for review and survives restarts. |
| Replicas | Any pod serves any request; nothing routes a caller back to the same pod. The pod that moves a thread on (a start, a message, a decision) holds the thread's Postgres advisory lock while it executes, so another pod asked to move it on at the same time answers 409 (an approval sent twice never applies twice) and every pod can say it is working. A pod that dies releases its lock with its connection; one that loses its lock connection stops before its next step (`locks.py`). |
| Streaming | The executing pod writes progress (the reply's text as it is written, each tool step, proposals, the end of the turn) to Postgres and NOTIFYs; any pod streams it as Server-Sent Events, and a reconnect resumes from `Last-Event-ID` (`store.py`). Progress is pruned after a day; the conversation itself stays in the checkpoint. |
| Runaway runs | `AGENT_MAX_STEPS` model turns, then a person decides to resume or stop. A conversation's model sees its last `AGENT_CHAT_HISTORY_TURNS` exchanges, so long conversations don't grow without bound. |
| Auth | OAuth 2.0 bearer tokens from the product's IdP: `agent:run` to start and read runs, `agent:review` to decide. A run is visible only to whoever started it and to reviewers of the same tenant (others get 404), and nobody approves a change they asked for themselves. |
| Audit | Who started each run, who approved or rejected which changes, each change the agent applied (on whose behalf) and stops and resumes are published as audit records (CloudEvents, `valxel.audit.v1`) to the product's Kafka audit topic; the platform's audit sink keeps them in an append-only, hash-chained table and archives them to object storage. If a decision can't be recorded it isn't applied, and the run says so. The run's own trace (`GET /runs/{id}`) still shows every model turn, tool call and refusal. |
| Tracing | OpenTelemetry: a span per run, model turn (`chat`, with model and token counts) and tool call (`execute_tool`), following the GenAI conventions and never carrying prompt or result text. Runs join the trace of the request that started them, and `traceparent` goes on to the gateway and MCP servers. Logs are JSON with `trace.id`. |
| Observability | `/metrics` (runs started/finished, decisions, errors, and requests by route, client and status), `/healthz`, `/readyz` (MCP servers reachable). |
| Who calls it | Every API request is counted by route, OAuth client, calling app and status: a JSON log line (`agent.requests`), span attributes, and `agent_requests_total` / `agent_request_duration_seconds`. No person, ids or request bodies. The agent names itself (clientInfo) to the MCP servers it calls, so their usage records show which agent called. |
| Data minimization | The prompt tells the model to repeat only what the next step needs; evals check it. |
| Product events | `events.record("referral_triaged")`: the counter `valxel_events_total` on /metrics, a `valxel.events` summary line a minute, and per-flag counts. The SDLC tool proves in staging that a feature's goal events fire, reads goals from them after release, and compares users with the feature on against off. Attributes are categories; one that names or identifies a person raises. |
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

Conversations (`agent:run`; a conversation is visible only to the person who started it):

```
POST /conversations                  {"text": "...", "title": "..."}   Idempotency-Key header
GET  /conversations                  the caller's conversations, latest first
GET  /conversations/{id}             messages (with each answer's steps), status, waiting_for
POST /conversations/{id}/messages    {"text": "..."}       409 while it answers or waits for a decision
POST /conversations/{id}/decision    {"decision": "approve"|"reject"|"resume"|"stop", ...}
GET  /conversations/{id}/stream      ?after=N or Last-Event-ID: Server-Sent Events
```

Sending answers 202 with `after`; open the stream from there. Events: `turn`, `text` (`delta`),
`step` (`tool`, `title`, `ok`), `approval` (`actions`), `reply` (`text`), `problem`, and `idle`,
which ends the turn and the stream. A conversation's status is `working`, `awaiting_review`,
`needs_attention`, `idle` or `failed`. The person approves changes in their own conversation (they
asked for them, and the changes run with their token); work that needs someone else's approval
goes through runs. People reach conversations through the product's BFF, which relays their
token and passes the stream through.

## Product events

When the HLD's Instrumentation section names this agent as the one that records an event, record
it where it happens:

```python
from agent import events

events.record("referral_triaged")
events.record("referral_triaged", urgency="routine")  # categories only
```

`agent.events` counts it on `valxel_events_total{event="referral_triaged"}` (on /metrics), in a
`valxel.events {"referral_triaged":12}` log line once a minute (the SDLC tool counts these in
staging before G6), and on each side of the request's `valxel.flag_cohorts` line as `"events"`
(compared during gradual releases). Names are lowercase words joined by `_` or `.`; an attribute
that names or identifies a person (a name, contact details, an id such as `patient_id`) raises.

## Who calls this agent

`agent.usage` records every API request (not `/healthz`, `/readyz` or `/metrics`) with two names
for the caller, neither of them a person:

| Name | From | Example |
| --- | --- | --- |
| `client` | The OAuth client the token was issued to (`azp` or `client_id`), set when the token is verified; `none` when refused before that | `care-web`, `event-bridge` |
| `app` | The calling software's User-Agent | `valxel-bff` |

Per request there is one log line, for example:

```json
{"message": "POST /runs 202", "http.route": "/runs", "http.request.method": "POST",
 "agent.client.id": "care-web", "agent.client.app": "valxel-bff", "http.response.status_code": 202,
 "event.duration_ms": 18.2, "trace.id": "..."}
```

and on `/metrics`, beside the run metrics:

```
agent_requests_total{route, method, client, app, status}
agent_request_duration_seconds{route, method, client, app}    # histogram
```

`route` is the route's template (`/runs/{run_id}`), never a path with ids in it. At most 100 OAuth
clients and 50 apps are named; later ones count as `other`. Which person started or decided which
run stays in the audit trail.

The other direction: the agent names itself to the MCP servers it calls (`clientInfo`, its
`AGENT_NAME` and version, sent on every request), so the MCP server template's usage records
(`mcp_tool_calls_total{app="intake-agent"}`) show which agent made each call.

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

## Prompt and model variants

A prompt or model change can go out to a share of people first, through the product's Flipt, like
a feature, without a redeploy (`src/agent/variants.py`):

1. Add the variant to `src/agent/variants.toml` in a pull request: its model (through the LLM
   gateway) and the prompts it uses for runs and conversations (new files under `prompts/`). The
   pull request is reviewed and its evals run like any other change.
2. In Flipt, create the variant flag `<AGENT_NAME>.variant` with variants named like the tables in
   `variants.toml`, and roll it out (a share of people, a tenant, everyone). A variant Flipt names
   but `variants.toml` doesn't have never runs; without the flag every run uses `default`.

A run picks its variant once, by the person who started it (or the triggering event), and keeps it
to the end, across replicas; a conversation keeps the variant of its first message. Every model call
carries it: the `chat` span's `valxel.variant`, the gateway's `variant` label (so its ledger and spans
show each variant's tokens and cost), and the run's trace. After the deploy to dev, the evals run
every variant against the real model, one test suite each (`python -m evals.run --live
--all-variants`); the `integration / report` status counts them all and passes when at least
`EVALS_MIN_PASS_RATE` of the cases pass (a repository variable, 1 by default). Remove a variant from
`variants.toml` once it is the default or abandoned.

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
| `AGENT_NAME` | `intake-agent` | Sent to the gateway as `x-valxel-agent` for budgets and usage |
| `AGENT_GATEWAY_URL`, `AGENT_GATEWAY_KEY` | | The product's LLM gateway and this agent's key |
| `AGENT_MODEL`, `AGENT_MAX_TOKENS` | `claude-sonnet-5-5`, `2048` | The gateway's model allowlist applies |
| `AGENT_MCP_SERVERS` | `[]` | JSON list of `{name, url, audience, scope}`: `audience` is the server's client in the realm |
| `AGENT_TOKEN_URL`, `AGENT_CLIENT_ID` | | The realm's token endpoint and this agent's client, for token exchange |
| `AGENT_CLIENT_KEY_FILE`, `AGENT_CLIENT_KEY_ID` | | `private_key_jwt` client authentication (preferred), or `AGENT_CLIENT_SECRET` |
| `AGENT_ALLOWED_TOOLS` | `[]` | fnmatch patterns on `<server>__<tool>`. Deny by default: list each tool (or `care__*`) the agent may use |
| `AGENT_TENANT_CLAIM` | `organization` | The token claim naming the caller's tenant: runs are visible to their requester and to reviewers of the same tenant |
| `AGENT_ALLOW_SELF_APPROVAL` | `false` | Whether the person who asked for a change may approve it themselves |
| `AGENT_MODEL_TIMEOUT_S` | `120` | One model call's timeout |
| `AGENT_MAX_STEPS` | `8` | |
| `AGENT_DATABASE_URL` | | Postgres; in memory if unset (development only: one replica, lost on restart) |
| `AGENT_DATABASE_POOL` | `10` | Connections per replica; each thread executing on the replica holds one more (up to 8), plus one for notifications. Keep replicas x (pool + 9) under the database's connection limit |
| `AGENT_CHAT_HISTORY_TURNS`, `AGENT_CHAT_MAX_CHARS` | `20`, `8000` | Exchanges the model sees; the longest message a person can send |
| `AGENT_PROGRESS_RETENTION_S` | `86400` | How long streamed progress is kept |
| `AGENT_OIDC_ISSUER`, `AGENT_OIDC_AUDIENCE`, `AGENT_OIDC_JWKS_URL` | | Required |
| `AGENT_RUN_SCOPE`, `AGENT_REVIEW_SCOPE` | `agent:run`, `agent:review` | |
| `AGENT_FLAGS_URL` | | The product's Flipt; unset, every feature flag reads as off |
| `AGENT_FLAGS_ENVIRONMENT`, `AGENT_FLAGS_NAMESPACE` | | Flipt environment (e.g. `production`) and the product's namespace (e.g. `emr`) |
| `AGENT_FLAGS_TOKEN` | | A Flipt client token, if the product's Flipt requires one |
| `AGENT_FLAGS_TIMEOUT_S`, `AGENT_FLAGS_CACHE_S` | `0.25`, `30` | A flag call waits at most this long; each answer (per flag, caller and attributes) is reused this long |
| `AGENT_PRODUCT`, `AGENT_ENVIRONMENT`, `AGENT_DATA_CLASS` | | Stamped on every audit record (`data_class`: `general`, `phi` or `payments`) |
| `AGENT_AUDIT_SINK` | `log` | `kafka` in deployed environments; `log` writes JSON lines locally |
| `AGENT_AUDIT_KAFKA_BOOTSTRAP`, `AGENT_AUDIT_TOPIC` | `audit.v1` | The product's Kafka and its audit topic |
| `AGENT_AUDIT_KAFKA_CONFIG` | `{}` | JSON of extra client settings (`security.protocol`, `sasl.*`). PHI and payments products must use `SSL` or `SASL_SSL`, or the agent refuses to start |
| `AGENT_AUDIT_REQUIRED` | `true` | A decision or change whose record isn't acknowledged by Kafka is not applied |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | | Standard OpenTelemetry settings; point the endpoint at the data plane's collector |

## Errors and traces

Every response carries a W3C `traceresponse` header with its trace id (`errors.py`), which the web
and mobile apps put on the errors they report, so the SDLC tool sees a failure people hit and an
error here as one incident. With `SENTRY_DSN` set, unhandled exceptions and `ERROR` logs go to
Sentry, grouped into issues, with the exception type, stack, endpoint path, a `trace_id` tag and a
`component` tag (`component.py`, which the SDLC tool sets to the component's name when it creates a
repo from this template, via `.valxel/identity`)
only: never exception messages, formatted messages, local variables, queries, headers, bodies,
users or breadcrumbs. Also `ENVIRONMENT` and `SENTRY_RELEASE`. Unset, Sentry is off; logs, traces
and metrics stay in OpenChoreo. For PHI products use a Sentry plan under a BAA, or self-hosted Sentry.

## Delivery

CI (`ci.yml`) runs `unit` and `evals`, dependency review and an image build; CodeQL comes from
the repo's default setup. These are the required checks the Valxel SDLC sets on repos with the
`agent` role. The SDLC deploys through OpenChoreo and then sends `valxel-deployed`, which runs
`post-deploy.yml`: the smoke check at every stage, and live evals in dev.

## Performance tests

When the Valxel SDLC labels a pull request `valxel:perf-smoke` (the graph, the tools, the API or
the runtime changes; or the reviewer saw a performance risk), `perf.yml` runs the recorded evals on
main's code and the PR's (`python -m evals.run --perf`): the most model turns a case takes, seconds
per case and tokens per case. `perf/compare.py` posts `perf-smoke / report`: failed when the PR
needs more turns, 20% more tokens than main, or misses the budget (8 turns, 60 s per run). Live
model latency shows in the post-deploy evals and the pilot's timings; an agent has no staging load
test (it would pay for model calls).

## How it runs in OpenChoreo

The agent is registered as the platform's `agent` component type, not a plain service. Its pods
hold no Kubernetes token, have a read-only file system (only `/tmp` is writable), run as a non-root
user with no capabilities and no privilege escalation, and reach only the product's LLM gateway,
MCP servers, database and event brokers. PHI and payments agents also run under gVisor. CI starts
the image the same way, so a change that writes elsewhere or needs a privilege fails here first.

The agent never runs model-written code itself. A design where it must goes through a separate
code-execution sandbox (planned), never this pod.
