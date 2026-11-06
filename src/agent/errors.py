"""Error reports to Sentry, and trace ids on responses.

Every HTTP response carries a W3C `traceresponse` header (`00-<trace id>-<span id>-01`). The web
and mobile apps put its trace id on the errors they report, so the SDLC tool can tell that a
failure people saw and an error here are one incident.

With SENTRY_DSN set, unhandled exceptions and ERROR logs go to Sentry, grouped into issues, and
Sentry's issue alerts reach the SDLC tool as incident signals. An event keeps the exception type,
its stack, the endpoint's path and method, the log message's pattern and a `trace_id` tag; never
exception messages, formatted messages or their arguments, local variables, query strings,
headers, cookies, bodies, users or breadcrumbs: any of them can carry PHI or a token. Unset,
Sentry is off. Also ENVIRONMENT and SENTRY_RELEASE (the image tag). Logs, traces and metrics stay
in OpenChoreo.
"""

from __future__ import annotations

import os
from typing import Any, cast

from opentelemetry import trace

HEADER = b"traceresponse"


def trace_response() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-01"


class TraceResponseMiddleware:
    """ASGI middleware: names the trace on every HTTP response. Add it before the app is
    instrumented, so the request's span is current when the response starts."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def with_header(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                value = trace_response()
                headers = list(message.get("headers") or [])
                if value and not any(k.lower() == HEADER for k, _ in headers):
                    headers.append((HEADER, value.encode()))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_header)


def _path_only(url: Any) -> Any:
    if not isinstance(url, str):
        return url
    for sep in ("?", "#"):
        url = url.split(sep, 1)[0]
    return url


def scrub(event: dict[str, Any], hint: Any = None) -> dict[str, Any]:
    """What a Sentry event may carry (before_send)."""
    for key in ("user", "extra", "breadcrumbs", "server_name"):
        event.pop(key, None)
    request = event.get("request")
    if isinstance(request, dict):
        event["request"] = {
            k: v for k, v in (("method", request.get("method")), ("url", _path_only(request.get("url")))) if v
        }
    for ex in (event.get("exception") or {}).get("values") or []:
        ex.pop("value", None)
        for frame in (ex.get("stacktrace") or {}).get("frames") or []:
            frame.pop("vars", None)
    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        event["logentry"] = {"message": logentry.get("message")}
    event.pop("message", None)
    value = trace_response()
    if value:
        event.setdefault("tags", {})["trace_id"] = value[3:35]
    return event


def start_errors(environ: dict[str, str] | None = None) -> bool:
    """Start Sentry when SENTRY_DSN is set; returns whether it did."""
    env = os.environ if environ is None else environ
    dsn = (env.get("SENTRY_DSN") or "").strip()
    if not dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        environment=env.get("ENVIRONMENT") or "local",
        release=env.get("SENTRY_RELEASE") or None,
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        # Traces go through OpenTelemetry (telemetry.py); Sentry only takes errors.
        traces_sample_rate=None,
        before_send=cast(Any, scrub),
        before_breadcrumb=lambda crumb, hint: None,
    )
    return True
