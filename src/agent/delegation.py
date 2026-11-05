"""The agent acts for a person, never instead of one (constitution VX-13, VX-17).

Every call to an MCP server carries a token that Keycloak issued for that server only, on behalf of
the person behind the step:

- the steps of a run use the token of the person who started it;
- an approved change runs with the token of the person who approved it, so a change never exceeds
  what its approver may do, and the audit trail names them.

The agent exchanges that person's token (RFC 8693 token exchange) for one whose audience is the MCP
server and whose scopes are only the server's. The subject stays the person; the agent becomes the
authorized party (`azp`). People's tokens are held in memory only, never in the run's checkpoint:
after a restart, a run that needs to call a tool again waits for a person (a new approval brings a
fresh token).

Listing a server's tools (tool metadata, no records) uses the agent's own client credentials.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx

_subject: ContextVar[str | None] = ContextVar("subject_token", default=None)

EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN = "urn:ietf:params:oauth:token-type:access_token"  # noqa: S105 (a token type URN, not a secret)
EARLY_S = 30


class DelegationError(RuntimeError):
    """No token for a step: the step must not run. Messages never contain tokens."""


@contextlib.contextmanager
def acting_for(token: str | None) -> Iterator[None]:
    """Run the enclosed steps on behalf of the person whose token this is."""
    reset = _subject.set(token)
    try:
        yield
    finally:
        _subject.reset(reset)


def current() -> str | None:
    return _subject.get()


class TokenExchange:
    """Tokens for MCP servers from the product's Keycloak realm, cached per person and server."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        *,
        client_secret: str = "",
        client_key_file: str = "",
        client_key_id: str = "",
        http: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not token_url or not client_id:
            raise ValueError("Set AGENT_TOKEN_URL and AGENT_CLIENT_ID to call MCP servers on people's behalf")
        if not client_secret and not client_key_file:
            raise ValueError("Set AGENT_CLIENT_KEY_FILE (private_key_jwt) or AGENT_CLIENT_SECRET")
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_key_file = client_key_file
        self.client_key_id = client_key_id
        self.http = http or httpx.Client(timeout=5.0)
        self.clock = clock
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def for_person(self, subject_token: str, audience: str, scope: str = "") -> str:
        """A token for `audience` on behalf of the person whose token `subject_token` is."""
        key = f"{audience}:{scope}:{hashlib.sha256(subject_token.encode()).hexdigest()}"
        form = {
            "grant_type": EXCHANGE,
            "subject_token": subject_token,
            "subject_token_type": ACCESS_TOKEN,
            "requested_token_type": ACCESS_TOKEN,
            "audience": audience,
        }
        if scope:
            form["scope"] = scope
        return self._cached(key, form, f"token exchange for {audience}")

    def for_agent(self, audience: str, scope: str = "") -> str:
        """The agent's own token (client credentials), for tool metadata only."""
        form = {"grant_type": "client_credentials", "audience": audience}
        if scope:
            form["scope"] = scope
        return self._cached(f"self:{audience}:{scope}", form, f"client credentials for {audience}")

    def _cached(self, key: str, form: dict[str, str], what: str) -> str:
        now = self.clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now < hit[1]:
                return hit[0]
        token, expires_in = self._request(form, what)
        with self._lock:
            if len(self._cache) > 10_000:
                self._cache = {k: v for k, v in self._cache.items() if now < v[1]}
            self._cache[key] = (token, now + max(0, expires_in - EARLY_S))
        return token

    def _request(self, form: dict[str, str], what: str) -> tuple[str, int]:
        headers = {"Accept": "application/json"}
        if self.client_key_file:
            form = {
                **form,
                "client_id": self.client_id,
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": self._assertion(),
            }
        else:
            basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {basic}"
        try:
            res = self.http.post(self.token_url, data=form, headers=headers)
        except httpx.HTTPError as exc:
            raise DelegationError(f"{what} failed: {type(exc).__name__}") from exc
        if res.status_code != 200:
            try:
                code = res.json().get("error", "unknown")
            except ValueError:
                code = "unknown"
            # The error code only: never the body, which may echo the token.
            raise DelegationError(f"{what} was refused ({res.status_code}, {code})")
        body = res.json()
        token = body.get("access_token") or ""
        if not token:
            raise DelegationError(f"{what} returned no access_token")
        return token, int(body.get("expires_in") or 60)

    def _assertion(self) -> str:
        import jwt

        now = int(self.clock())
        claims = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.token_url,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + 60,
        }
        headers = {"kid": self.client_key_id} if self.client_key_id else None
        key = Path(self.client_key_file).read_text()
        return jwt.encode(claims, key, algorithm="RS256", headers=headers)


def from_settings(settings: Any) -> TokenExchange | None:
    if not settings.token_url:
        return None
    return TokenExchange(
        settings.token_url,
        settings.client_id,
        client_secret=settings.client_secret,
        client_key_file=settings.client_key_file,
        client_key_id=settings.client_key_id,
    )


def unverified_expiry(token: str) -> float | None:
    """When a token we were given (and verified) expires: to drop it from memory in time."""
    import jwt

    try:
        exp = jwt.decode(token, options={"verify_signature": False}).get("exp")
    except jwt.PyJWTError:
        return None
    return float(exp) if exp else None


__all__: list[Any] = ["DelegationError", "TokenExchange", "acting_for", "current", "from_settings"]
