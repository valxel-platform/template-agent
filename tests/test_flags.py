from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider
from openfeature.provider.no_op_provider import NoOpProvider

from agent import flags
from agent.settings import Settings

KEY = "emr.emr-feat-3.referral-triage"


class Flipt(BaseHTTPRequestHandler):
    """Flipt's OFREP endpoint: POST /ofrep/v1/evaluate/flags/{key}."""

    flags: ClassVar[dict[str, bool]] = {}
    seen: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        key = self.path.rsplit("/", 1)[-1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
        Flipt.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if key not in Flipt.flags:
            out, code = {"key": key, "errorCode": "FLAG_NOT_FOUND"}, 404
        else:
            on = Flipt.flags[key]
            out, code = {"key": key, "value": on, "variant": str(on).lower(), "reason": "DEFAULT"}, 200
        raw = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def flipt() -> Iterator[str]:
    Flipt.flags, Flipt.seen = {KEY: True}, []
    server = HTTPServer(("127.0.0.1", 0), Flipt)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    flags.use_provider(NoOpProvider())


def test_without_a_flag_server_flags_read_as_their_default() -> None:
    flags.use_provider(NoOpProvider())
    flags.configure(Settings(flags_url=""))
    assert flags.enabled(KEY) is False
    assert flags.enabled(KEY, default=True) is True


def test_flags_come_from_the_products_flipt(flipt: str) -> None:
    flags.configure(
        Settings(flags_url=flipt, flags_environment="production", flags_namespace="emr", flags_token="t")
    )
    assert flags.enabled(KEY, user="nurse-1", clinic="north") is True
    req = Flipt.seen[-1]
    assert req["path"] == f"/ofrep/v1/evaluate/flags/{KEY}"
    assert req["headers"]["x-flipt-environment"] == "production"
    assert req["headers"]["x-flipt-namespace"] == "emr"
    assert req["headers"]["Authorization"] == "Bearer t"
    assert req["body"]["context"] == {"targetingKey": "nurse-1", "clinic": "north"}
    assert flags.enabled("emr.unknown") is False  # unknown flags fall back to the default


def test_a_test_can_set_flags_in_memory() -> None:
    flags.use_provider(InMemoryProvider({KEY: InMemoryFlag("on", {"on": True, "off": False})}))
    assert flags.enabled(KEY) is True
    flags.use_provider(NoOpProvider())
