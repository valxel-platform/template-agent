"""Metrics on the management port (8081), apart from the API, as in the Java templates."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from agent import management


def get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - a local test server
            return r.status, r.headers.get("Content-Type", ""), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, "", ""


def test_metrics_are_served_on_their_own_port_and_nothing_else_is() -> None:
    server = management.serve(
        lambda: (b"x_total 1.0\n", "text/plain; version=0.0.4"), host="127.0.0.1", port=0
    )
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert get(base + "/metrics") == (200, "text/plain; version=0.0.4", "x_total 1.0\n")
        assert get(base + "/metrics?x=1")[0] == 200
        for path in ("/", "/healthz", "/metricsx", "/api"):
            assert get(base + path)[0] == 404, path
    finally:
        server.shutdown()


def test_the_port_is_8081_unless_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANAGEMENT_PORT", raising=False)
    assert management.default_port() == 8081
    monkeypatch.setenv("MANAGEMENT_PORT", "9464")
    assert management.default_port() == 9464
