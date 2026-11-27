"""The management port, as in the Java templates (Spring's actuator on MANAGEMENT_PORT): /metrics
for the data plane's metrics collector, on a port of its own (default 8081), apart from the API.
The mesh lets the collector reach this port on every workload and no other, so the API port serves
no metrics, and the metrics need none of the API's authentication. Kubernetes' probes stay on the
API port (the kubelet reaches it outside the mesh)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# What GET /metrics answers: the exposition and its content type (prometheus_client's).
Body = Callable[[], tuple[bytes, str]]


def default_port() -> int:
    """MANAGEMENT_PORT, 8081 unless set: where the component types point the metrics collector."""
    return int(os.environ.get("MANAGEMENT_PORT", "8081"))


def serve(body: Body, *, host: str = "0.0.0.0", port: int | None = None) -> ThreadingHTTPServer:  # noqa: S104
    """Serves GET /metrics from `body` on a daemon thread, and nothing else (404). Returns the server:
    its `server_address` (port 0 picks a free one) and `shutdown()`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # http.server's name for it
            if self.path.split("?", 1)[0] != "/metrics":
                self.send_error(404)
                return
            data, content_type = body()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass  # a scrape every 30 seconds isn't news

    server = ThreadingHTTPServer((host, default_port() if port is None else port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="management", daemon=True).start()
    return server
