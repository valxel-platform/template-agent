"""The template conformance suite: the built image, run as the platform runs it, does what the
template contract says every repo of its role does (valxel-platform/agentic-sdlc,
config/template-contract.yaml; tracker T-14). The platform manages this file; don't edit it here.

    python .github/scripts/conformance.py --image ci          the image CI just built
    python .github/scripts/conformance.py --cmd "uv run app"  the app as a process, for local work

The repo's own `conformance.yaml` says how to start it and what to do to it; this script stands up
the services around it on 127.0.0.1 (the image runs with --network host), runs the steps, then checks
what it was asked to:

  audit     an audit record, one JSON line on stdout, that meets the shared schema
            (contracts/audit/valxel.audit.v1.schema.json)
  tracing   spans reach an OpenTelemetry collector (OTLP over HTTP)
  events    /metrics has valxel_events_total
  flags     a flag is evaluated through OFREP (the product's Flipt)

Around it: an OTLP collector, an OFREP flag service, an OpenID provider whose tokens the app accepts
({token: {claims}} on a step), and an Anthropic-compatible model API answering every call with
`model_reply`. `ready` and `metrics` (a path, or a whole URL for a management port) say where the
app answers when it's up and where its metrics are. In conformance.yaml, env values may name them: {otlp}, {ofrep}, {issuer}, {jwks},
{model}, {audience}, and in later steps any value an earlier step saved (save: {name: "header:<name>"
| "json:<field>"}).

A repo whose app needs a database or Kafka to start says so (`postgres: <image>`, `kafka: <image>`):
with --image they run as containers too, and {postgres} and {kafka} name them; with --cmd, pass
--postgres and --kafka instead.

Needs pyyaml, jsonschema and pyjwt[crypto]; exits 1 naming every check that failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

AUDIENCE = "conformance"


# --------------------------------------------------------------------------- the services around it

class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes]] = []
        self.lock = threading.Lock()

    def add(self, method: str, path: str, body: bytes) -> None:
        with self.lock:
            self.calls.append((method, path, body))

    def count(self, prefix: str) -> int:
        with self.lock:
            return sum(1 for _, p, b in self.calls if p.startswith(prefix))


def _serve(handler_cls: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _handler(rec: Recorder, answer):
    class H(BaseHTTPRequestHandler):
        def _go(self) -> None:
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n) if n else b""
            rec.add(self.command, self.path, body)
            status, ctype, out = answer(self.command, self.path, body)
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        do_GET = do_POST = do_PUT = _go

        def log_message(self, *a: Any) -> None:
            pass

    return H


class Services:
    def __init__(self, model_reply: str, audience: str = AUDIENCE) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = uuid.uuid4().hex[:8]
        self.otlp, self.ofrep, self.oidc, self.model = Recorder(), Recorder(), Recorder(), Recorder()
        self._servers = []
        for name, rec, answer in (("otlp", self.otlp, self._otlp), ("ofrep", self.ofrep, self._ofrep),
                                  ("oidc", self.oidc, self._oidc), ("model", self.model, self._model)):
            server, url = _serve(_handler(rec, answer))
            self._servers.append(server)
            setattr(self, f"{name}_url", url)
        self.model_reply, self.audience = model_reply, audience

    def stop(self) -> None:
        for s in self._servers:
            s.shutdown()

    def names(self) -> dict[str, str]:
        return {"otlp": self.otlp_url, "ofrep": self.ofrep_url, "issuer": self.oidc_url,
                "jwks": f"{self.oidc_url}/jwks", "model": self.model_url, "audience": self.audience}

    @staticmethod
    def _otlp(method: str, path: str, body: bytes):
        return 200, "application/json", b"{}"

    @staticmethod
    def _ofrep(method: str, path: str, body: bytes):
        key = path.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
        if "/ofrep/v1/evaluate/flags" in path and not path.endswith("/flags"):
            out = {"key": key, "value": True, "reason": "STATIC", "variant": "on", "metadata": {}}
        else:
            out = {"flags": []}
        return 200, "application/json", json.dumps(out).encode()

    def _oidc(self, method: str, path: str, body: bytes):
        import jwt

        if path.startswith("/jwks") or path.endswith("/certs"):
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
            return 200, "application/json", json.dumps({"keys": [{**jwk, "kid": self.kid, "use": "sig",
                                                                   "alg": "RS256"}]}).encode()
        if path.endswith("/.well-known/openid-configuration"):
            return 200, "application/json", json.dumps({
                "issuer": self.oidc_url, "jwks_uri": f"{self.oidc_url}/jwks",
                "authorization_endpoint": f"{self.oidc_url}/auth", "token_endpoint": f"{self.oidc_url}/token",
                "userinfo_endpoint": f"{self.oidc_url}/userinfo", "end_session_endpoint": f"{self.oidc_url}/logout",
                "response_types_supported": ["code"], "subject_types_supported": ["public"],
                "id_token_signing_alg_values_supported": ["RS256"]}).encode()
        return 404, "application/json", b"{}"

    def _model(self, method: str, path: str, body: bytes):
        req = json.loads(body or b"{}")
        if path.endswith("/messages"):
            return 200, "application/json", json.dumps({
                "id": "msg_conformance", "type": "message", "role": "assistant", "model": req.get("model", "m"),
                "content": [{"type": "text", "text": self.model_reply}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}).encode()
        if path.endswith("/chat/completions"):
            return 200, "application/json", json.dumps({
                "id": "c", "object": "chat.completion", "model": req.get("model", "m"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": self.model_reply}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()
        if path.endswith("/embeddings"):
            n = len(req.get("input") or [1]) if isinstance(req.get("input"), list) else 1
            return 200, "application/json", json.dumps({"object": "list", "model": req.get("model", "m"), "data": [
                {"object": "embedding", "index": i, "embedding": [0.01] * int(req.get("dimensions") or 8)} for i in range(n)],
                "usage": {"prompt_tokens": 1, "total_tokens": 1}}).encode()
        return 404, "application/json", b"{}"

    def token(self, claims: dict[str, Any]) -> str:
        import jwt

        now = int(time.time())
        payload = {"iss": self.oidc_url, "aud": self.audience, "iat": now, "exp": now + 600, "sub": "conformance-user",
                   **claims}
        return jwt.encode(payload, self.key, algorithm="RS256", headers={"kid": self.kid})


# --------------------------------------------------------------------------- the app

def _fill(value: Any, names: dict[str, str]) -> Any:
    if isinstance(value, str):
        for k, v in names.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, dict):
        return {k: _fill(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, names) for v in value]
    return value


class App:
    def __init__(self, spec: dict, env: dict[str, str], *, image: str = "", cmd: str = "") -> None:
        self.spec, self.env, self.image, self.cmd = spec, env, image, cmd
        self.log = Path(tempfile.mkstemp(prefix="conformance-", suffix=".log")[1])
        self.name = f"conformance-{uuid.uuid4().hex[:8]}"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.image:
            args = ["docker", "run", "-d", "--name", self.name, "--network", "host", "--read-only",
                    "--tmpfs", "/tmp:rw,size=64m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
            for k, v in self.env.items():
                args += ["-e", f"{k}={v}"]
            args += [self.image, *self.spec.get("args", [])]
            subprocess.run(args, check=True, capture_output=True)
        else:
            out = self.log.open("wb")
            self.proc = subprocess.Popen(shlex.split(self.cmd), env={**os.environ, **self.env}, stdout=out,
                                         stderr=subprocess.STDOUT)

    def logs(self) -> str:
        if self.image:
            r = subprocess.run(["docker", "logs", self.name], capture_output=True, text=True)
            return r.stdout + r.stderr
        return self.log.read_text(errors="replace")

    def running(self) -> bool:
        if self.image:
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name], capture_output=True, text=True)
            return r.stdout.strip() == "true"
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.image:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        elif self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _http(method: str, url: str, body: Any = None, headers: dict[str, str] | None = None,
          ) -> tuple[int, str, dict[str, str]]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        **({"content-type": "application/json"} if data else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), {k.lower(): v for k, v in exc.headers.items()}
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        return 0, str(exc), {}


def _saved(how: str, text: str, headers: dict[str, str]) -> str:
    """`header:<name>`, or `json:<field>` of the body (an SSE body's last data line counts)."""
    kind, _, name = how.partition(":")
    if kind == "header":
        return headers.get(name.lower(), "")
    for chunk in [text, *[ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")][::-1]]:
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        for part in name.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        if obj is not None:
            return str(obj)
    return ""


# --------------------------------------------------------------------------- checks

def audit_records(logs: str) -> list[dict]:
    """Every valxel.audit.v1 event in the logs: a JSON line, or a JSON log line whose message is one."""
    out = []
    for line in logs.splitlines():
        line = line.strip()
        if "valxel.audit.v1" not in line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        for cand in (obj, obj.get("message"), obj.get("msg"), obj.get("event")):
            if isinstance(cand, str) and cand.startswith("{"):
                try:
                    cand = json.loads(cand)
                except ValueError:
                    continue
            if isinstance(cand, dict) and cand.get("type") == "valxel.audit.v1":
                out.append(cand)
                break
    return out


def check(name: str, svc: Services, app: App, base: str, problems: list[str]) -> str:
    if name == "audit":
        from jsonschema import Draft202012Validator

        schema = json.loads(Path("contracts/audit/valxel.audit.v1.schema.json").read_text(encoding="utf-8"))
        v = Draft202012Validator(schema)
        records = audit_records(app.logs())
        if not records:
            problems.append("audit: no valxel.audit.v1 record in the app's output")
            return "no record"
        bad = [f"{'/'.join(map(str, e.absolute_path)) or 'event'}: {e.message}" for r in records for e in v.iter_errors(r)]
        if bad:
            problems.append("audit: a record breaks the schema: " + "; ".join(bad[:3]))
        return f"{len(records)} records" + (f", {len(bad)} problems" if bad else ", valid")
    if name == "tracing":
        n = svc.otlp.count("/v1/traces")
        if not n:
            problems.append("tracing: no spans reached the collector")
        return f"{n} exports"
    if name == "events":
        url = app.spec.get("metrics", "/metrics")
        status, text, _ = _http("GET", url if url.startswith("http") else base + url)
        if status != 200 or "valxel_events_total" not in text:
            problems.append(f"events: /metrics ({status}) has no valxel_events_total")
            return "missing"
        return "valxel_events_total present"
    if name == "flags":
        n = svc.ofrep.count("/ofrep/v1/evaluate")
        if not n:
            problems.append("flags: no flag was evaluated through OFREP")
        return f"{n} evaluations"
    problems.append(f"{name}: not a check this suite knows")
    return "unknown"


PG_PORT, KAFKA_PORT = 55432, 59092


def _backing(spec: dict, *, image: str, postgres: str, kafka: str) -> tuple[dict[str, str], list[str]]:
    """{postgres} and {kafka}: what --cmd was given, or containers next to the image's."""
    names, started = {}, []
    if spec.get("postgres"):
        if postgres:
            names["postgres"] = postgres
        elif image:
            name = f"conformance-pg-{uuid.uuid4().hex[:6]}"
            subprocess.run(["docker", "run", "-d", "--name", name, "--network", "host", "-e", "POSTGRES_PASSWORD=conformance",
                            "-e", f"PGPORT={PG_PORT}", spec["postgres"]], check=True, capture_output=True)
            started.append(name)
            names["postgres"] = f"postgresql://postgres:conformance@127.0.0.1:{PG_PORT}/postgres"
            for _ in range(60):
                if subprocess.run(["docker", "exec", name, "pg_isready", "-p", str(PG_PORT)], capture_output=True).returncode == 0:
                    break
                time.sleep(1)
    if spec.get("kafka"):
        if kafka:
            names["kafka"] = kafka
        elif image:
            name = f"conformance-kafka-{uuid.uuid4().hex[:6]}"
            subprocess.run(["docker", "run", "-d", "--name", name, "--network", "host",
                            "-e", f"KAFKA_LISTENERS=PLAINTEXT://127.0.0.1:{KAFKA_PORT},CONTROLLER://127.0.0.1:59093",
                            "-e", f"KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:{KAFKA_PORT}",
                            "-e", "KAFKA_CONTROLLER_QUORUM_VOTERS=1@127.0.0.1:59093", "-e", "KAFKA_NODE_ID=1",
                            "-e", "KAFKA_PROCESS_ROLES=broker,controller", "-e", "KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER",
                            "-e", "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
                            "-e", "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1", "-e", "KAFKA_AUTO_CREATE_TOPICS_ENABLE=true",
                            spec["kafka"]], check=True, capture_output=True)
            started.append(name)
            names["kafka"] = f"127.0.0.1:{KAFKA_PORT}"
            time.sleep(10)
    return names, started


def run(spec: dict, *, image: str = "", cmd: str = "", postgres: str = "", kafka: str = "") -> list[str]:
    svc = Services(spec.get("model_reply", "Done."), spec.get("audience", AUDIENCE))
    backing, containers = _backing(spec, image=image, postgres=postgres, kafka=kafka)
    names = {**svc.names(), **backing}
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": svc.otlp_url, "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
           "OTEL_TRACES_SAMPLER": "always_on", "OTEL_BSP_SCHEDULE_DELAY": "200",
           **{k: str(v) for k, v in _fill(spec.get("env") or {}, names).items()}}
    base = f"http://127.0.0.1:{spec.get('port', 8080)}"
    app = App(spec, env, image=image, cmd=cmd)
    problems: list[str] = []
    app.start()
    try:
        ready = spec.get("ready", "/healthz")
        ready_url = ready if ready.startswith("http") else base + ready
        for _ in range(int(spec.get("start_s", 90))):
            if _http("GET", ready_url)[0] == 200:
                break
            if not app.running():
                break
            time.sleep(1)
        else:
            problems.append(f"the app didn't answer {ready}")
        if not problems:
            saved: dict[str, str] = {}
            for i, raw_step in enumerate(spec.get("steps") or []):
                step = _fill(_fill(raw_step, names), saved)
                headers = dict(step.get("headers") or {})
                if "token" in step:
                    headers["authorization"] = "Bearer " + svc.token(step["token"] or {})
                expect = step.get("expect", [200, 201, 202, 204])
                deadline = time.time() + float(step.get("within_s", 0))
                while True:
                    status, text, got = _http(step.get("method", "GET"), base + step["path"], step.get("json"), headers)
                    ok = status in (expect if isinstance(expect, list) else [expect]) and \
                        all(s in text for s in step.get("contains") or [])
                    if ok or time.time() >= deadline:
                        break
                    time.sleep(0.5)
                print(f"step {i + 1}: {step.get('method', 'GET')} {step['path']} -> {status}")
                for name, how in (step.get("save") or {}).items():
                    saved[name] = _saved(how, text, got)
                if not ok:
                    problems.append(f"step {i + 1} ({step.get('method', 'GET')} {step['path']}) answered {status}: {text[:300]}")
            time.sleep(float(spec.get("settle_s", 3)))  # batched spans and logs go out
            for name in spec.get("checks") or []:
                print(f"{name}: {check(name, svc, app, base, problems)}")
    finally:
        logs = app.logs()
        app.stop()
        svc.stop()
        for name in containers:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if problems:
        print("\n--- the app's output (last 80 lines) ---\n" + "\n".join(logs.splitlines()[-80:]))
    return problems


def main(argv: list[str] | None = None) -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", default="")
    p.add_argument("--cmd", default="")
    p.add_argument("--spec", default="conformance.yaml")
    p.add_argument("--postgres", default="", help="with --cmd: a database URL for {postgres}")
    p.add_argument("--kafka", default="", help="with --cmd: Kafka's bootstrap servers for {kafka}")
    a = p.parse_args(argv)
    if bool(a.image) == bool(a.cmd):
        p.error("give --image or --cmd")
    spec = yaml.safe_load(Path(a.spec).read_text(encoding="utf-8"))
    problems = run(spec, image=a.image, cmd=a.cmd, postgres=a.postgres, kafka=a.kafka)
    for x in problems:
        print(f"FAILED {x}")
    print("conformance: " + ("passed" if not problems else f"{len(problems)} problems"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
