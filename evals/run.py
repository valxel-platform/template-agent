"""Agent evaluations.

  python -m evals.run                     replay recorded model responses (CI, every PR: offline, free)
  python -m evals.run --live              the real model through the LLM gateway (post-deploy in dev)
  python -m evals.run --live --record     ... and save the responses as the new recordings

Each case runs against the in-memory world, with a scripted reviewer, and is graded on what the
agent did. Replay catches regressions in the graph, prompts' plumbing and graders; live runs catch
model and prompt regressions. Results go to JUnit for the "evals / report" status.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.model import GatewayModel, Model, ScriptedModel
from agent.settings import Settings
from agent.tools import McpToolHost

from .graders import grade
from .world import ALLOWED_TOOLS, World

HERE = Path(__file__).parent


class Recorder:
    def __init__(self, inner: Model) -> None:
        self.inner, self.responses = inner, []  # type: ignore[var-annotated]

    def respond(self, **kw: Any) -> dict[str, Any]:
        out = self.inner.respond(**kw)
        self.responses.append(out)
        return out


def load_cases(only: str | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    return [c for c in cases if not only or c["id"] == only]


def run_case(case: dict[str, Any], model: Model, max_steps: int = 8) -> tuple[dict[str, Any], World]:
    world = World()
    tools = McpToolHost({"care": world.server}, ALLOWED_TOOLS)
    graph = build_graph(model=model, tools=tools, product="the care team", checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": case["id"]}}
    graph.invoke({"run_id": case["id"], "task": case["task"], "max_steps": max_steps}, cfg)
    for _ in range(10):
        snap = graph.get_state(cfg)
        if not snap.interrupts:
            break
        waiting = snap.interrupts[0].value
        review = case.get("review", "approve") if waiting["type"] == "approval" else "stop"
        decision = {"decision": review} if isinstance(review, str) else dict(review)
        graph.invoke(Command(resume={**decision, "reviewer": "evals"}), cfg)
    v = graph.get_state(cfg).values
    return {
        "status": v.get("status"),
        "result": v.get("result", ""),
        "steps": v.get("steps", 0),
        "trace": v.get("trace", []),
    }, world


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--junit", default="reports/evals.xml")
    args = ap.parse_args(argv)

    suite = ET.Element("testsuite", name="evals")
    failed = 0
    for case in load_cases(args.case):
        cassette = HERE / "cassettes" / f"{case['id']}.json"
        if args.live:
            s = Settings()
            model: Any = Recorder(
                GatewayModel(
                    base_url=s.gateway_url,
                    api_key=s.gateway_key,
                    model=s.model,
                    max_tokens=s.max_tokens,
                    agent=f"{s.name}-evals",
                )
            )
        else:
            model = ScriptedModel(json.loads(cassette.read_text()))
        started = time.monotonic()
        try:
            run, world = run_case(case, model)
            fails = grade(case, run, world)
        except Exception as exc:  # noqa: BLE001  (a crash is a failed case, not a failed suite)
            fails = [f"{type(exc).__name__}: {exc}"]
        tc = ET.SubElement(
            suite, "testcase", classname="evals", name=case["id"], time=f"{time.monotonic() - started:.3f}"
        )
        if fails:
            failed += 1
            ET.SubElement(tc, "failure", message=fails[0]).text = "\n".join(fails)
        print(f"{'FAIL' if fails else 'pass'}  {case['id']}" + "".join(f"\n      {f}" for f in fails))
        if args.record and isinstance(model, Recorder) and not fails:
            cassette.write_text(json.dumps(model.responses, indent=1))
    suite.set("tests", str(len(suite)))
    suite.set("failures", str(failed))
    Path(args.junit).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(args.junit, encoding="utf-8", xml_declaration=True)
    print(f"{len(suite) - failed}/{len(suite)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
