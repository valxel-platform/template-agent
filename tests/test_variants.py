"""Prompt and model variants, rolled out through Flipt (variants.py)."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider
from openfeature.provider.no_op_provider import NoOpProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent import flags, telemetry, variants
from agent.graph import build_graph
from agent.model import ScriptedModel
from agent.tools import McpToolHost

from .conftest import reply, text

KNOWN = {
    "default": variants.Variant("default"),
    "small": variants.Variant("small", model="claude-haiku-5-5"),
}


@pytest.fixture(autouse=True)
def no_flags() -> Any:
    yield
    flags.use_provider(NoOpProvider())


def rollout(**by_user: str) -> None:
    """Flipt's answer for `agent.variant` per user (the targeting key)."""

    class Rules(InMemoryProvider):
        def resolve_string_details(self, flag_key, default_value, evaluation_context=None):  # type: ignore[no-untyped-def]
            from openfeature.flag_evaluation import FlagResolutionDetails

            user = evaluation_context.targeting_key if evaluation_context else None
            return FlagResolutionDetails(value=by_user.get(user or "", default_value))

    flags.use_provider(Rules({}))


def test_the_shipped_variants_load_and_bad_ones_dont() -> None:
    assert variants.load()["default"] == variants.Variant("default")
    with pytest.raises(ValueError, match=r"needs a \[default\]"):
        variants.load('[small]\nmodel = "x"\n')
    with pytest.raises(ValueError, match=r"prompts/missing\.md"):
        variants.load('[default]\nsystem = "missing"\n')


def test_flipt_picks_a_variant_per_person_and_unknown_ones_fall_back() -> None:
    rollout(ana="small", bob="gone")
    assert variants.pick("agent", KNOWN, user="ana") == "small"
    assert variants.pick("agent", KNOWN, user="bob") == "default"  # not reviewed: never runs
    assert variants.pick("agent", KNOWN, user="cy") == "default"
    flags.use_provider(NoOpProvider())
    assert variants.pick("agent", KNOWN, user="ana") == "default"  # no Flipt: default
    flags.use_provider(
        InMemoryProvider({"agent.variant": InMemoryFlag("small", {"small": "small"})}), cache_s=30
    )
    assert flags.variant("agent.variant", "default", user="z") == "small"
    flags.use_provider(NoOpProvider(), cache_s=30)
    assert flags.variant("agent.variant", "default", user="z") == "default"


def run(tools: McpToolHost, model: ScriptedModel, *, user: str, variant: str | None = None) -> dict[str, Any]:
    graph = build_graph(
        model=model,
        tools=tools,
        product="EMR",
        checkpointer=InMemorySaver(),
        agent="agent",
        variants=KNOWN,
        variant=variant,
    )
    cfg = {"configurable": {"thread_id": f"r-{user}"}}
    graph.invoke(
        {
            "run_id": f"r-{user}",
            "task": {"type": "referral", "text": "hi", "requested_by": user},
            "max_steps": 4,
        },
        cfg,
    )
    return dict(graph.get_state(cfg).values)


def test_a_run_keeps_its_variant_and_every_call_says_which(tools: McpToolHost) -> None:
    exporter = InMemorySpanExporter()
    telemetry.setup("agent", exporter=exporter)
    rollout(ana="small")
    model = ScriptedModel([reply(text("Done."))])
    v = run(tools, model, user="ana")
    assert v["variant"] == "small" and v["trace"][0]["variant"] == "small"
    (call,) = model.calls
    assert call["model"] == "claude-haiku-5-5" and call["labels"] == {"run": "r-ana", "variant": "small"}
    chat = [s for s in exporter.get_finished_spans() if s.name.startswith("chat ")][-1]
    assert chat.name == "chat claude-haiku-5-5" and chat.attributes["valxel.variant"] == "small"
    other = ScriptedModel([reply(text("Done."))])
    assert run(tools, other, user="bob")["variant"] == "default"
    assert other.calls[0]["model"] is None and other.calls[0]["labels"]["variant"] == "default"


def test_the_evals_can_force_a_variant(tools: McpToolHost) -> None:
    model = ScriptedModel([reply(text("Done."))])
    assert run(tools, model, user="ana", variant="small")["variant"] == "small"
    with pytest.raises(ValueError, match="no variant 'nope'"):
        run(tools, ScriptedModel([]), user="ana", variant="nope")


def test_the_eval_runner_takes_a_variant_and_reports_a_pass_rate(tmp_path: Any, monkeypatch: Any) -> None:
    import subprocess
    import sys

    from evals import run as evals_run

    out = tmp_path / "evals.xml"
    assert evals_run.main(["--junit", str(out)]) == 0 and "<testsuite" in out.read_text()
    with pytest.raises(SystemExit):
        evals_run.main(
            ["--variant", "small", "--junit", str(out)]
        )  # replay only has the default's recordings
    junit = tmp_path / "r.xml"
    junit.write_text(
        '<testsuites><testsuite name="evals" tests="20" failures="1"/>'
        '<testsuite name="evals[small]" tests="20" failures="0"/></testsuites>'
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def report(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - the repo's own script, with fixed arguments
            [
                sys.executable,
                ".github/scripts/report.py",
                "--context",
                "integration / report",
                "--junit",
                str(junit),
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    strict = report()
    assert strict.returncode == 1 and "39 passed, 1 failed" in strict.stdout
    lenient = report("--min-pass-rate", "0.95")
    assert lenient.returncode == 0 and "Pass rate: 97.5% (at least 95%)" in lenient.stdout
