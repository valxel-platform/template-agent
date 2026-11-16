"""The recorded evals, also run by the evals CI job with JUnit output."""

from __future__ import annotations

import json

import pytest

from agent.model import ScriptedModel
from evals.graders import grade
from evals.run import HERE, load_cases, run_case


@pytest.mark.evals
@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["id"])
def test_case(case: dict) -> None:  # type: ignore[type-arg]
    responses = json.loads((HERE / "cassettes" / f"{case['id']}.json").read_text())
    run, world = run_case(case, ScriptedModel(responses))
    assert grade(case, run, world) == []


def test_answers_from_a_knowledge_search_cite_their_passages() -> None:
    from evals.world import World

    run = {
        "status": "done",
        "trace": [],
        "steps": 2,
        "result": "The dose was cut to 4 mg [notes:n-1001]; the INR is back in range [messages:m-201].",
    }
    case = {"expect": {"cites": ["notes:n-1001", "messages:m-201"], "never_cites": ["notes:n-1005"]}}
    assert grade(case, run, World()) == []
    missing = {"expect": {"cites": ["policies:pol-1"]}}
    assert grade(missing, run, World()) == ["result doesn't cite [policies:pol-1]"]
    wrong = {"expect": {"never_cites": ["notes:n-1001"]}}
    assert grade(wrong, run, World()) == ["result cites [notes:n-1001], which it must not"]
