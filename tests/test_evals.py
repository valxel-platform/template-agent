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
