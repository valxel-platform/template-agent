"""agent.yaml, which the platform turns into AGENT_ALLOWED_TOOLS and AGENT_READ_ONLY_TOOLS on deploy."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

TOOL = re.compile(r"^[a-z0-9][a-z0-9_-]*__[a-z0-9_][a-z0-9_-]*$")


def test_read_tools_are_named_exactly_and_never_need_approval_too() -> None:
    tools = yaml.safe_load((Path(__file__).resolve().parents[1] / "agent.yaml").read_text())["tools"]
    assert all(TOOL.match(t) for t in tools["read_only"])  # no wildcard runs without a person
    assert not set(tools["read_only"]) & set(tools["approval"])
