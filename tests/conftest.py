from __future__ import annotations

from typing import Any

import pytest

from agent.tools import McpToolHost
from evals.world import ALLOWED_TOOLS, World


def tool_use(i: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": i, "name": name, "input": args}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def reply(*content: dict[str, Any]) -> dict[str, Any]:
    stop = "tool_use" if any(c["type"] == "tool_use" for c in content) else "end_turn"
    return {
        "role": "assistant",
        "content": list(content),
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


FIND = tool_use("tu_1", "care__find_patient", {"name": "Ana Lopez", "date_of_birth": "1984-03-02"})
CREATE = tool_use(
    "tu_2", "care__create_task", {"patient_id": "p-102", "title": "Echo", "priority": "routine"}
)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def tools(world: World) -> McpToolHost:
    return McpToolHost({"care": world.server}, ALLOWED_TOOLS)
