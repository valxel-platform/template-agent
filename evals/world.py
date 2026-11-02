"""A small, deterministic stand-in for the product's MCP servers, used by tests and evals.
Replace it with your product's tools; keep it in memory and seeded so runs are repeatable."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True)

PATIENTS = {
    ("ana lopez", "1984-03-02"): {
        "patient_id": "p-102",
        "name": "Ana Lopez",
        "mrn": "MRN-55821",
        "phone": "555-0142",
    },
}


class World:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.server = self._build()

    def _build(self) -> MCPServer:
        server = MCPServer("care")
        world = self

        @server.tool(annotations=READ)
        def find_patient(name: str, date_of_birth: str) -> dict[str, Any]:
            """Find a patient by full name and date of birth (YYYY-MM-DD). Returns the patient id."""
            world.calls.append(("find_patient", {"name": name, "date_of_birth": date_of_birth}))
            p = PATIENTS.get((name.strip().lower(), date_of_birth))
            if not p:
                return {"found": False}
            return {"found": True, **p}

        @server.tool(annotations=READ)
        def list_open_tasks(patient_id: str) -> dict[str, Any]:
            """Open care-team tasks for a patient."""
            world.calls.append(("list_open_tasks", {"patient_id": patient_id}))
            return {"tasks": [t for t in world.tasks.values() if t["patient_id"] == patient_id]}

        @server.tool(annotations=WRITE)
        def create_task(
            patient_id: str, title: str, priority: Literal["routine", "urgent"], idempotency_key: str = ""
        ) -> dict[str, Any]:
            """Create a care-team task for a patient. Changes data: needs review."""
            world.calls.append(
                (
                    "create_task",
                    {
                        "patient_id": patient_id,
                        "title": title,
                        "priority": priority,
                        "idempotency_key": idempotency_key,
                    },
                )
            )
            if idempotency_key and idempotency_key in world.tasks:
                return world.tasks[idempotency_key]
            task = {
                "task_id": f"t-{len(world.tasks) + 1}",
                "patient_id": patient_id,
                "title": title,
                "priority": priority,
            }
            world.tasks[idempotency_key or task["task_id"]] = task
            return task

        @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
        def delete_task(task_id: str) -> str:
            """Delete a task. Not on this agent's allowlist."""
            world.calls.append(("delete_task", {"task_id": task_id}))
            return "deleted"

        return server

    def called(self, tool: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == tool]


ALLOWED_TOOLS = ["care__find_patient", "care__list_open_tasks", "care__create_task"]
