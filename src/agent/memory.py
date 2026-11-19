"""Long-term memory, off unless AGENT_MEMORY is set: what a person asked the agent to keep across
their conversations.

Each memory belongs to one person in one tenant scope (the tenants their token named when they
said it). Nobody else reads it: not another person, not the same person in another tenant. In
Postgres that is row-level security on the table, not only a where clause: every statement runs
with the person and tenant set for its transaction, and the policy hides every other row.

What goes in: only what the person says, through the `remember` tool, in a conversation (never a
run started by an event: there's no person to own it). The model reads it back with `recall`, as
untrusted text. People see and delete theirs (GET and DELETE /memories). Memories expire after
AGENT_MEMORY_RETENTION_DAYS (90 by default); each person keeps at most AGENT_MEMORY_MAX_PER_PERSON,
oldest dropped first. They hold what people wrote: the same data class as the conversations, in
the same database (encrypted at rest, in the product's region).
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

MAX_CHARS = 500

REMEMBER = {
    "name": "remember",
    "description": "Keep something for this person's future conversations: a preference, a standing "
    "instruction, or a fact about their work that they asked you to remember. Only what they said, in "
    f"their words, one thing per call, at most {MAX_CHARS} characters. Never passwords, keys or other "
    "secrets.",
    "input_schema": {
        "type": "object",
        "properties": {"fact": {"type": "string", "maxLength": MAX_CHARS}},
        "required": ["fact"],
    },
}
RECALL = {
    "name": "recall",
    "description": "Look up what this person asked you to remember in earlier conversations. Give a few "
    "words about what you need; empty returns the most recent.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
TOOLS = ("remember", "recall")


def scope_of(tenants: list[str] | None) -> str:
    """The tenant scope a memory lives in: the tenants the person's token named, in order."""
    return ",".join(sorted({str(t) for t in tenants or [] if str(t)}))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class Memory(Protocol):
    def remember(self, owner: str, tenant: str, fact: str, source: str = "") -> dict[str, Any]: ...
    def recall(self, owner: str, tenant: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]: ...
    def mine(self, owner: str, tenant: str) -> list[dict[str, Any]]: ...
    def forget(self, owner: str, tenant: str, memory_id: str | None = None) -> int: ...
    def prune(self) -> int: ...


def _clean(fact: str) -> str:
    fact = " ".join(str(fact or "").split())
    if not fact:
        raise ValueError("Nothing to remember")
    if len(fact) > MAX_CHARS:
        raise ValueError(f"Keep it under {MAX_CHARS} characters")
    return fact


class InMemory:
    """One process, lost on restart: local development and tests."""

    def __init__(self, retention_days: float = 90, max_per_person: int = 200) -> None:
        self.retention = timedelta(days=retention_days)
        self.max = max_per_person
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def _live(self, owner: str, tenant: str) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        return [
            r for r in self.rows if r["owner"] == owner and r["tenant"] == tenant and r["expires_at"] > now
        ]

    @staticmethod
    def _out(r: dict[str, Any]) -> dict[str, Any]:
        return {"id": r["id"], "fact": r["fact"], "created_at": r["created_at"].isoformat()}

    def remember(self, owner: str, tenant: str, fact: str, source: str = "") -> dict[str, Any]:
        fact = _clean(fact)
        now = datetime.now(UTC)
        with self.lock:
            for r in self._live(owner, tenant):
                if r["fact"].lower() == fact.lower():
                    return self._out(r)
            row = {
                "id": uuid.uuid4().hex,
                "owner": owner,
                "tenant": tenant,
                "fact": fact,
                "source": source,
                "created_at": now,
                "expires_at": now + self.retention,
            }
            self.rows.append(row)
            mine = sorted(self._live(owner, tenant), key=lambda r: r["created_at"])
            for old in mine[: max(0, len(mine) - self.max)]:
                self.rows.remove(old)
        return self._out(row)

    def recall(self, owner: str, tenant: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        rows = sorted(self._live(owner, tenant), key=lambda r: r["created_at"], reverse=True)
        want = _words(query)
        if want:
            rows = [r for r in rows if want & _words(r["fact"])]
            rows.sort(key=lambda r: len(want & _words(r["fact"])), reverse=True)
        return [self._out(r) for r in rows[:limit]]

    def mine(self, owner: str, tenant: str) -> list[dict[str, Any]]:
        return [
            self._out(r)
            for r in sorted(self._live(owner, tenant), key=lambda r: r["created_at"], reverse=True)
        ]

    def forget(self, owner: str, tenant: str, memory_id: str | None = None) -> int:
        with self.lock:
            gone = [
                r
                for r in self.rows
                if r["owner"] == owner
                and r["tenant"] == tenant
                and (memory_id is None or r["id"] == memory_id)
            ]
            for r in gone:
                self.rows.remove(r)
        return len(gone)

    def prune(self) -> int:
        now = datetime.now(UTC)
        with self.lock:
            gone = [r for r in self.rows if r["expires_at"] <= now]
            for r in gone:
                self.rows.remove(r)
        return len(gone)


SCHEMA = """
create table if not exists agent_memories (
  id text primary key,
  owner text not null,
  tenant text not null,
  fact text not null,
  source text not null default '',
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);
create index if not exists agent_memories_owner on agent_memories (owner, tenant, created_at desc);
create index if not exists agent_memories_expires on agent_memories (expires_at);
alter table agent_memories enable row level security;
alter table agent_memories force row level security;
drop policy if exists agent_memories_own on agent_memories;
create policy agent_memories_own on agent_memories
  using (owner = current_setting('agent.memory_owner', true)
         and tenant = current_setting('agent.memory_tenant', true))
  with check (owner = current_setting('agent.memory_owner', true)
              and tenant = current_setting('agent.memory_tenant', true));
-- Pruning sees and deletes expired rows only (a delete's where clause needs select rights too).
drop policy if exists agent_memories_expire on agent_memories;
create policy agent_memories_expire on agent_memories for delete
  using (current_setting('agent.memory_prune', true) = 'on' and expires_at <= now());
drop policy if exists agent_memories_expired on agent_memories;
create policy agent_memories_expired on agent_memories for select
  using (current_setting('agent.memory_prune', true) = 'on' and expires_at <= now());
"""


class PostgresMemory:
    """Memories in Postgres, each statement in a transaction scoped to one person and tenant.

    Row-level security holds for the table's owner too (force), so connect as an ordinary role:
    a superuser or a BYPASSRLS role would see past it."""

    def __init__(
        self, pool: Any, retention_days: float = 90, max_per_person: int = 200, setup: bool = True
    ) -> None:
        self.pool, self.retention_days, self.max = pool, retention_days, max_per_person
        if setup:
            with self.pool.connection() as conn:
                conn.execute(SCHEMA)

    def _as(self, conn: Any, owner: str, tenant: str) -> None:
        if not owner:
            raise ValueError("A memory belongs to a person")
        conn.execute(
            "select set_config('agent.memory_owner', %s, true), set_config('agent.memory_tenant', %s, true)",
            (owner, tenant),
        )

    @staticmethod
    def _out(r: Any) -> dict[str, Any]:
        return {"id": r["id"], "fact": r["fact"], "created_at": r["created_at"].isoformat()}

    def remember(self, owner: str, tenant: str, fact: str, source: str = "") -> dict[str, Any]:
        fact = _clean(fact)
        with self.pool.connection() as conn, conn.transaction():
            self._as(conn, owner, tenant)
            same = conn.execute(
                "select id, fact, created_at from agent_memories where lower(fact) = lower(%s)"
                " and expires_at > now()",
                (fact,),
            ).fetchone()
            if same:
                return self._out(same)
            row = conn.execute(
                "insert into agent_memories (id, owner, tenant, fact, source, expires_at)"
                " values (%s, %s, %s, %s, %s, now() + make_interval(days => %s))"
                " returning id, fact, created_at",
                (uuid.uuid4().hex, owner, tenant, fact, source[:200], int(self.retention_days)),
            ).fetchone()
            conn.execute(
                "delete from agent_memories where id in (select id from agent_memories"
                " order by created_at desc offset %s)",
                (self.max,),
            )
        return self._out(row)

    def recall(self, owner: str, tenant: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.transaction():
            self._as(conn, owner, tenant)
            if query.strip():
                rows = conn.execute(
                    "select id, fact, created_at from agent_memories, websearch_to_tsquery('simple', %s) q"
                    " where expires_at > now() and to_tsvector('simple', fact) @@ q"
                    " order by ts_rank(to_tsvector('simple', fact), q) desc, created_at desc limit %s",
                    (" or ".join(_words(query)) or query, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "select id, fact, created_at from agent_memories where expires_at > now()"
                    " order by created_at desc limit %s",
                    (limit,),
                ).fetchall()
        return [self._out(r) for r in rows]

    def mine(self, owner: str, tenant: str) -> list[dict[str, Any]]:
        return self.recall(owner, tenant, "", limit=self.max)

    def forget(self, owner: str, tenant: str, memory_id: str | None = None) -> int:
        with self.pool.connection() as conn, conn.transaction():
            self._as(conn, owner, tenant)
            if memory_id is None:
                cur = conn.execute("delete from agent_memories")
            else:
                cur = conn.execute("delete from agent_memories where id = %s", (memory_id,))
            return int(cur.rowcount or 0)

    def prune(self) -> int:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("select set_config('agent.memory_prune', 'on', true)")
            return int(conn.execute("delete from agent_memories where expires_at <= now()").rowcount or 0)
