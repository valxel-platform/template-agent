"""What a run or conversation is doing right now, readable from any replica, and the list of each
person's conversations.

Progress: the pod executing a thread writes what happens as it happens (the reply's text as the
model writes it, each tool step, a proposal waiting for approval, the turn ending) to a table,
and any pod streams it to the person (GET /conversations/{id}/stream). A pod tells the others a
thread moved on with NOTIFY, so a stream wakes at once; streams also re-read every two seconds,
so a missed notification only delays a line. Progress is short-lived: the conversation itself
(every message) is in the checkpoint, and progress older than a day is pruned.

Conversations: who started each one, for "my conversations".

Both hold what people wrote and what the agent answered: the same data class as the checkpoint,
in the same database (encrypted at rest, in the product's region).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Protocol

log = logging.getLogger("agent.store")
CHANNEL = "agent_progress"


class Progress(Protocol):
    def emit(self, thread: str, kind: str, data: dict[str, Any]) -> int: ...
    def read(self, thread: str, after: int = 0, limit: int = 500) -> list[dict[str, Any]]: ...
    def last(self, thread: str) -> int: ...
    def latest(self, thread: str, kinds: tuple[str, ...]) -> dict[str, Any] | None: ...
    def watch(self, thread: str) -> Watch: ...
    def unwatch(self, w: Watch) -> None: ...
    def prune(self, older_than_s: float) -> int: ...


class Conversations(Protocol):
    def add(self, cid: str, owner: str, tenants: list[str], title: str) -> None: ...
    def touch(self, cid: str) -> None: ...
    def mine(self, owner: str, limit: int = 50) -> list[dict[str, Any]]: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Watch:
    """A stream waiting for a thread to move on, without holding a thread of its own."""

    def __init__(self, thread: str) -> None:
        self.thread = thread
        self.loop = asyncio.get_running_loop()
        self.event = asyncio.Event()

    def wake(self) -> None:
        self.loop.call_soon_threadsafe(self.event.set)

    async def wait(self, timeout: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.event.wait(), timeout)
        self.event.clear()


class _Watchers:
    def __init__(self) -> None:
        self._watches: dict[str, set[Watch]] = defaultdict(set)
        self._guard = threading.Lock()

    def watch(self, thread: str) -> Watch:
        w = Watch(thread)
        with self._guard:
            self._watches[thread].add(w)
        return w

    def unwatch(self, w: Watch) -> None:
        with self._guard:
            self._watches[w.thread].discard(w)
            if not self._watches[w.thread]:
                del self._watches[w.thread]

    def wake(self, thread: str) -> None:
        with self._guard:
            watches = list(self._watches.get(thread, ()))
        for w in watches:
            try:
                w.wake()
            except RuntimeError:  # its event loop has closed
                self.unwatch(w)


class MemoryProgress(_Watchers):
    def __init__(self) -> None:
        super().__init__()
        self.items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.seq = 0
        self.cond = threading.Condition()

    def emit(self, thread: str, kind: str, data: dict[str, Any]) -> int:
        with self.cond:
            self.seq += 1
            self.items[thread].append({"seq": self.seq, "kind": kind, "data": data, "at": _now()})
            seq = self.seq
        self.wake(thread)
        return seq

    def read(self, thread: str, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self.cond:
            return [e for e in self.items.get(thread, []) if e["seq"] > after][:limit]

    def last(self, thread: str) -> int:
        with self.cond:
            got = self.items.get(thread) or []
            return got[-1]["seq"] if got else 0

    def latest(self, thread: str, kinds: tuple[str, ...]) -> dict[str, Any] | None:
        with self.cond:
            return next((e for e in reversed(self.items.get(thread, [])) if e["kind"] in kinds), None)

    def prune(self, older_than_s: float) -> int:
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, UTC).isoformat()
        n = 0
        with self.cond:
            for t, items in list(self.items.items()):
                keep = [e for e in items if e["at"] >= cutoff]
                n += len(items) - len(keep)
                self.items[t] = keep
        return n


class MemoryConversations:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def add(self, cid: str, owner: str, tenants: list[str], title: str) -> None:
        self.rows.setdefault(
            cid,
            {
                "conversation_id": cid,
                "owner": owner,
                "tenants": tenants,
                "title": title,
                "created_at": _now(),
                "updated_at": _now(),
            },
        )

    def touch(self, cid: str) -> None:
        if cid in self.rows:
            self.rows[cid]["updated_at"] = _now()

    def mine(self, owner: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = [r for r in self.rows.values() if r["owner"] == owner]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]


SCHEMA = """
create table if not exists agent_progress (
  seq bigserial primary key,
  thread text not null,
  kind text not null,
  data jsonb not null,
  at timestamptz not null default now()
);
create index if not exists agent_progress_thread on agent_progress (thread, seq);
create index if not exists agent_progress_at on agent_progress (at);
create table if not exists agent_conversations (
  id text primary key,
  owner text not null,
  tenants jsonb not null default '[]',
  title text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists agent_conversations_owner on agent_conversations (owner, updated_at desc);
"""


def _row(r: Any) -> dict[str, Any]:
    d = dict(r) if isinstance(r, dict) else {"seq": r[0], "kind": r[1], "data": r[2], "at": r[3]}
    at: Any = d.get("at")
    return {
        "seq": int(d["seq"]),
        "kind": d["kind"],
        "data": d["data"],
        "at": str(at.isoformat()) if hasattr(at, "isoformat") else str(at),
    }


class PostgresProgress(_Watchers):
    """Progress in Postgres, with one LISTEN connection per pod that wakes its waiting streams."""

    def __init__(self, pool: Any, dsn: str, listen: bool = True) -> None:
        super().__init__()
        self.pool, self.dsn = pool, dsn
        with self.pool.connection() as conn:
            for statement in filter(None, (s.strip() for s in SCHEMA.split(";"))):
                conn.execute(statement)
        if listen:
            threading.Thread(target=self._listen, name="progress-listen", daemon=True).start()

    def _listen(self) -> None:
        import psycopg

        while True:
            try:
                with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as conn:
                    conn.execute(f"listen {CHANNEL}")
                    while True:
                        for n in conn.notifies(timeout=30):
                            self.wake(n.payload)
                        conn.execute("select 1")  # notice a dead connection
            except Exception as exc:  # noqa: BLE001 - keep listening; streams poll meanwhile
                log.warning("progress listener reconnecting: %s", type(exc).__name__)
                time.sleep(2)

    def emit(self, thread: str, kind: str, data: dict[str, Any]) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "with i as (insert into agent_progress (thread, kind, data)"
                " values (%s, %s, %s) returning seq) select seq, pg_notify(%s, %s) from i",
                (thread, kind, json.dumps(data), CHANNEL, thread),
            ).fetchone()
        return int(row["seq"] if isinstance(row, dict) else row[0])

    def read(self, thread: str, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "select seq, kind, data, at from agent_progress where thread = %s and seq > %s"
                " order by seq limit %s",
                (thread, after, limit),
            ).fetchall()
        return [_row(r) for r in rows]

    def last(self, thread: str) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select coalesce(max(seq), 0) as seq from agent_progress where thread = %s", (thread,)
            ).fetchone()
        return int(row["seq"] if isinstance(row, dict) else row[0])

    def latest(self, thread: str, kinds: tuple[str, ...]) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select seq, kind, data, at from agent_progress where thread = %s and kind = any(%s)"
                " order by seq desc limit 1",
                (thread, list(kinds)),
            ).fetchone()
        return _row(row) if row else None

    def prune(self, older_than_s: float) -> int:
        with self.pool.connection() as conn:
            cur = conn.execute(
                "delete from agent_progress where at < now() - make_interval(secs => %s)", (older_than_s,)
            )
            return int(cur.rowcount or 0)


class PostgresConversations:
    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def add(self, cid: str, owner: str, tenants: list[str], title: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "insert into agent_conversations (id, owner, tenants, title) values (%s, %s, %s, %s)"
                " on conflict (id) do nothing",
                (cid, owner, json.dumps(tenants), title),
            )

    def touch(self, cid: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("update agent_conversations set updated_at = now() where id = %s", (cid,))

    def mine(self, owner: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "select id, owner, tenants, title, created_at, updated_at from agent_conversations"
                " where owner = %s order by updated_at desc limit %s",
                (owner, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = (
                dict(r)
                if isinstance(r, dict)
                else dict(
                    zip(("id", "owner", "tenants", "title", "created_at", "updated_at"), r, strict=True)
                )
            )
            out.append(
                {
                    "conversation_id": d["id"],
                    "owner": d["owner"],
                    "tenants": d["tenants"],
                    "title": d["title"],
                    "created_at": d["created_at"].isoformat(),
                    "updated_at": d["updated_at"].isoformat(),
                }
            )
        return out
