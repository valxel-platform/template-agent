"""One step at a time per run or conversation, across every replica.

A run (or a conversation) executes on whichever pod received the request that moved it on: the
start, a person's decision, a chat message. Nothing routes a caller back to the same pod, and
nothing needs to. The pod holds a Postgres advisory lock for the thread while it executes, so:

- another pod asked to move the same thread on at the same time refuses (409), so an approval
  sent twice never applies twice, and one message is answered at a time;
- any pod can tell a caller the thread is working: the lock is visible in pg_locks;
- a pod that dies releases its lock with its connection, and the thread can be picked up again.

The lock lives on a connection of its own, held for the whole execution. If that connection is
lost while the pod keeps going (a network blip), another pod could take the thread over, so the
running pod checks it still holds the lock before each step (`check`) and stops if it doesn't.

Without a database (local development, tests) the locks are in memory: one process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class LeaseLost(RuntimeError):
    """This pod no longer holds the thread's lock: stop before the next step."""


class Lease(Protocol):
    def alive(self) -> bool: ...
    def release(self) -> None: ...


class Locks(Protocol):
    def acquire(self, thread: str) -> Lease | None: ...
    def busy(self, thread: str) -> bool: ...


_held: ContextVar[Lease | None] = ContextVar("thread_lease", default=None)


@contextmanager
def holding(lease: Lease) -> Iterator[None]:
    """Steps run inside this hold the lease; `check` confirms it before each one."""
    token = _held.set(lease)
    try:
        yield
    finally:
        _held.reset(token)


def check() -> None:
    lease = _held.get()
    if lease is not None and not lease.alive():
        raise LeaseLost("this replica lost the thread's lock; another may have taken it over")


class MemoryLocks:
    """One process: a set of threads being worked on."""

    def __init__(self) -> None:
        self._busy: set[str] = set()
        self._guard = threading.Lock()

    def acquire(self, thread: str) -> Lease | None:
        with self._guard:
            if thread in self._busy:
                return None
            self._busy.add(thread)
        return _MemoryLease(self, thread)

    def busy(self, thread: str) -> bool:
        with self._guard:
            return thread in self._busy

    def _release(self, thread: str) -> None:
        with self._guard:
            self._busy.discard(thread)


class _MemoryLease:
    def __init__(self, owner: MemoryLocks, thread: str) -> None:
        self.owner, self.thread, self.done = owner, thread, False

    def alive(self) -> bool:
        return not self.done

    def release(self) -> None:
        if not self.done:
            self.done = True
            self.owner._release(self.thread)


# The lock's two keys: the agent (so agents sharing a database never collide) and the thread.
_TRY = "select pg_try_advisory_lock(hashtext(%s), hashtext(%s))"
_HELD = """
select exists (
  select 1 from pg_locks
   where locktype = 'advisory' and granted and objsubid = 2
     and database = (select oid from pg_database where datname = current_database())
     and classid = (hashtext(%s)::bigint & 4294967295)::oid
     and objid = (hashtext(%s)::bigint & 4294967295)::oid)
"""


class PostgresLocks:
    """Advisory locks: a dedicated connection per execution holds the thread's lock; `busy` asks
    pg_locks through the shared pool."""

    def __init__(self, dsn: str, pool: Any, agent: str, connect: Any = None) -> None:
        import psycopg

        self.dsn, self.pool, self.namespace = dsn, pool, f"agent:{agent}"
        self.connect = connect or (lambda: psycopg.connect(dsn, autocommit=True, connect_timeout=5))

    def acquire(self, thread: str) -> Lease | None:
        conn = self.connect()
        try:
            got = conn.execute(_TRY, (self.namespace, thread)).fetchone()
        except Exception:
            conn.close()
            raise
        if not got or not _first(got):
            conn.close()
            return None
        return _PostgresLease(conn, self.namespace, thread)

    def busy(self, thread: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(_HELD, (self.namespace, thread)).fetchone()
        return bool(row and _first(row))


def _first(row: Any) -> Any:
    return next(iter(row.values())) if isinstance(row, dict) else row[0]


class _PostgresLease:
    def __init__(self, conn: Any, namespace: str, thread: str) -> None:
        self.conn, self.namespace, self.thread = conn, namespace, thread

    def alive(self) -> bool:
        """The session that took the lock is still open, so the lock is still ours."""
        try:
            self.conn.execute("select 1").fetchone()
            return True
        except Exception:  # noqa: BLE001 - any failure means the session (and its lock) may be gone
            return False

    def release(self) -> None:
        try:
            self.conn.execute(
                "select pg_advisory_unlock(hashtext(%s), hashtext(%s))", (self.namespace, self.thread)
            )
        except Exception:  # noqa: BLE001, S110 - closing the session releases it anyway
            pass
        finally:
            self.conn.close()
