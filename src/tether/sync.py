"""sync.py - the connection factory.

Zero config -> a stdlib sqlite3 connection (the local-only default). Sync
credentials present -> a libSQL embedded replica: local-speed reads, writes
that round-trip to the hosted primary. ANY failure on the replica path
degrades to the local file. Memory must never break the agent's work, so
open_connection never raises.

SPIKE FINDINGS (verified live against libsql-experimental, PyPI releases
0.0.41/0.0.55, on macOS arm64):
  - Import name and connect signature match what's used below:
    `libsql_experimental.connect(database, sync_url=None, auth_token="", ...)`.
  - `.sync()` on a connection opened WITHOUT sync_url raises ValueError
    ("Sync is not supported in databases opened in File mode.") -- confirms
    the local path must never call `.sync()`, which is why `_local()` below
    uses a no-op.
  - Cross-thread `.execute()`/`.sync()` calls did not hit any thread-safety
    guard in the versions tested, so the background-thread + join(timeout)
    pattern is safe to use.
  - IMPORTANT DEVIATION FROM THE ORIGINAL PLAN: `.sync()` does NOT fail fast
    against an unreachable/bogus sync_url. It retries the handshake
    internally (observed every ~2-3s) and does not return control or raise
    -- it was still retrying after 20+ seconds in testing. A bare, inline
    `conn.sync()` used as an initial connectivity probe would therefore hang
    server startup indefinitely instead of raising. So the initial probe
    below is bounded by the same background-thread + timeout pattern used
    for later syncs, and a timeout is treated as a failure.
  - KNOWN LIMITATION (accepted for v0.1's experimental, opt-in sync layer):
    if the initial probe times out, the abandoned libSQL connection's
    background thread keeps retrying against the same db_path that the
    local fallback then also opens. This is a daemon thread (never blocks
    process exit) and, in the common failure case (persistently unreachable
    network), it never actually writes -- so there is no realistic data
    corruption path, but it is not a fully clean cancellation.
"""

import sqlite3
import sys
import threading

_INITIAL_SYNC_TIMEOUT = 5.0
_BUSY_TIMEOUT_MS = 5000


def _local(db_path, mode="local"):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # WAL lets readers (e.g. recall) proceed alongside a writer instead of
    # blocking; busy_timeout makes a contended write retry instead of an
    # immediate "database is locked" (#43 - recall itself writes when the
    # associative graph is enabled, so concurrent recalls can contend).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn, (lambda timeout=2.0: None), mode


def _safe_sync(conn):
    try:
        conn.sync()
    except Exception:
        pass  # a failed background sync must never surface


def _open_replica(db_path, sync_cfg):
    """Open a libSQL embedded replica. Raises on any failure; caller degrades.

    See the module docstring's SPIKE FINDINGS for why the initial sync is
    bounded with a background thread rather than called inline.
    """
    import libsql_experimental as libsql

    conn = libsql.connect(
        str(db_path), sync_url=sync_cfg.url, auth_token=sync_cfg.token,
        check_same_thread=False)

    errors = []

    def probe():
        try:
            conn.sync()  # initial pull; part of "did the backend work?"
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(_INITIAL_SYNC_TIMEOUT)
    if t.is_alive():
        raise TimeoutError(
            f"sync backend unreachable after {_INITIAL_SYNC_TIMEOUT}s: {sync_cfg.url}")
    if errors:
        raise errors[0]

    def sync_now(timeout=2.0):
        t = threading.Thread(target=_safe_sync, args=(conn,), daemon=True)
        t.start()
        t.join(timeout)  # bounded: a hung sync never blocks a read

    return conn, sync_now, "replica"


def open_connection(db_path, sync_cfg):
    """Returns (conn, sync_now, mode). mode is "local" (no sync configured),
    "replica" (embedded libSQL replica live), or "degraded" (sync was
    configured but the replica path failed, so it fell back to the local
    file) - the status resource (#51) surfaces this to tell "sync isn't
    configured" apart from "sync is configured but broken"."""
    if sync_cfg is None:
        return _local(db_path)
    try:
        return _open_replica(db_path, sync_cfg)
    except Exception as e:  # import missing, connect failed, initial sync failed
        sys.stderr.write(f"tether: sync offline ({e}); using local file\n")
        return _local(db_path, mode="degraded")
