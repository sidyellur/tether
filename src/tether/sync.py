"""sync.py - the connection factory.

Zero config -> a stdlib sqlite3 connection (the local-only default). Sync
credentials present -> a libSQL embedded replica: local-speed reads, writes
that round-trip to the hosted primary. ANY failure on the replica path
degrades to the local file. Memory must never break the agent's work, so
open_connection never raises.

SPIKE FINDINGS (originally verified against libsql-experimental 0.0.41/0.0.55
on macOS arm64; re-verified against mainline libsql 0.1.11 on linux for #63,
with deltas noted):
  - Import name and connect signature match what's used below:
    `libsql_experimental.connect(database, sync_url=None, auth_token="", ...)`.
    STILL TRUE on mainline `libsql`, with ONE breaking rename:
    `check_same_thread` became `_check_same_thread`. Passing the old name to
    the new client raises TypeError at connect() -- which open_connection
    would catch and degrade, so the failure mode of a naive dependency bump
    is a SILENT drop to local-only, not a crash. `_import_libsql()` resolves
    the right name for whichever client is installed.
    Mainline also adds `sync_interval` (native periodic background sync) and
    `offline`; neither is used yet -- `sync_interval` is a natural follow-up
    to the read-path debounce added in #62.
  - `.sync()` on a connection opened WITHOUT sync_url raises ValueError
    ("Sync is not supported in databases opened in File mode.") -- confirms
    the local path must never call `.sync()`, which is why `_local()` below
    uses a no-op. RE-VERIFIED on mainline: same behavior, byte-identical
    message.
  - Cross-thread `.execute()`/`.sync()` calls did not hit any thread-safety
    guard in the versions tested, so the background-thread + join(timeout)
    pattern is safe to use.
  - NOT RE-VERIFIED for #63: the two findings below (retry-forever on an
    unreachable host, and the abandoned probe thread) both need a network
    black hole to observe. The sandbox used for the #63 re-verification gets
    a definitive 403 from its egress proxy instead, which mainline libsql
    surfaces immediately (degrades in ~0.2s) -- that exercises the error
    path, not the hang. So the bounded-probe machinery below stays exactly as
    it was: it is still the only thing standing between a black-holed backend
    and a server that never finishes starting.
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


def _import_libsql():
    """(module, thread_kwarg_name) for whichever libSQL client is installed.

    Prefers mainline `libsql`; falls back to `libsql-experimental`, which is
    frozen at 0.0.55 and superseded (#63). The two differ in one detail that
    matters here: mainline renamed `check_same_thread` to `_check_same_thread`.

    That rename is why this cannot be a plain dependency bump. Passing the old
    name to the new client raises TypeError at connect() - and open_connection
    catches everything and degrades - so a naive version bump would have
    silently dropped every sync user to local-only, with nothing but a `sync
    offline` line to show for it. Resolving the name here keeps both clients
    working and keeps that failure impossible.
    """
    try:
        import libsql
        return libsql, "_check_same_thread"
    except ImportError:
        import libsql_experimental
        return libsql_experimental, "check_same_thread"


def _local(db_path, mode="local"):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # WAL lets readers (e.g. recall) proceed alongside a writer instead of
    # blocking; busy_timeout makes a contended write retry instead of an
    # immediate "database is locked" (#43 - recall itself writes when the
    # associative graph is enabled, so concurrent recalls can contend).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    # #84: SQLite's default synchronous=FULL fsyncs the WAL on EVERY commit,
    # and tether commits on the hot path (remember once; recall twice with
    # the graph on). In WAL mode NORMAL is still safe against corruption -
    # it only fsyncs at checkpoints - and the sole exposure is that the last
    # few committed transactions can roll back after a power loss or OS
    # crash (an application crash loses nothing). Measured 0.29 ms -> 0.02 ms
    # per commit here; an SSD fsync is typically 1-5 ms. Connection-level,
    # so it lives with busy_timeout rather than in the file.
    conn.execute("PRAGMA synchronous=NORMAL")
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
    libsql, thread_kwarg = _import_libsql()

    conn = libsql.connect(
        str(db_path), sync_url=sync_cfg.url, auth_token=sync_cfg.token,
        **{thread_kwarg: False})

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
