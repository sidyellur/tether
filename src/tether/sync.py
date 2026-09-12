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
  - PROBE TIMEOUT CLEANUP (#109): if the initial probe times out, `conn.close()`
    is now called (best-effort, wrapped in try/except) before raising
    TimeoutError, so the abandoned client stops retrying in the background
    instead of continuing to run against the same db_path that the local
    fallback then also opens.
  - SYNC WORKER (#109): `sync_now` used to spawn a brand-new daemon thread on
    every call and abandon it after joining for `timeout` seconds -- no cap on
    live threads, no check for an in-flight sync, so a slow/unreachable
    primary left one blocked thread per write (`tether import` of a 500-record
    backup spawned 500 of them), all calling `.sync()` concurrently on the
    same connection. `_open_replica` now starts exactly one daemon worker
    thread per connection, lazily on the first `sync_now` call.

    #132 (a follow-up on #109 itself): the first version of the
    worker coalesced a caller arriving WHILE a sync was already in flight
    onto that in-flight round unconditionally - correct when the caller's
    request predates the round it joins, wrong when it doesn't. A write
    that commits AFTER a sync has already started reading/pushing isn't
    necessarily reflected by that round, so a `sync_now()` called right
    after such a write, finding `in_flight` True, would return "done" as
    soon as the ALREADY-RUNNING round finished - without ever causing a
    round that started at-or-after its own request. No follow-up was ever
    scheduled, so that write's data was never guaranteed synced by the
    call meant to guarantee it - silently contradicting the write-path
    contract this docstring already promised ("every write still requests
    a sync and waits ... for it").

    Fixed with a generation counter (`requested`, bumped once per
    `sync_now()` call) instead of a plain in-flight flag. The worker
    snapshots `requested` into `started_serving` the instant it begins a
    round - which is exactly the set of calls that round can causally
    satisfy, since none of their writes could have raced ahead of a
    snapshot taken after they were counted - and advances the completion
    counter to that snapshot when the round finishes, so a caller only
    returns once a round has started NO EARLIER than its own request. If
    more requests land while a round is running, the worker immediately
    starts another round for them the moment the current one finishes,
    rather than waiting for a fresh caller to show up and re-trigger it.
    Callers whose requests land together, before any round has yet
    snapshotted them, still coalesce onto that one shared round - the
    efficiency property #109 exists for is unaffected; what changes is
    only the case where a request genuinely arrives after a round is
    already under way.
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
        try:
            conn.close()  # stop the abandoned client's background retries
        except Exception:
            pass
        raise TimeoutError(
            f"sync backend unreachable after {_INITIAL_SYNC_TIMEOUT}s: {sync_cfg.url}")
    if errors:
        raise errors[0]

    # One coalescing worker per connection (#109/#132), started lazily on the
    # first sync_now() call, instead of a new thread per call. See the module
    # docstring's "SYNC WORKER" note for the design and why a plain in-flight
    # flag (the #109 original) isn't enough on its own.
    wake = threading.Event()
    condition = threading.Condition()
    counter = 0          # rounds completed, as a generation number
    requested = 0         # sync_now() calls made, as a generation number
    worker_started = False

    def worker():
        nonlocal counter, requested
        while True:
            wake.wait()
            with condition:
                wake.clear()
                # Everything requested up to THIS instant is what this round
                # can causally satisfy - a request that lands after this
                # snapshot raced in too late to be reflected by the sync
                # that's about to run.
                started_serving = requested
            _safe_sync(conn)
            with condition:
                counter = started_serving
                if requested > counter:
                    # Something new landed while this round was running - go
                    # again immediately rather than waiting for another
                    # caller to show up and re-trigger it.
                    wake.set()
                condition.notify_all()

    def sync_now(timeout=2.0):
        nonlocal worker_started, requested
        with condition:
            if not worker_started:
                threading.Thread(target=worker, daemon=True).start()
                worker_started = True
            requested += 1
            target = requested
            wake.set()
            condition.wait_for(lambda: counter >= target, timeout=timeout)

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
