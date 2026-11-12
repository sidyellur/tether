import os
import socket
import sqlite3
import stat
import threading
import time

import pytest

from tether import sync
from tether.config import SyncConfig


@pytest.fixture(autouse=True)
def _no_network_preflight(monkeypatch, request):
    """The fake clients below use made-up hostnames; the real preflight would
    go and look for them. Tests of the preflight itself opt back in."""
    if "real_preflight" not in request.keywords:
        monkeypatch.setattr(sync, "_preflight", lambda url, timeout: None)


def test_no_config_returns_plain_sqlite(tmp_path):
    conn, sync_now, mode = sync.open_connection(tmp_path / "m.db", None)
    assert isinstance(conn, sqlite3.Connection)
    assert mode == "local"
    assert sync_now() is None  # no-op, no error
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    assert conn.execute("SELECT x FROM t").fetchone()[0] == 1


def test_local_connection_sets_wal_and_busy_timeout(tmp_path):
    # #43: recall itself writes (session tracking) when the associative graph
    # is enabled, so concurrent recalls can contend for the write lock. WAL
    # lets readers proceed alongside a writer, and busy_timeout makes a
    # contended write retry instead of failing instantly.
    conn, _, _ = sync.open_connection(tmp_path / "m.db", None)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == sync._BUSY_TIMEOUT_MS
    # #84: WAL + synchronous=NORMAL (1) - no fsync per commit, only at
    # checkpoints; still corruption-safe in WAL mode.
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_synchronous_normal_is_per_connection_not_in_the_file(tmp_path):
    """#84: `synchronous` is connection-level (unlike journal_mode, which
    persists in the file), so a second plain sqlite3 connection to the same
    file gets SQLite's default again - which is why it must be set in
    _local() alongside busy_timeout rather than once at migrate time."""
    conn, _, _ = sync.open_connection(tmp_path / "m.db", None)
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    other = sqlite3.connect(str(tmp_path / "m.db"))
    assert other.execute("PRAGMA synchronous").fetchone()[0] == 2     # FULL default
    conn2, _, _ = sync.open_connection(tmp_path / "m.db", None)
    assert conn2.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_every_local_connection_gets_the_pragmas(tmp_path, monkeypatch):
    # Same guarantee via the degrade-to-local path (#43 covers "every
    # connection opened in _local()", not just the zero-config startup path).
    def boom(*a, **k):
        raise RuntimeError("cannot reach turso")
    monkeypatch.setattr(sync, "_open_replica", boom)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, _, _ = sync.open_connection(tmp_path / "m.db", cfg)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == sync._BUSY_TIMEOUT_MS
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_backend_failure_degrades_to_local(tmp_path, monkeypatch, capsys):
    # Force the replica path to blow up however it likes.
    def boom(*a, **k):
        raise RuntimeError("cannot reach turso")
    monkeypatch.setattr(sync, "_open_replica", boom)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)

    # Degraded, not dead: a real local connection and a safe no-op sync.
    assert isinstance(conn, sqlite3.Connection)
    assert mode == "degraded"
    assert sync_now() is None
    conn.execute("CREATE TABLE t(x)")
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    assert "sync offline" in capsys.readouterr().err


# --- #63: works with mainline libsql AND the frozen experimental client ------

def _fake_client(monkeypatch, name, thread_kwarg):
    """Install a stand-in libSQL module that records how connect() was called."""
    import sys
    import types

    calls = {}

    def connect(database, **kwargs):
        calls["database"] = database
        calls["kwargs"] = kwargs
        if thread_kwarg not in kwargs:
            raise TypeError(
                f"connect() got an unexpected keyword argument "
                f"{sorted(set(kwargs) - {'sync_url', 'auth_token'})}")

        class Conn:
            def sync(self):
                return None

        return Conn()

    mod = types.ModuleType(name)
    mod.connect = connect
    monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_uses_underscored_kwarg_for_mainline_libsql(tmp_path, monkeypatch):
    """Mainline libsql renamed check_same_thread -> _check_same_thread. Passing
    the old name raises TypeError at connect(), which open_connection catches
    and degrades - so getting this wrong silently drops every sync user to
    local-only instead of failing loudly."""
    calls = _fake_client(monkeypatch, "libsql", "_check_same_thread")

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    _conn, _sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)

    assert mode == "replica", "replica path degraded when it should have worked"
    assert calls["kwargs"]["_check_same_thread"] is False
    assert "check_same_thread" not in calls["kwargs"]
    assert calls["kwargs"]["sync_url"] == "libsql://x.turso.io"


def test_falls_back_to_experimental_client_with_its_own_kwarg(tmp_path, monkeypatch):
    """An existing libsql-experimental install must keep working - the extra
    moved, but nobody's environment has to."""
    import sys

    monkeypatch.setitem(sys.modules, "libsql", None)   # simulate not installed
    calls = _fake_client(monkeypatch, "libsql_experimental", "check_same_thread")

    real_import = sync._import_libsql

    def only_experimental():
        import libsql_experimental
        return libsql_experimental, "check_same_thread"

    monkeypatch.setattr(sync, "_import_libsql", only_experimental)
    try:
        cfg = SyncConfig("libsql://x.turso.io", "tok")
        _conn, _sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)
        assert mode == "replica"
        assert calls["kwargs"]["check_same_thread"] is False
    finally:
        monkeypatch.setattr(sync, "_import_libsql", real_import)


def test_missing_client_still_degrades_to_local(tmp_path, monkeypatch, capsys):
    """No libSQL client installed at all: sync is an optional extra, so this
    must degrade rather than raise."""
    def no_client():
        raise ImportError("no libsql here")

    monkeypatch.setattr(sync, "_import_libsql", no_client)
    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, _sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)
    assert mode == "degraded"
    assert isinstance(conn, sqlite3.Connection)
    assert "sync offline" in capsys.readouterr().err


# --- #109: one coalescing worker instead of a thread per sync_now() call ----

def _fake_replica_client(monkeypatch, connect):
    """Install a stand-in `libsql` module whose connect() is the given
    callable, and make sure the mainline import path picks it up."""
    import sys
    import types

    mod = types.ModuleType("libsql")
    mod.connect = connect
    monkeypatch.setitem(sys.modules, "libsql", mod)


def test_sync_now_does_not_spawn_a_thread_per_call(tmp_path, monkeypatch):
    """A slow/unreachable primary used to leave one thread blocked in
    conn.sync() per sync_now() call. Now there is exactly one worker thread,
    started lazily, regardless of how many times sync_now() is called."""
    block_forever = threading.Event()  # never set: simulates a hung primary

    class Conn:
        def __init__(self):
            self.calls = 0

        def sync(self):
            self.calls += 1
            if self.calls > 1:  # first call is the initial probe; let it pass
                block_forever.wait()

        def close(self):
            pass

    conn_holder = {}

    def connect(database, **kwargs):
        c = Conn()
        conn_holder["conn"] = c
        return c

    _fake_replica_client(monkeypatch, connect)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    _conn, sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)
    assert mode == "replica"

    baseline = threading.active_count()
    for _ in range(50):
        sync_now(0.05)
    grew = threading.active_count() - baseline
    assert grew <= 2, f"expected ~1 worker thread total, active count grew by {grew}"

    block_forever.set()  # release the blocked worker so nothing lingers


def test_sync_now_caller_arriving_mid_flight_gets_its_own_round(tmp_path, monkeypatch):
    """#132: a caller whose sync_now() lands WHILE a round is already running
    must not be satisfied by that round alone - its own request arrived too
    late to be reflected by a sync that had already started. It must cause a
    fresh, guaranteed round once the in-flight one finishes, rather than
    silently being told "done" by a round that never saw its write.

    Pre-#132 this coalesced onto the in-flight round for free (calls["n"]
    stayed at 2 - exactly one more sync() beyond the initial probe) and never
    scheduled the follow-up; that was the bug, not the fix."""
    import time

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    class Conn:
        def sync(self):
            calls["n"] += 1
            if calls["n"] > 1:  # first call is the initial probe
                entered.set()
                release.wait()

        def close(self):
            pass

    def connect(database, **kwargs):
        return Conn()

    _fake_replica_client(monkeypatch, connect)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    _conn, sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)
    assert mode == "replica"
    assert calls["n"] == 1  # the initial probe already ran

    results = []

    def caller():
        sync_now(2.0)
        results.append(True)

    t1 = threading.Thread(target=caller)
    t1.start()
    assert entered.wait(2.0), "worker never entered the blocking sync() call"

    t2 = threading.Thread(target=caller)
    t2.start()
    time.sleep(0.2)  # let t2 register while t1's round is still in flight

    release.set()   # release t1's round; its own sync() call unblocks...
    # ...but t2's follow-up round immediately starts a SECOND sync() call,
    # which also blocks on `release` (already set, so it returns at once) -
    # give both threads a moment to actually finish.
    t1.join(2.0)
    t2.join(2.0)

    assert results == [True, True]
    assert calls["n"] == 3, (
        "t2's mid-flight request was satisfied without its own sync() round")


def test_sync_now_requests_queued_during_a_round_still_coalesce(tmp_path, monkeypatch):
    """The efficiency property #109 exists for is unaffected by #132: any
    number of sync_now() calls made WHILE a round is running (a burst of
    writes queued up behind one slow round) still coalesce onto a single
    follow-up round - only when they land is what matters, and "during a
    round already running" is the case the worker cannot have snapshotted
    yet, so its NEXT snapshot naturally captures all of them together.
    Deterministic by construction: round 1 is held open with an Event, so
    t1/t2 are guaranteed to queue while the worker is blocked inside
    conn.sync() and hasn't taken its next snapshot yet - unlike true
    simultaneity, which would race the worker's own scheduling."""
    import time

    round1_entered = threading.Event()
    release_round1 = threading.Event()
    calls = {"n": 0}

    class Conn:
        def sync(self):
            calls["n"] += 1
            if calls["n"] == 2:            # round 1 (call 1 is the probe)
                round1_entered.set()
                release_round1.wait()
            # round 2 (call 3): returns immediately, nothing held it back

        def close(self):
            pass

    def connect(database, **kwargs):
        return Conn()

    _fake_replica_client(monkeypatch, connect)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    _conn, sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)
    assert calls["n"] == 1

    results = []

    def caller():
        sync_now(2.0)
        results.append(True)

    t0 = threading.Thread(target=caller)   # triggers round 1
    t0.start()
    assert round1_entered.wait(2.0), "round 1 never started"

    # t1 and t2 both queue up WHILE round 1 is held open - guaranteed to land
    # before the worker's next snapshot, since it can't take one until round
    # 1's conn.sync() (blocked on release_round1) returns.
    t1 = threading.Thread(target=caller)
    t2 = threading.Thread(target=caller)
    t1.start()
    t2.start()
    time.sleep(0.3)   # give both a moment to reach their wait_for() and queue up

    release_round1.set()
    t0.join(2.0)
    t1.join(2.0)
    t2.join(2.0)

    assert results == [True, True, True]
    assert calls["n"] == 3, (
        f"queued-together requests triggered {calls['n'] - 2} follow-up "
        "rounds instead of coalescing onto 1")


def test_probe_timeout_closes_the_abandoned_connection(tmp_path, monkeypatch):
    """#109: if the initial probe times out, the abandoned client must be
    closed so its background retry loop stops running against the same
    db_path the local fallback then opens."""
    never = threading.Event()  # never set: the probe call hangs forever

    class Conn:
        def __init__(self):
            self.closed = False

        def sync(self):
            never.wait()

        def close(self):
            self.closed = True

    conn_holder = {}

    def connect(database, **kwargs):
        c = Conn()
        conn_holder["conn"] = c
        return c

    _fake_replica_client(monkeypatch, connect)
    monkeypatch.setattr(sync, "_INITIAL_SYNC_TIMEOUT", 0.05)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, _sync_now, mode = sync.open_connection(tmp_path / "m.db", cfg)

    assert mode == "degraded"
    assert conn_holder["conn"].closed is True
    assert isinstance(conn, sqlite3.Connection)


# --- start-up hang, unencrypted URLs, log redaction, file permissions -------

def _silent_listener():
    """A server that accepts a connection and then never says anything."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    held = []

    def accept():
        try:
            while True:
                held.append(srv.accept()[0])
        except OSError:
            pass
    threading.Thread(target=accept, daemon=True).start()
    return srv, srv.getsockname()[1]


@pytest.mark.real_preflight
def test_a_host_that_accepts_but_never_answers_degrades_instead_of_hanging(
        tmp_path, monkeypatch, capsys):
    srv, port = _silent_listener()
    connected = []

    class FakeLibsql:
        @staticmethod
        def connect(*a, **k):
            connected.append(1)             # the real client would block here forever
            raise AssertionError("connect must not be reached")

    monkeypatch.setattr(sync, "_import_libsql", lambda: (FakeLibsql, "_check_same_thread"))
    monkeypatch.setattr(sync, "_INITIAL_SYNC_TIMEOUT", 0.3)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    started = time.monotonic()
    try:
        conn, _, mode = sync.open_connection(
            tmp_path / "m.db", SyncConfig(f"http://127.0.0.1:{port}", "tok"))
    finally:
        srv.close()
    assert mode == "degraded" and not connected
    assert time.monotonic() - started < 5
    assert "did not answer" in capsys.readouterr().err
    assert isinstance(conn, sqlite3.Connection)


@pytest.mark.real_preflight
def test_preflight_lets_a_refused_connection_through(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                                    # nothing listens there now
    sync._preflight(f"http://127.0.0.1:{port}", 0.5)  # the client reports this itself
    sync._preflight("file:///nowhere", 0.5)           # not a network URL: no check


def test_unencrypted_sync_url_is_refused_unless_loopback(tmp_path, monkeypatch, capsys):
    calls = []

    class FakeLibsql:
        @staticmethod
        def connect(*a, **k):
            calls.append(k)
            raise RuntimeError("stop here")

    monkeypatch.setattr(sync, "_import_libsql", lambda: (FakeLibsql, "_check_same_thread"))
    monkeypatch.delenv("TETHER_SYNC_ALLOW_INSECURE", raising=False)
    public = ("http://db.example.com", "ws://db.example.com:8080/x", "http://8.8.8.8:8080",
              "http://box.tailnet.ts.net:8080")
    for url in public:
        _, _, mode = sync.open_connection(tmp_path / "a.db", SyncConfig(url, "SECRET-TOKEN"))
        assert mode == "degraded" and not calls
        err = capsys.readouterr().err
        assert "not encrypted" in err and "TETHER_SYNC_ALLOW_INSECURE" in err
        assert "SECRET-TOKEN" not in err
    # this machine, a LAN, a compose service, or anything encrypted: allowed
    allowed = ("http://127.0.0.1:8080", "http://localhost:8080", "http://192.168.1.20:8080",
               "http://10.0.0.5", "http://100.64.1.2:8080", "ws://sqld:8080",
               "http://[::1]:8080", "https://db.example.com", "libsql://db.example.com")
    for url in allowed:
        sync.open_connection(tmp_path / "b.db", SyncConfig(url, "SECRET-TOKEN"))
    assert len(calls) == len(allowed)
    # ...and an explicit opt-in covers a private route tether cannot recognize
    monkeypatch.setenv("TETHER_SYNC_ALLOW_INSECURE", "1")
    sync.open_connection(tmp_path / "c.db", SyncConfig(public[3], "SECRET-TOKEN"))
    assert len(calls) == len(allowed) + 1


def test_sync_failure_log_never_carries_credentials(tmp_path, monkeypatch, capsys):
    url = "libsql://user:hunter2@db.example.com/path?authToken=QUERYSECRET"

    def boom(db_path, cfg):
        raise RuntimeError(f"handshake with {cfg.url} failed (token {cfg.token})")

    monkeypatch.setattr(sync, "_open_replica", boom)
    sync.open_connection(tmp_path / "m.db", SyncConfig(url, "TOKEN-123"))
    err = capsys.readouterr().err
    assert "libsql://db.example.com" in err
    for secret in ("hunter2", "QUERYSECRET", "TOKEN-123"):
        assert secret not in err
    # a stub-length token must not chew up ordinary words in the message
    monkeypatch.setattr(sync, "_open_replica",
                        lambda p, c: (_ for _ in ()).throw(RuntimeError("token broken")))
    sync.open_connection(tmp_path / "n.db", SyncConfig("libsql://db.example.com", "tok"))
    assert "token broken" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_store_files_and_a_new_directory_are_private(tmp_path):
    old = os.umask(0o022)
    try:
        path = tmp_path / "fresh" / "memory.db"
        conn, _, _ = sync.open_connection(path, None)
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
        for entry in os.listdir(path.parent):
            assert stat.S_IMODE(os.stat(path.parent / entry).st_mode) == 0o600, entry
        # an existing directory is left exactly as it was
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o755)
        sync.open_connection(shared / "memory.db", None)
        assert stat.S_IMODE(os.stat(shared).st_mode) == 0o755
        assert stat.S_IMODE(os.stat(shared / "memory.db").st_mode) == 0o600
        # an EXISTING store keeps its owner and group access (a deliberately
        # shared one stays shared) and only loses access for everyone else;
        # a neighbouring file that merely shares the name is not tether's
        os.chmod(shared / "memory.db", 0o664)
        (shared / "memory.db-backup.json").write_text("{}")
        os.chmod(shared / "memory.db-backup.json", 0o644)
        sync.open_connection(shared / "memory.db", None)
        assert stat.S_IMODE(os.stat(shared / "memory.db").st_mode) == 0o660
        assert stat.S_IMODE(os.stat(shared / "memory.db-backup.json").st_mode) == 0o644
    finally:
        os.umask(old)


@pytest.mark.real_preflight
def test_preflight_ignores_proxy_settings_like_the_client_does(monkeypatch):
    # A proxy that swallows the request must not be what gets tested.
    proxy, proxy_port = _silent_listener()
    backend = socket.socket()
    backend.bind(("127.0.0.1", 0))
    backend.listen(1)
    hits = []

    def serve():
        try:
            conn, _ = backend.accept()
            hits.append(conn.recv(2048))
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            conn.close()
        except OSError:
            pass
    threading.Thread(target=serve, daemon=True).start()
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, f"http://127.0.0.1:{proxy_port}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    try:
        sync._preflight(f"http://127.0.0.1:{backend.getsockname()[1]}", 2)
    finally:
        proxy.close()
        backend.close()
    assert hits and hits[0].startswith(b"GET /health")
