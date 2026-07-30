import sqlite3

from tether import sync
from tether.config import SyncConfig


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
