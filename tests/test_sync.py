import sqlite3

from tether import sync
from tether.config import SyncConfig


def test_no_config_returns_plain_sqlite(tmp_path):
    conn, sync_now = sync.open_connection(tmp_path / "m.db", None)
    assert isinstance(conn, sqlite3.Connection)
    assert sync_now() is None  # no-op, no error
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    assert conn.execute("SELECT x FROM t").fetchone()[0] == 1


def test_local_connection_sets_wal_and_busy_timeout(tmp_path):
    # #43: recall itself writes (session tracking) when the associative graph
    # is enabled, so concurrent recalls can contend for the write lock. WAL
    # lets readers proceed alongside a writer, and busy_timeout makes a
    # contended write retry instead of failing instantly.
    conn, _ = sync.open_connection(tmp_path / "m.db", None)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == sync._BUSY_TIMEOUT_MS


def test_every_local_connection_gets_the_pragmas(tmp_path, monkeypatch):
    # Same guarantee via the degrade-to-local path (#43 covers "every
    # connection opened in _local()", not just the zero-config startup path).
    def boom(*a, **k):
        raise RuntimeError("cannot reach turso")
    monkeypatch.setattr(sync, "_open_replica", boom)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, _ = sync.open_connection(tmp_path / "m.db", cfg)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == sync._BUSY_TIMEOUT_MS


def test_backend_failure_degrades_to_local(tmp_path, monkeypatch, capsys):
    # Force the replica path to blow up however it likes.
    def boom(*a, **k):
        raise RuntimeError("cannot reach turso")
    monkeypatch.setattr(sync, "_open_replica", boom)

    cfg = SyncConfig("libsql://x.turso.io", "tok")
    conn, sync_now = sync.open_connection(tmp_path / "m.db", cfg)

    # Degraded, not dead: a real local connection and a safe no-op sync.
    assert isinstance(conn, sqlite3.Connection)
    assert sync_now() is None
    conn.execute("CREATE TABLE t(x)")
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    assert "sync offline" in capsys.readouterr().err
