import sqlite3

import pytest

from tether.store import Store


def make_store():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="test-device", sync_now=lambda *a, **k: None)
    s.migrate()
    return s


def test_migrate_creates_tables():
    s = make_store()
    names = {r[0] for r in s._conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')").fetchall()}
    assert "memories" in names
    assert "memories_fts" in names
    assert "memories_ai" in names  # insert trigger keeps FTS in sync


def test_migrate_is_idempotent():
    s = make_store()
    s.migrate()  # second call must not raise
    s.migrate()


def test_remember_inserts_new():
    s = make_store()
    r = s.remember("user", "Prefers TDD", "Wants tests first.")
    assert r["action"] == "created" and isinstance(r["id"], int)
    row = s._conn.execute("SELECT type, title, body, device_id FROM memories WHERE id=?",
                          (r["id"],)).fetchone()
    assert row == ("user", "Prefers TDD", "Wants tests first.", "test-device")


def test_remember_upserts_on_same_type_and_title():
    s = make_store()
    first = s.remember("user", "Prefers TDD", "Wants tests first.")
    again = s.remember("user", "  prefers   tdd ", "Wants tests first, evidence before done.")
    assert again["action"] == "updated"
    assert again["id"] == first["id"]  # same row, not a duplicate
    n = s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert n == 1
    body = s._conn.execute("SELECT body FROM memories WHERE id=?", (first["id"],)).fetchone()[0]
    assert "evidence before done" in body


def test_remember_same_title_different_type_is_distinct():
    s = make_store()
    a = s.remember("user", "Testing", "x")
    b = s.remember("project", "Testing", "y")
    assert a["id"] != b["id"]
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2


def test_remember_rejects_bad_type():
    s = make_store()
    with pytest.raises(ValueError):
        s.remember("nonsense", "t", "b")
