import json as _json
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


def test_recall_matches_on_body_and_title():
    s = make_store()
    s.remember("user", "Prefers TDD", "Wants tests written first.")
    s.remember("project", "cleat", "Headless terminal layer for agents.")
    hits = s.recall("tests")
    assert len(hits) == 1
    assert hits[0]["title"] == "Prefers TDD"
    assert set(hits[0]) == {"id", "type", "title", "body", "tags", "updated_at"}


def test_recall_filters_by_type():
    s = make_store()
    s.remember("user", "Testing habits", "Likes pytest.")
    s.remember("project", "Testing infra", "pytest in CI.")
    assert len(s.recall("pytest")) == 2
    only = s.recall("pytest", type="project")
    assert len(only) == 1 and only[0]["type"] == "project"


def test_recall_empty_query_returns_empty():
    s = make_store()
    s.remember("user", "x", "y")
    assert s.recall("   ") == []


def test_recall_tolerates_punctuation():
    s = make_store()
    s.remember("user", "C++ notes", "Uses C++ and pytest.")
    # A raw MATCH of 'C++' would be an FTS5 syntax error; must not raise.
    hits = s.recall("C++")
    assert isinstance(hits, list)


def test_link_is_bidirectional_and_idempotent():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    b = s.remember("project", "B", "b")["id"]
    s.link(a, b)
    s.link(a, b)  # idempotent
    la = _json.loads(s._conn.execute("SELECT links FROM memories WHERE id=?", (a,)).fetchone()[0])
    lb = _json.loads(s._conn.execute("SELECT links FROM memories WHERE id=?", (b,)).fetchone()[0])
    assert la == [b] and lb == [a]


def test_link_missing_id_raises():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    with pytest.raises(ValueError):
        s.link(a, 9999)


def test_forget_deletes_and_reports_existence():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    assert s.forget(a) == {"forgotten": a, "existed": True}
    assert s.forget(a) == {"forgotten": a, "existed": False}
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    # deleted rows leave no FTS ghost
    assert s.recall("A") == []


def test_boot_index_lists_newest_first():
    s = make_store()
    assert s.boot_index() == "(no memories yet)"
    a = s.remember("user", "First", "x")["id"]
    b = s.remember("project", "Second", "y")["id"]
    lines = s.boot_index().splitlines()
    assert lines[0] == f"[project] #{b} Second"
    assert lines[1] == f"[user] #{a} First"
