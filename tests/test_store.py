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


class FakeEmbedder:
    """Deterministic 3-axis embedder for hermetic tests: vehicle/food/code.
    Lets us prove semantic recall matches synonyms keyword search misses,
    with no model download and no numpy needed for the write path."""
    name = "fake-3d"
    dims = 3
    _AXES = [
        ("car", "automobile", "vehicle", "drive", "driving"),
        ("pizza", "eat", "food", "meal", "cooking"),
        ("python", "code", "test", "tests", "pytest"),
    ]

    def embed(self, text):
        import math
        t = text.lower()
        v = [float(sum(w in t for w in axis)) for axis in self._AXES]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


def make_semantic_store():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="test-device",
              sync_now=lambda *a, **k: None, embedder=FakeEmbedder())
    s.migrate()
    return s


def test_migrate_adds_embedding_column_and_meta_table():
    s = make_store()  # no embedder
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "embedding" in cols
    tables = {r[0] for r in s._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "meta" in tables


def test_backfill_is_noop_without_embedder():
    s = make_store()
    s.remember("user", "A", "car and driving")
    assert s.backfill_embeddings() == 0
    assert s._conn.execute("SELECT embedding FROM memories").fetchone()[0] is None


def test_backfill_embeds_rows_written_without_a_vector():
    # Row inserted by an embedder-less store, then a later store backfills it.
    conn = sqlite3.connect(":memory:")
    s0 = Store(conn, "d", lambda *a, **k: None)  # embedder is None
    s0.migrate()
    s0.remember("user", "A", "car and driving")
    s1 = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder())
    assert s1.backfill_embeddings() == 1
    blob = conn.execute("SELECT embedding FROM memories").fetchone()[0]
    assert blob is not None and len(blob) == 3 * 4  # 3 float32 values


def test_backfill_resets_when_model_changes():
    s = make_semantic_store()
    s.remember("user", "A", "car")          # embedding still NULL (embed-on-write is Task 4)
    assert s.backfill_embeddings() == 1     # embeds it; records model=fake-3d
    assert s.backfill_embeddings() == 0     # nothing left to embed
    s._meta_set("embedding_model", "a-different-model")
    s._conn.commit()
    assert s.backfill_embeddings() == 1     # model changed -> cleared + re-embedded
    assert s._meta_get("embedding_model") == "fake-3d"


def test_migrate_upgrades_a_populated_pre_embedding_db():
    # Simulate a real v0.1 DB: a `memories` table with NO embedding column and
    # a row already in it. migrate() must ALTER-add the column WITHOUT dropping
    # the row, and backfill must then embed it. (The no-migration promise.)
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " type TEXT NOT NULL CHECK (type IN ('user','feedback','project','reference')),"
        " title TEXT NOT NULL, title_norm TEXT NOT NULL, body TEXT NOT NULL,"
        " tags TEXT NOT NULL DEFAULT '', links TEXT NOT NULL DEFAULT '[]',"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " device_id TEXT NOT NULL DEFAULT '');")
    conn.execute(
        "INSERT INTO memories(type,title,title_norm,body,created_at,updated_at)"
        " VALUES('user','Old','old','a car note','t','t')")
    conn.commit()
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder())
    s.migrate()  # must add the column in place
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert "embedding" in {r[1] for r in
                           conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert s.backfill_embeddings() == 1
    assert conn.execute(
        "SELECT embedding FROM memories WHERE title='Old'").fetchone()[0] is not None


def test_backfill_survives_a_broken_embedder():
    class BrokenEmbedder:
        name = "broken"
        dims = 3

        def embed(self, text):
            raise RuntimeError("model died mid-run")

    conn = sqlite3.connect(":memory:")
    s0 = Store(conn, "d", lambda *a, **k: None)
    s0.migrate()
    s0.remember("user", "A", "car")
    s1 = Store(conn, "d", lambda *a, **k: None, embedder=BrokenEmbedder())
    assert s1.backfill_embeddings() == 0     # degrades, does not raise
    assert conn.execute("SELECT embedding FROM memories").fetchone()[0] is None


def test_remember_stores_embedding_when_embedder_present():
    s = make_semantic_store()
    r = s.remember("user", "A", "I love my car and driving")
    blob = s._conn.execute(
        "SELECT embedding FROM memories WHERE id=?", (r["id"],)).fetchone()[0]
    assert blob is not None and len(blob) == 3 * 4


def test_remember_updates_embedding_on_upsert():
    from tether.store import _unpack
    s = make_semantic_store()
    r = s.remember("user", "A", "car and driving")           # vehicle axis
    r2 = s.remember("user", "A", "pizza and food for lunch")  # same title -> update
    assert r2["id"] == r["id"]
    v = _unpack(s._conn.execute(
        "SELECT embedding FROM memories WHERE id=?", (r["id"],)).fetchone()[0])
    assert v[1] > v[0]  # now weighted to the 'food' axis, not 'vehicle'


def test_remember_leaves_embedding_null_without_embedder():
    s = make_store()  # no embedder
    r = s.remember("user", "A", "car")
    assert s._conn.execute(
        "SELECT embedding FROM memories WHERE id=?", (r["id"],)).fetchone()[0] is None


def test_rrf_fuse_prefers_items_ranked_high_in_both_lists():
    from tether.store import _rrf_fuse
    fused = _rrf_fuse([[1, 2, 3], [2, 5, 1]])
    assert fused[0] == 2          # top-ish in both lists wins
    assert set(fused) == {1, 2, 3, 5}


def test_recall_finds_semantic_synonym_that_keyword_misses():
    pytest.importorskip("numpy")
    s = make_semantic_store()
    car = s.remember("user", "Commute", "I love my car and driving to work")["id"]
    s.remember("project", "Lunch", "pizza and food for the team")
    assert s._fts_ids("automobile") == []      # 'automobile' never appears literally
    hits = s.recall("automobile")
    assert hits and hits[0]["id"] == car
    assert set(hits[0]) == {"id", "type", "title", "body", "tags", "updated_at"}


def test_recall_type_filter_applies_to_semantic_path():
    pytest.importorskip("numpy")
    s = make_semantic_store()
    s.remember("user", "U", "car and driving")
    p = s.remember("project", "P", "car and driving")["id"]
    hits = s.recall("automobile", type="project")
    assert [h["id"] for h in hits] == [p]


def test_recall_degrades_to_keyword_without_embedder():
    s = make_store()  # no embedder
    s.remember("user", "A", "car and driving")
    assert s.recall("automobile") == []        # no semantic -> keyword miss -> empty
    assert len(s.recall("car")) == 1           # keyword still works


def test_recall_degrades_when_numpy_missing(monkeypatch):
    # Embedder present and vectors stored, but numpy is unavailable at query
    # time: the vector path must silently yield to keyword-only recall.
    import sys
    s = make_semantic_store()
    s.remember("user", "A", "car and driving")   # embedded on write (Task 4)
    monkeypatch.setitem(sys.modules, "numpy", None)  # `import numpy` now raises
    assert len(s.recall("car")) == 1           # keyword still works, no crash
    assert s.recall("automobile") == []        # semantic unavailable -> empty, not an error


def test_migrate_adds_consolidation_columns():
    s = make_store()
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert {"author", "valid_from", "valid_to", "superseded_by"} <= cols


def test_migrate_backfills_valid_from_for_existing_rows():
    # A pre-consolidation row (has created_at, no valid_from) gets valid_from set.
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s.migrate()
    s.remember("user", "A", "a note")
    conn.execute("UPDATE memories SET valid_from=NULL")  # simulate a legacy row
    conn.commit()
    s.migrate()  # idempotent + heals valid_from
    vf, ca = conn.execute(
        "SELECT valid_from, created_at FROM memories").fetchone()
    assert vf == ca and vf is not None
