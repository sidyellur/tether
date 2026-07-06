import json as _json
import sqlite3
import tempfile
import threading

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


def test_forget_soft_deletes_and_reports_existence():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    assert s.forget(a) == {"forgotten": a, "existed": True}
    assert s.forget(a) == {"forgotten": a, "existed": False}
    # soft-delete: row retained, just marked no-longer-current
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    valid_to = s._conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (a,)).fetchone()[0]
    assert valid_to is not None
    assert s.recall("A") == []


def test_forget_is_reversible():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    s.forget(a)
    s._conn.execute("UPDATE memories SET valid_to=NULL WHERE id=?", (a,))
    s._conn.commit()
    assert a in [h["id"] for h in s.recall("A")]


def test_forget_keeps_edges_for_reversibility():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s._graph.enabled = True
    s.migrate()
    a = s.remember("user", "A", "x")["id"]
    b = s.remember("project", "B", "y")["id"]
    s.link(a, b)
    s.forget(a)
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] >= 1


def test_forget_nonexistent_id_reports_false():
    s = make_store()
    assert s.forget(9999) == {"forgotten": 9999, "existed": False}


def test_purge_hard_deletes():
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    assert s.purge(a) == {"purged": a, "existed": True}
    assert s.purge(a) == {"purged": a, "existed": False}
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    # deleted rows leave no FTS ghost
    assert s.recall("A") == []


def test_purge_removes_edges():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s._graph.enabled = True
    s.migrate()
    a = s.remember("user", "A", "x")["id"]
    b = s.remember("project", "B", "y")["id"]
    s.link(a, b)
    s.purge(a)
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_export_all_returns_current_memories_as_json_ready_dicts():
    s = make_store()
    a = s.remember("user", "A", "body a", tags="x,y")["id"]
    b = s.remember("project", "B", "body b")["id"]
    s.link(a, b)
    s.forget(b)
    out = s.export_all()
    assert len(out) == 1
    assert out[0]["id"] == a
    assert out[0]["title"] == "A"
    assert out[0]["tags"] == "x,y"
    assert out[0]["links"] == [b]
    _json.dumps(out)  # must be JSON-serializable as-is


def test_export_all_empty_store():
    s = make_store()
    assert s.export_all() == []


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
    key = s._embedding_meta_key("embedding_model")
    s._meta_set(key, "a-different-model")
    s._conn.commit()
    assert s.backfill_embeddings() == 1     # model changed -> cleared + re-embedded
    assert s._meta_get(key) == "fake-3d"


def test_backfill_embedding_model_key_is_per_device_scoped():
    # #45: two "devices" sharing one DB/meta table, each configured with a
    # different embedding model, must each converge after their own first
    # backfill instead of re-wiping and re-embedding forever on every boot.
    class ModelA:
        name, dims = "model-a", 3

        def embed(self, text):
            return [1.0, 0.0, 0.0]

    class ModelB:
        name, dims = "model-b", 3

        def embed(self, text):
            return [0.0, 1.0, 0.0]

    conn = sqlite3.connect(":memory:")
    dev_a = Store(conn, device_id="dev-a", sync_now=lambda *a, **k: None,
                  embedder=ModelA(), author="dev-a")
    dev_a.migrate()
    dev_a.remember("user", "T", "b")
    assert dev_a.backfill_embeddings() == 1    # first run ever: embeds under model-a
    assert dev_a.backfill_embeddings() == 0    # stable: no re-wipe on a second call

    dev_b = Store(conn, device_id="dev-b", sync_now=lambda *a, **k: None,
                  embedder=ModelB(), author="dev-b")
    dev_b.migrate()
    assert dev_b.backfill_embeddings() == 1    # dev-b's own first run: wipes + re-embeds

    # dev-a "reboots" (fresh process, same synced meta table). Its own scoped
    # key still says model-a == its current model, so it must NOT re-wipe -
    # the pre-fix global key would see dev-b's write and re-trigger here.
    dev_a2 = Store(conn, device_id="dev-a", sync_now=lambda *a, **k: None,
                   embedder=ModelA(), author="dev-a")
    assert dev_a2.backfill_embeddings() == 0

    keys = {k: v for k, v in conn.execute("SELECT key, value FROM meta").fetchall()}
    assert keys["embedding_model:dev-a"] == "model-a"
    assert keys["embedding_model:dev-b"] == "model-b"


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


def make_authored_store(author="sid", **kw):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="dev", sync_now=lambda *a, **k: None,
              author=author, **kw)
    s.migrate()
    return s


def test_remember_records_author_and_valid_from():
    s = make_authored_store("sid")
    r = s.remember("user", "A", "a note")
    row = s._conn.execute(
        "SELECT author, valid_from, valid_to, created_at FROM memories WHERE id=?",
        (r["id"],)).fetchone()
    author, valid_from, valid_to, created_at = row
    assert author == "sid"
    assert valid_from == created_at
    assert valid_to is None  # brand-new fact is current


def test_remember_upsert_skips_superseded():
    s = make_authored_store()
    a = s.remember("user", "A", "first")["id"]
    # Manually supersede it (as Task 4 would): mark it not-current.
    s._conn.execute("UPDATE memories SET valid_to='t', superseded_by=999 WHERE id=?", (a,))
    s._conn.commit()
    again = s.remember("user", "A", "second")  # same title, but old one is superseded
    assert again["action"] == "created"        # a fresh current row, not an update
    assert again["id"] != a


def test_remember_action_unchanged_without_consolidate():
    s = make_authored_store()  # consolidate defaults False
    assert s.remember("user", "A", "x")["action"] == "created"
    assert s.remember("user", "A", "y")["action"] == "updated"  # exact-title refine


def make_consolidating_store(threshold=0.92):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="dev", sync_now=lambda *a, **k: None,
              embedder=FakeEmbedder(), author="sid",
              consolidate=True, dedup_threshold=threshold)
    s.migrate()
    return s


def test_consolidate_supersedes_near_duplicate():
    pytest.importorskip("numpy")
    s = make_consolidating_store(threshold=0.9)
    a = s.remember("user", "Commute A", "I drive my car to work")["id"]
    # Different title, same meaning (vehicle axis) -> should consolidate.
    r = s.remember("user", "Commute B", "driving the car every day")
    assert r["action"] == "consolidated"
    old = s._conn.execute(
        "SELECT valid_to, superseded_by FROM memories WHERE id=?", (a,)).fetchone()
    assert old[0] is not None and old[1] == r["id"]   # old row retained + linked
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2  # not deleted


def test_consolidate_keeps_distinct_facts_separate():
    pytest.importorskip("numpy")
    s = make_consolidating_store(threshold=0.9)
    s.remember("user", "Car", "I drive my car")            # vehicle axis
    r = s.remember("user", "Lunch", "pizza and food today")  # food axis, unrelated
    assert r["action"] == "created"                        # NOT merged
    assert s._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL").fetchone()[0] == 2


def test_consolidate_noop_without_embedder():
    # consolidate=True but no embedder -> plain insert, never raises.
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "dev", lambda *a, **k: None, consolidate=True)
    s.migrate()
    s.remember("user", "A", "car")
    assert s.remember("user", "B", "car")["action"] == "created"


def test_rrf_fuse_still_orders_by_combined_rank():
    from tether.store import _rrf_fuse
    fused = _rrf_fuse([[1, 2, 3], [2, 5, 1]])
    assert fused[0] == 2 and set(fused) == {1, 2, 3, 5}


def test_decay_factor_math():
    from tether.store import _decay_factor
    assert _decay_factor(0.0, 30.0) == 1.0
    assert abs(_decay_factor(30.0, 30.0) - 0.5) < 1e-9
    assert _decay_factor(60.0, 30.0) < 0.3


def test_recall_excludes_superseded():
    pytest.importorskip("numpy")
    s = make_consolidating_store(threshold=0.9)
    a = s.remember("user", "Commute A", "I drive my car to work")["id"]
    r = s.remember("user", "Commute B", "driving the car every day")
    assert r["action"] == "consolidated"           # a is now superseded
    hits = s.recall("car")
    ids = [h["id"] for h in hits]
    assert r["id"] in ids and a not in ids          # only the current fact


def test_boot_index_excludes_superseded():
    pytest.importorskip("numpy")
    s = make_consolidating_store(threshold=0.9)
    s.remember("user", "Commute A", "I drive my car to work")
    r = s.remember("user", "Commute B", "driving the car every day")
    lines = s.boot_index().splitlines()
    assert len(lines) == 1 and f"#{r['id']}" in lines[0]


def test_recency_breaks_ties():
    # Two equally-relevant keyword hits; the newer updated_at wins.
    s = make_authored_store()
    old = s.remember("user", "Old", "the keyword apple")["id"]
    new = s.remember("project", "New", "the keyword apple")["id"]
    s._conn.execute("UPDATE memories SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old,))
    s._conn.execute("UPDATE memories SET updated_at='2030-01-01T00:00:00+00:00' WHERE id=?", (new,))
    s._conn.commit()
    hits = s.recall("apple")
    assert [h["id"] for h in hits][0] == new


def test_decay_downranks_old_memories():
    # With decay on, a very old memory is pushed below a fresh one of equal relevance.
    s = make_authored_store(decay_half_life_days=1.0)  # 1-day half-life
    old = s.remember("user", "Old", "the keyword apple")["id"]
    new = s.remember("project", "New", "the keyword apple")["id"]
    s._conn.execute("UPDATE memories SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old,))
    s._conn.commit()
    hits = s.recall("apple")
    assert [h["id"] for h in hits][0] == new


def test_recency_does_not_override_strong_match():
    # A memory that matches BOTH keyword and semantic signals (agreeing at
    # rank 0 in both lists) outranks a memory that only weakly matches -
    # even when the weak match is decades newer. The relevance gap from two
    # agreeing signals is large enough that the gentle 0.25 recency weight
    # (which only ever pulls from a single ranked list) cannot flip it.
    pytest.importorskip("numpy")
    s = make_semantic_store()
    best = s.remember("user", "Best", "I drive my car to work every day")["id"]
    weak = s.remember("reference", "Weak",
                       "a note mostly about pizza and food, "
                       "with one incidental car mention")["id"]
    s._conn.execute("UPDATE memories SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (best,))
    s._conn.execute("UPDATE memories SET updated_at='2030-01-01T00:00:00+00:00' WHERE id=?", (weak,))
    s._conn.commit()
    hits = s.recall("car")
    assert [h["id"] for h in hits][0] == best


def test_remember_writes_semantic_edges_when_assoc_on():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder())
    s._graph.enabled = True                     # force association on (assoc arg lands in Task 6)
    s.migrate()
    s.remember("user", "Commute", "I drive my car to work")
    s.remember("user", "Errand", "driving the automobile downtown")
    n = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='semantic'").fetchone()[0]
    assert n >= 1


def test_link_writes_explicit_edge():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s._graph.enabled = True
    s.migrate()
    a = s.remember("user", "A", "x")["id"]
    b = s.remember("project", "B", "y")["id"]
    s.link(a, b)
    row = conn.execute("SELECT kind, weight FROM edges").fetchone()
    assert row == ("explicit", 1.0)


def make_assoc_store():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, recall_budget=16)
    s.migrate()
    return s


def test_recall_disabled_matches_v2():
    # assoc defaults False -> identical to the v0.2 recall path (no 'via' field).
    s = make_store()  # helper from the existing suite; assoc off
    s.remember("user", "A", "car and driving")
    hits = s.recall("car")
    assert hits and "via" not in hits[0]
    assert set(hits[0]) == {"id", "type", "title", "body", "tags", "updated_at"}


def test_recall_associative_finds_linked_neighbor():
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why not sessions", "sessions were rejected for scaling")["id"]
    s.link(a, b)                                  # explicit edge a<->b
    # 'JWT' matches only A; B is reached across the explicit edge
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert a in ids and b in ids


def test_recall_via_receipts_present():
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why", "the rationale doc")["id"]
    s.link(a, b)
    hits = {h["id"]: h for h in s.recall("JWT tokens", budget=8)}
    assert hits[a]["via"] == {"seed": True}
    assert "path" in hits[b]["via"] and hits[b]["via"]["path"][0]["from"] == a


def test_recall_budget_zero_is_passthrough():
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why", "the rationale doc")["id"]
    s.link(a, b)
    ids = [h["id"] for h in s.recall("JWT tokens", budget=0)]
    assert ids == [a]                             # no spreading -> only the direct match


def test_recall_seed_not_buried_by_high_weight_hebbian_neighbor():
    # #25: a within-task co-recalled neighbor (NOT a query match), reached over a
    # capped Hebbian edge (factor 5.0*0.4=2.0, amplifying), must not outrank the
    # query's own direct hit. The seed-activation floor guarantees this for a
    # single hop (seed_score + floor > one hop's transmit).
    # NOTE: budget=1 caps the walk to the single a->b hop the bug report
    # describes. (With only 2 memories, budget>=2 lets `b` fire back across the
    # same bidirectional edge and re-boost `a`, which masked the bug pre-fix; the
    # floor makes a>b hold at any budget, but budget=1 is the tightest check.)
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]     # matches query
    b = s.remember("project", "Picnic", "quarterly pizza budget review")["id"]  # no match
    s._graph._upsert_edge(a, b, "hebbian", 5.0, "2026-01-01T00:00:00+00:00", mode="max")
    s._conn.commit()
    ids = [h["id"] for h in s.recall("JWT tokens", budget=1)]
    assert a in ids and b in ids
    assert ids.index(a) < ids.index(b)          # seed dominates the spread-reached node


def test_learn_from_head_is_a_knob(monkeypatch):
    # HEBBIAN_LEARN_FROM_HEAD must be reversible like the other B1 knobs: by
    # default (True) only the protected direct-hit head can gain co-recall
    # edges from a recall() call; flipped to False, tail members (reached only
    # via spread/link, never a query match) can gain edges too.
    pytest.importorskip("numpy")
    import tether.graph as graph_mod

    def hebbian_pairs(s):
        return {tuple(sorted((r[0], r[1]))) for r in s._conn.execute(
            "SELECT src, dst FROM edges WHERE kind='hebbian'").fetchall()}

    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Auth rationale", "JWT tokens were chosen for scaling")["id"]
    c = s.remember("project", "Picnic", "quarterly pizza budget review")["id"]
    s.link(a, c)                                   # c never matches the query
    s.recall("JWT tokens", budget=8)
    edges_on = hebbian_pairs(s)
    assert tuple(sorted((a, b))) in edges_on        # both direct hits (head) wired
    assert tuple(sorted((a, c))) not in edges_on    # tail-only neighbor NOT wired
    assert tuple(sorted((b, c))) not in edges_on

    monkeypatch.setattr(graph_mod, "HEBBIAN_LEARN_FROM_HEAD", False)
    s2 = make_assoc_store()
    a2 = s2.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b2 = s2.remember("project", "Auth rationale", "JWT tokens were chosen for scaling")["id"]
    c2 = s2.remember("project", "Picnic", "quarterly pizza budget review")["id"]
    s2.link(a2, c2)
    s2.recall("JWT tokens", budget=8)
    edges_off = hebbian_pairs(s2)
    assert tuple(sorted((a2, c2))) in edges_off     # now the tail neighbor gets wired too


def make_b1_store(assoc=True, **kw):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, assoc=assoc, **kw)
    s.migrate()
    return s


def _add_edge(s, a, b, kind="hebbian", w=1.0):
    lo, hi = (a, b) if a < b else (b, a)
    s._conn.execute("INSERT INTO edges(src, dst, kind, weight, updated_at) "
                    "VALUES (?,?,?,?,?)", (lo, hi, kind, w, "t"))
    s._conn.commit()


def test_boot_index_small_store_unchanged():
    s = make_b1_store(boot_index_cap=50)
    for i in range(3):
        s.remember("user", f"T{i}", "b")
    idx = s.boot_index()
    assert "# Load-bearing" not in idx
    assert len(idx.splitlines()) == 3


def test_boot_index_curates_above_cap_with_hubs():
    s = make_b1_store(boot_index_cap=4)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(8)]
    hub = ids[0]                                  # oldest -> not in the recent reserve
    for other in ids[1:4]:
        _add_edge(s, hub, other, "hebbian", 1.0)
    idx = s.boot_index()
    assert "# Load-bearing" in idx and "# Recent" in idx
    assert f"#{hub} " in idx.split("# Recent")[0]         # hub is in the load-bearing slice
    body = [ln for ln in idx.splitlines() if not ln.startswith("#")]
    assert len(body) <= 4                                 # capped


def test_boot_index_recent_only_when_no_behavioral_hubs():
    s = make_b1_store(boot_index_cap=4)
    for i in range(8):
        s.remember("user", f"T{i}", "b")          # no edges at all
    idx = s.boot_index()
    assert "# Load-bearing" not in idx            # no hubs -> recent-only, no headers
    assert len(idx.splitlines()) == 4             # bounded to cap


def test_boot_index_still_capped_when_graph_disabled():
    # #52: the size cap must apply regardless of graph state - only the
    # curation strategy (hub vs. plain-recency) depends on having a graph.
    s = make_b1_store(assoc=False, boot_index_cap=4)   # graph OFF
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(8)]
    lines = s.boot_index().splitlines()
    assert len(lines) == 4                             # still capped
    assert "# Load-bearing" not in lines[0]             # no hub curation, just recency
    newest_four = list(reversed(ids))[:4]
    assert [f"#{mid}" in ln for mid, ln in zip(newest_four, lines)] == [True] * 4


_OLD = "2020-01-01T00:00:00+00:00"


def make_forget_store(**kw):
    kw.setdefault("boot_index_cap", 2)            # size floor = 2*2 = 4
    kw.setdefault("forget_max_per_sweep", 10)
    return make_b1_store(assoc=True, forget=True, **kw)


def _age(s, mid, iso=_OLD):
    s._conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (iso, mid))
    s._conn.commit()


def test_forgetting_archives_old_isolated():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")       # live behavioral graph elsewhere
    _age(s, ids[0])                                # old + isolated
    assert s._run_forgetting_sweep() == 1
    vt, sb = s._conn.execute(
        "SELECT valid_to, superseded_by FROM memories WHERE id=?", (ids[0],)).fetchone()
    assert vt is not None and sb is None           # archived, not superseded
    assert ids[0] not in [h["id"] for h in s.recall("T0")]


def test_forgetting_keeps_old_but_connected():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[0], ids[1], "explicit")       # behaviorally connected
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_keeps_isolated_but_recent():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    # ids[0] isolated but NOT aged -> kept
    assert s._run_forgetting_sweep() == 0


def test_forgetting_semantic_only_does_not_protect():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")        # live behavioral graph
    _add_edge(s, ids[0], ids[1], "semantic")       # ids[0] has ONLY a semantic edge
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 1          # semantic doesn't protect


def test_forgetting_noop_when_disabled():
    s = make_b1_store(assoc=True, forget=False, boot_index_cap=2)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_noop_without_behavioral_graph():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _age(s, ids[0])                                # old + isolated, but NO behavioral edges anywhere
    assert s._run_forgetting_sweep() == 0


def test_forgetting_respects_size_floor():
    s = make_forget_store()                        # cap=2 -> floor 4
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(3)]   # only 3 < 4
    _add_edge(s, ids[1], ids[2], "hebbian")
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_bounded_per_sweep():
    s = make_forget_store(forget_max_per_sweep=2)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(8)]
    _add_edge(s, ids[6], ids[7], "hebbian")        # keep two connected (live graph)
    for i in range(6):
        _age(s, ids[i])                            # 6 old + isolated
    assert s._run_forgetting_sweep() == 2          # capped


def test_forgetting_is_reversible():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])
    s._run_forgetting_sweep()
    s._conn.execute("UPDATE memories SET valid_to=NULL WHERE id=?", (ids[0],))
    s._conn.commit()
    assert ids[0] in [h["id"] for h in s.recall("T0")]           # un-forgotten
    assert s._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] >= 1   # edges retained


def test_forget_trigger_fires_at_interval():
    s = make_forget_store(forget_interval=3)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])                                # now old + isolated
    before = s._conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (ids[0],)).fetchone()[0]
    for i in range(3):                             # 3 writes -> counter hits interval
        s.remember("user", f"X{i}", "b")
    after = s._conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (ids[0],)).fetchone()[0]
    assert before is None and after is not None    # the trigger archived it


def test_forget_trigger_disabled_never_fires():
    s = make_b1_store(assoc=True, forget=False, boot_index_cap=2)
    for i in range(6):
        s.remember("user", f"T{i}", "b")
    assert s._conn.execute(
        "SELECT value FROM meta WHERE key='forget_counter'").fetchone() is None


def test_seed_floor_excludes_low_cosine_from_seeds():
    # #15: a memory whose cosine to the query is below the seed floor must not
    # be seeded (it should be reachable only via edges, not as a near-tied
    # whole-store seed). Query "automobile" (axis0) has no lexical overlap with
    # either body, so ONLY the vector arm can seed them -> the floor is decisive.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              seed_floor=0.5)
    s.migrate()
    keep = s.remember("user", "keep", "car")["id"]            # cos=1.0  >= 0.5
    drop = s.remember("user", "drop", "car pizza food")["id"] # cos=0.447 < 0.5
    seeds = s._seed_scores("automobile", None)
    assert keep in seeds
    assert drop not in seeds


def test_seed_floor_zero_keeps_all_vector_hits():
    # Floor at 0 reproduces the pre-#15 behavior: every embedded row is a seed.
    # Proves the floor is what excludes the low-cosine row, nothing else.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              seed_floor=0.0)
    s.migrate()
    keep = s.remember("user", "keep", "car")["id"]
    drop = s.remember("user", "drop", "car pizza food")["id"]
    seeds = s._seed_scores("automobile", None)
    assert keep in seeds and drop in seeds


def test_remember_crystallizes_links_sources_when_enabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    a = s.remember("project", "auth outage", "login 500s under load")["id"]
    b = s.remember("project", "pool fix", "raised the connection pool ceiling")["id"]
    p = s.remember("reference", "principle: fail fast on saturation",
                   "cap the pool and time out", crystallizes=[a, b])["id"]
    rows = conn.execute(
        "SELECT src, dst FROM edges WHERE kind='crystallized' ORDER BY dst").fetchall()
    assert rows == [(p, a), (p, b)]


def test_remember_crystallizes_ignored_when_disabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=False)          # feature off
    s.migrate()
    a = s.remember("project", "x", "y")["id"]
    s.remember("reference", "p", "z", crystallizes=[a])
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='crystallized'").fetchone()[0] == 0


def test_crystallization_candidates_empty_when_disabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=False)
    s.migrate()
    assert s.crystallization_candidates() == []


def test_crystallization_candidates_memoized_until_write():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    first = s.crystallization_candidates()
    # same signature -> same object identity (cache hit, no recompute)
    assert s.crystallization_candidates() is first


def test_dismiss_invalidates_candidate_memo():
    # Regression: dismiss_cluster writes crystallize_dismissed (not edges), so an
    # edges-only memo signature would NOT recompute and the dismissal would no-op.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    s.crystallization_candidates()                  # populate the memo signature
    sig_before = s._cryst_sig
    s.dismiss_cluster(1, 2)                          # writes crystallize_dismissed
    s.crystallization_candidates()                  # must recompute
    assert s._cryst_sig != sig_before               # signature reflects the dismissal


def test_crystallized_edge_surfaces_principle_from_source():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True, recall_budget=16)
    s.migrate()
    a = s.remember("project", "Auth", "we switched to JWT tokens")["id"]
    p = s.remember("reference", "Principle", "fail fast under load",
                   crystallizes=[a])["id"]
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert a in ids and p in ids                    # principle reached from its source


def test_crystallized_hub_does_not_bury_direct_hit():
    # #25 back-door: a max-fan-out principle must not outrank a query's own hit.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True, recall_budget=16)
    s.migrate()
    hits = [s.remember("project", f"n{i}", "quarterly pizza budget review")["id"]
            for i in range(6)]
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]  # the direct hit
    p = s.remember("reference", "Principle", "a big fan-out principle",
                   crystallizes=hits + [a])["id"]   # hub over everything incl. a
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert ids[0] == a                              # seed still dominates the hub


# ---------------------------------------------------------------------------
# #41: concurrent remember() with the same (type, title) must not duplicate
# #47: remember() must not clobber `links` when links isn't re-passed
# ---------------------------------------------------------------------------

def _file_store(path, **kwargs):
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    s = Store(conn, device_id="test-device", sync_now=lambda *a, **k: None, **kwargs)
    s.migrate()
    return s


def test_migrate_creates_a_unique_partial_dedup_index():
    s = make_store()
    assert s._has_unique_dedup_index is True
    row = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_memories_dedup'").fetchone()
    assert "UNIQUE" in row[0].upper()


def test_migrate_upgrades_a_preexisting_plain_dedup_index():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " type TEXT NOT NULL CHECK (type IN ('user','feedback','project','reference')),"
        " title TEXT NOT NULL, title_norm TEXT NOT NULL, body TEXT NOT NULL,"
        " tags TEXT NOT NULL DEFAULT '', links TEXT NOT NULL DEFAULT '[]',"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " device_id TEXT NOT NULL DEFAULT '');"
        "CREATE INDEX idx_memories_dedup ON memories(type, title_norm);")
    conn.commit()
    s = Store(conn, "d", lambda *a, **k: None)
    s.migrate()
    assert s._has_unique_dedup_index is True
    row = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_memories_dedup'").fetchone()
    assert "UNIQUE" in row[0].upper()


def test_migrate_degrades_gracefully_with_preexisting_duplicate_rows():
    # Simulate a live DB that already has two "current" rows for the same
    # (type, title_norm) from before this fix - creating the unique index
    # must not crash migrate(); it should warn and fall back instead.
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " type TEXT NOT NULL CHECK (type IN ('user','feedback','project','reference')),"
        " title TEXT NOT NULL, title_norm TEXT NOT NULL, body TEXT NOT NULL,"
        " tags TEXT NOT NULL DEFAULT '', links TEXT NOT NULL DEFAULT '[]',"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " device_id TEXT NOT NULL DEFAULT '');"
        "CREATE INDEX idx_memories_dedup ON memories(type, title_norm);")
    conn.execute(
        "INSERT INTO memories(type,title,title_norm,body,created_at,updated_at)"
        " VALUES('user','Dup','dup','one','t1','t1')")
    conn.execute(
        "INSERT INTO memories(type,title,title_norm,body,created_at,updated_at)"
        " VALUES('user','Dup','dup','two','t2','t2')")
    conn.commit()
    s = Store(conn, "d", lambda *a, **k: None)
    with pytest.warns(RuntimeWarning):
        s.migrate()                                  # must not raise
    assert s._has_unique_dedup_index is False
    # remember() must still work via the locking fallback, not crash.
    r = s.remember("user", "Dup", "three")
    assert r["action"] == "updated"


def test_remember_concurrent_same_title_yields_one_current_row(tmp_path):
    db_path = tmp_path / "memory.db"
    s1 = _file_store(db_path)
    s2 = _file_store(db_path)

    barrier = threading.Barrier(2)
    results = {}

    def call(store, key, body):
        barrier.wait(timeout=5)
        results[key] = store.remember("user", "Race Title", body)

    t1 = threading.Thread(target=call, args=(s1, "a", "from thread A"))
    t2 = threading.Thread(target=call, args=(s2, "b", "from thread B"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert "a" in results and "b" in results          # neither call raised/hung
    check = sqlite3.connect(str(db_path))
    total = check.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    current = check.execute(
        "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL "
        "AND type='user' AND title_norm='race title'").fetchone()[0]
    assert total == 1                                 # no duplicate row was created
    assert current == 1


def test_remember_without_links_preserves_previous_links():
    s = make_store()
    a = s.remember("user", "A", "first body")["id"]
    b = s.remember("user", "B", "other")["id"]
    s.link(a, b)
    before = _json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (a,)).fetchone()[0])
    assert b in before

    again = s.remember("user", "A", "refined body")   # no links= passed
    assert again["id"] == a
    after = _json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (a,)).fetchone()[0])
    assert b in after                                 # link survives the re-remember


def test_remember_with_links_unions_rather_than_replaces():
    s = make_store()
    a = s.remember("user", "A", "x")["id"]
    b = s.remember("user", "B", "y")["id"]
    c = s.remember("user", "C", "z")["id"]

    s.remember("user", "A", "x2", links=[b])
    s.remember("user", "A", "x3", links=[c])          # must union with b, not replace it

    links = _json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (a,)).fetchone()[0])
    assert set(links) == {b, c}


def test_session_sweep_trigger_cleans_abandoned_session_members():
    # #48: a session that's never touched again leaves its session_members
    # rows behind forever (decay/cleanup are keyed on that specific session
    # id, which never runs again). The periodic sweep must reap it anyway.
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, assoc=True, session_sweep_interval=3)
    s.migrate()
    m = s.remember("user", "T0", "body")["id"]
    conn.execute(
        "INSERT INTO session_members VALUES('abandoned', ?, 0.9, '2000-01-01T00:00:00+00:00')",
        (m,))
    conn.commit()
    for i in range(3):                              # 3 recalls -> counter hits interval
        s.recall("T0")
    remaining = {r[0] for r in conn.execute(
        "SELECT session_id FROM session_members").fetchall()}
    assert "abandoned" not in remaining


def test_session_sweep_noop_when_graph_disabled():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, assoc=False, session_sweep_interval=1)
    s.migrate()
    s.remember("user", "T0", "body")
    s.recall("T0")
    assert conn.execute(
        "SELECT value FROM meta WHERE key='session_sweep_counter'").fetchone() is None


def test_recall_concurrent_processes_do_not_hebbian_wire_unrelated_topics():
    # #53: two "processes" (e.g. two parallel subagents sharing one DB) each
    # recalling their own unrelated topic, with no explicit session, in the
    # same instant must not get spuriously Hebbian-wired via a shared
    # implicit time-bucket session.
    conn = sqlite3.connect(":memory:")
    s1 = Store(conn, "d", lambda *a, **k: None, assoc=True, recall_budget=16)
    s1.migrate()
    s2 = Store(conn, "d", lambda *a, **k: None, assoc=True, recall_budget=16)
    a = s1.remember("user", "cars", "I drive my car to work")["id"]
    b = s1.remember("user", "pizza", "pizza night with friends")["id"]
    s1.recall("cars")            # process 1, implicit session
    s2.recall("pizza")           # process 2, implicit session, same instant
    count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert count == 0
    ids = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM session_members")}
    assert len(ids) == 2                             # kept in separate session buckets
