import json as _json
import sqlite3
import threading

import pytest

from tether import sync
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


# --- #89: keyword recall must work for question-shaped queries --------------

def test_natural_language_query_matches_partial_words():
    """#89: the FTS query used to AND every token, so a memory had to contain
    EVERY word of the query. A question like this matched nothing for 99.7%
    of LoCoMo questions and the keyword arm was silent. OR semantics: a memory
    that contains SOME of the words is a hit."""
    s = make_store()
    s.remember("project", "Test runner", "We settled on pytest; it runs the tests.")
    # shares only "runs" and "tests" with the memory; under AND this was []
    hits = s.recall("when did we decide which framework runs the tests?")
    assert [h["title"] for h in hits] == ["Test runner"]


def test_memory_matching_more_query_words_ranks_first():
    s = make_store()
    s.remember("project", "Partial", "pytest is installed")
    s.remember("project", "Full", "pytest runs the whole test suite in CI")
    hits = s.recall("pytest test suite CI")
    assert hits[0]["title"] == "Full"


def test_stopword_only_query_still_searches():
    """Dropping stop-words must never empty a query that was all stop-words:
    fall back to searching the words as given."""
    s = make_store()
    s.remember("user", "The band", "They call themselves The The.")
    assert [h["title"] for h in s.recall("the")] == ["The band"]


def test_fts_query_shape():
    from tether.store import _fts_query
    assert _fts_query("   ") is None
    assert _fts_query("?? --") is None                       # no word characters
    assert _fts_query("pytest") == '"pytest"'
    assert _fts_query("when did we pick pytest?") == '"pick" OR "pytest?"'
    assert _fts_query('say "hi"') == '"say" OR """hi"""'     # quotes still escaped


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


def test_forget_unprimes_session_members():
    # #42: forget() is a third valid_to-setting transition (alongside
    # consolidation and the forgetting sweep) - it must scrub session_members
    # too, or a soft-forgotten memory keeps inflating a session's priming.
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s._graph.enabled = True
    s.migrate()
    a = s.remember("user", "A", "x")["id"]
    s._graph.touch_session("sess1", [a])
    assert a in s._graph.session_activation("sess1")
    s.forget(a)
    assert a not in s._graph.session_activation("sess1")


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


def make_assoc_consolidating_store(threshold=0.9):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="dev", sync_now=lambda *a, **k: None,
              embedder=FakeEmbedder(), author="sid",
              consolidate=True, dedup_threshold=threshold,
              assoc=True, recall_budget=16)
    s.migrate()
    return s


def test_hydrate_filters_out_valid_to_rows():
    # #42 blanket safety net: _hydrate must never return a non-current row,
    # even if asked for it by id directly.
    s = make_store()
    a = s.remember("user", "A", "a")["id"]
    s._conn.execute("UPDATE memories SET valid_to='t' WHERE id=?", (a,))
    s._conn.commit()
    assert s._hydrate([a]) == []


def test_consolidate_unprimes_superseded_memory():
    # #42: once `a` is primed into a session, superseding it via consolidation
    # must scrub the session_members row so a later unrelated recall in the
    # same session can't resurface it (mislabeled via={"seed": True}).
    pytest.importorskip("numpy")
    s = make_assoc_consolidating_store()
    a = s.remember("user", "Commute A", "I drive my car to work")["id"]
    s.recall("car", session="sess1")            # primes `a` into session_members
    assert a in s._graph.session_activation("sess1")
    r = s.remember("user", "Commute B", "driving the car every day")
    assert r["action"] == "consolidated"         # supersedes `a`
    assert a not in s._graph.session_activation("sess1")
    other = s.remember("project", "Lunch", "pizza and food for the team")["id"]
    hits = s.recall("pizza", session="sess1")
    ids = [h["id"] for h in hits]
    assert other in ids
    assert a not in ids                          # superseded memory never resurfaces


def test_forgetting_sweep_unprimes_archived_memory():
    # #42: the forgetting sweep must scrub session_members too, mirroring
    # what consolidation and on_forget already do.
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")      # live behavioral graph elsewhere
    _age(s, ids[0])                               # old + isolated -> archived
    s._graph.touch_session("sess1", [ids[0]])
    assert ids[0] in s._graph.session_activation("sess1")
    assert s._run_forgetting_sweep() == 1
    assert ids[0] not in s._graph.session_activation("sess1")


def test_recall_empty_seeds_does_not_return_primed_context():
    # #46: a query that matches nothing must return [] even when the session
    # has previously-primed members - it must NOT fall back to returning
    # those primed members mislabeled via={"seed": True}.
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    s.recall("JWT tokens", session="sess1")       # primes `a`
    assert a in s._graph.session_activation("sess1")
    hits = s.recall("nothing here matches anything at all", session="sess1")
    assert hits == []


def test_tags_match_is_exact_not_substring():
    from tether.store import _tags_match
    assert _tags_match("proj:tether,other", ["proj:tether"])
    assert not _tags_match("proj:tether2", ["proj:tether"])   # #50: no LIKE-style match
    assert _tags_match("a,b,c", ["a", "c"])
    assert not _tags_match("a,b", ["a", "c"])


def test_parse_tags_handles_str_list_and_none():
    from tether.store import _parse_tags
    assert _parse_tags("a, b ,c") == ["a", "b", "c"]
    assert _parse_tags(["a", " b "]) == ["a", "b"]
    assert _parse_tags(None) == []
    assert _parse_tags("") == []


def test_recall_tags_filter_is_exact_not_substring():
    s = make_store()
    a = s.remember("user", "A", "note", tags="proj:tether")["id"]
    b = s.remember("user", "B", "note", tags="proj:tether2")["id"]
    hits = s.recall("note", tags="proj:tether")
    ids = [h["id"] for h in hits]
    assert a in ids and b not in ids


def test_recall_tags_standalone_without_query():
    # #50: tags alone (no query) is a guaranteed-complete, exact-match lookup.
    s = make_store()
    a = s.remember("user", "A", "x", tags="blog-journal,proj:tether")["id"]
    s.remember("user", "B", "y", tags="proj:tether")["id"]     # missing blog-journal
    c = s.remember("project", "C", "z", tags="blog-journal,proj:tether")["id"]
    hits = s.recall("", tags="blog-journal,proj:tether")
    assert {h["id"] for h in hits} == {a, c}


def test_recall_no_query_no_tags_returns_empty():
    s = make_store()
    s.remember("user", "x", "y")
    assert s.recall("") == []
    assert s.recall(None) == []


def test_recall_tags_combined_with_query_narrows_ranked_hits():
    s = make_store()
    a = s.remember("user", "Apple note", "apple pie recipe", tags="cooking")["id"]
    s.remember("user", "Apple gadget", "apple watch review", tags="tech")["id"]
    hits = s.recall("apple", tags="cooking")
    assert [h["id"] for h in hits] == [a]


def test_recall_tags_filters_associative_seeds_too():
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens", tags="infra")["id"]
    s.remember("project", "Auth rationale", "JWT tokens were chosen", tags="other")["id"]
    hits = s.recall("JWT tokens", tags="infra", budget=8)
    ids = [h["id"] for h in hits]
    assert ids == [a]


def test_recall_tags_filters_spread_reached_tail_too():
    # The tag filter must hold for the WHOLE result, not just the seed tier -
    # `b` is reached only via the explicit link (never matches the query
    # itself), so it lands in the associative tail, not `seeds`. It must still
    # be excluded when its tags don't satisfy the filter.
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens", tags="infra")["id"]
    b = s.remember("project", "Picnic", "quarterly pizza budget review",
                   tags="other")["id"]
    s.link(a, b)
    unfiltered_ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert b in unfiltered_ids                    # sanity: b is reachable at all
    filtered_ids = [h["id"] for h in s.recall("JWT tokens", tags="infra", budget=8)]
    assert a in filtered_ids and b not in filtered_ids


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


class _DeadReplica:
    """Stands in for a sync replica connection whose WRITES fail (#44 - e.g.
    a mid-session network drop) while reads keep working -- an embedded
    replica serves reads from the local file, so a dropped connection to the
    remote primary plausibly still lets a caller like link() read current
    state before its write fails."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, sql, *a, **k):
        if sql.strip().split(None, 1)[0].upper() in ("SELECT", "PRAGMA"):
            return self._real.execute(sql, *a, **k)
        raise RuntimeError("replica unreachable")

    def executescript(self, *a, **k):
        raise RuntimeError("replica unreachable")

    def commit(self):
        raise RuntimeError("replica unreachable")


def _make_degradable_store(tmp_path):
    """A store with a real local file (schema migrated, ready to seed rows
    against the working connection) and db_path wired up so it can degrade
    to that same file, matching what server.py wires up for a replica.
    Swap in a _DeadReplica AFTER seeding any rows a test needs."""
    db_path = tmp_path / "m.db"
    conn, sync_now, _mode = sync._local(db_path)
    s = Store(conn, "d", sync_now, db_path=db_path)
    s.migrate()
    return s, db_path


def _kill_replica(s):
    dead = _DeadReplica(s._conn)
    s._conn = dead
    s._graph._conn = dead


def test_remember_degrades_to_local_on_replica_write_failure(tmp_path, capsys):
    s, db_path = _make_degradable_store(tmp_path)
    _kill_replica(s)
    result = s.remember("user", "A", "a note")
    assert result["action"] == "created"
    assert s._degraded is True
    assert isinstance(s._conn, sqlite3.Connection)
    assert "degrading to local-only" in capsys.readouterr().err

    # the write actually landed on the local file, not lost
    conn2 = sqlite3.connect(db_path)
    assert conn2.execute("SELECT title FROM memories WHERE title='A'").fetchone()

    # subsequent writes go straight to the (already-local) connection
    result2 = s.remember("user", "B", "another note")
    assert result2["action"] == "created"


def test_link_degrades_to_local_on_replica_write_failure(tmp_path):
    s, _ = _make_degradable_store(tmp_path)
    # seed both memories on the working connection first, then take it down
    a = s.remember("user", "A", "a note")["id"]
    b = s.remember("user", "B", "b note")["id"]
    _kill_replica(s)

    result = s.link(a, b)
    assert result == {"linked": [a, b]}
    assert s._degraded is True
    assert isinstance(s._conn, sqlite3.Connection)


def test_forget_degrades_to_local_on_replica_write_failure(tmp_path):
    s, _ = _make_degradable_store(tmp_path)
    mid = s.remember("user", "A", "a note")["id"]
    _kill_replica(s)

    result = s.forget(mid)
    assert result == {"forgotten": mid, "existed": True}
    assert s._degraded is True
    assert isinstance(s._conn, sqlite3.Connection)


def test_no_db_path_means_failures_still_raise():
    # A plain local-only store (db_path=None, the default) has nothing to
    # degrade to -- a write failure must surface, not be silently swallowed.
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s.migrate()
    s._conn = _DeadReplica(conn)
    with pytest.raises(RuntimeError):
        s.remember("user", "A", "a note")
    assert s._degraded is False


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


def test_remember_concurrent_new_title_unions_links_not_last_writer_wins(
        monkeypatch, tmp_path):
    # Regression test for a gap in the #41/#47 fix: `_upsert_via_conflict`'s
    # probe SELECT ("does a current row already exist?") can itself be raced.
    # Two connections both remembering a genuinely brand-new (type, title)
    # at the same instant both legitimately see `existing=None`; only one
    # actually performs the INSERT; the other's statement resolves via the
    # ON CONFLICT DO UPDATE branch instead. The links merge must still union
    # with whatever the winner just wrote, not silently replace it with only
    # the loser's own `links` - that would reintroduce #47's clobber inside
    # the very race #41 closes. Forces the interleaving deterministically
    # (rather than relying on OS thread-scheduling luck) by gating both
    # threads on a barrier right after their probe SELECT.
    db_path = tmp_path / "memory.db"
    s1 = _file_store(db_path)
    s2 = _file_store(db_path)

    barrier = threading.Barrier(2)
    orig = Store._upsert_via_conflict

    def gated(self, *a, **kw):
        barrier.wait(timeout=5)
        return orig(self, *a, **kw)

    monkeypatch.setattr(Store, "_upsert_via_conflict", gated)

    results = {}

    def call(store, key, links):
        results[key] = store.remember("user", "Race Title", f"body {key}", links=links)

    t1 = threading.Thread(target=call, args=(s1, "a", [1]))
    t2 = threading.Thread(target=call, args=(s2, "b", [2]))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert "a" in results and "b" in results
    check = sqlite3.connect(str(db_path))
    row = check.execute(
        "SELECT id, links FROM memories WHERE valid_to IS NULL "
        "AND type='user' AND title_norm='race title'").fetchone()
    assert row is not None
    links = set(_json.loads(row[1]))
    assert links == {1, 2}  # both racers' links survive - neither got dropped


# --- #62: reads pull too, debounced ------------------------------------------

class _RecordingSync:
    """Stands in for sync_now, recording the timeout each call was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, timeout=2.0):
        self.calls.append(timeout)


def _sync_store(interval=30, **kw):
    conn = sqlite3.connect(":memory:")
    rec = _RecordingSync()
    s = Store(conn, "d", rec, sync_read_interval=interval, **kw)
    s.migrate()
    return s, rec


def test_recall_pulls_before_reading():
    """#62: sync_now only ran after writes, so a device that merely READS never
    saw other devices' updates - it was stuck at its own startup probe until it
    happened to write something."""
    s, rec = _sync_store()
    s.remember("user", "A", "body a")
    write_calls = len(rec.calls)
    assert write_calls >= 1                       # the write itself synced

    s._last_sync_at = None                        # simulate time having passed
    s.recall("body")
    assert len(rec.calls) == write_calls + 1, "recall did not pull"


def test_boot_index_pulls_before_reading():
    s, rec = _sync_store()
    s.remember("user", "A", "body a")
    before = len(rec.calls)
    s._last_sync_at = None
    s.boot_index()
    assert len(rec.calls) == before + 1


def test_read_sync_is_debounced():
    """A burst of reads must pull once, not once per read."""
    s, rec = _sync_store(interval=30)
    s._last_sync_at = None
    s.recall("anything")
    after_first = len(rec.calls)
    for _ in range(5):
        s.recall("anything")
    assert len(rec.calls) == after_first, "every read pulled; debounce is not working"


def test_writes_reset_the_read_debounce():
    """A chatty writer already syncs on every write; reads shouldn't then pull
    again on top of that."""
    s, rec = _sync_store(interval=30)
    s._last_sync_at = None
    s.recall("x")                                 # arms the debounce
    before = len(rec.calls)
    s.remember("user", "B", "body b")             # write syncs, resets the clock
    after_write = len(rec.calls)
    assert after_write > before
    s.recall("x")
    assert len(rec.calls) == after_write, "read pulled despite a just-synced write"


def test_read_sync_uses_a_short_timeout():
    """recall is the latency-visible path: a read-path pull must be bounded far
    below the 2.0s write default, since the background pull keeps going and
    lands for the next read anyway."""
    from tether.store import _READ_SYNC_TIMEOUT

    s, rec = _sync_store()
    s._last_sync_at = None
    s.recall("x")
    assert rec.calls[-1] == _READ_SYNC_TIMEOUT
    assert _READ_SYNC_TIMEOUT < 2.0


def test_read_sync_can_be_disabled():
    s, rec = _sync_store(interval=0)
    s._last_sync_at = None
    s.recall("x")
    s.boot_index()
    assert rec.calls == [], "interval 0 must restore write-only syncing"


def test_read_sync_failure_never_breaks_the_read():
    """Degrade-never: an unreachable backend must not turn a recall into an
    error - it serves local data, exactly as it did before reads pulled."""
    conn = sqlite3.connect(":memory:")

    def boom(timeout=2.0):
        raise RuntimeError("backend on fire")

    s = Store(conn, "d", boom, sync_read_interval=30)
    s.migrate()
    conn.execute(
        "INSERT INTO memories(type,title,title_norm,body,tags,links,"
        "created_at,updated_at,device_id,valid_from) "
        "VALUES('user','A','a','findable','','[]','2026-01-01','2026-01-01','d','2026-01-01')")
    conn.commit()
    s._last_sync_at = None
    hits = s.recall("findable")
    assert [h["title"] for h in hits] == ["A"]
    assert s.boot_index() != ""


# --- #61: cached embedding matrix -------------------------------------------

def _cache_store(**kw):
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              sync_read_interval=0, **kw)
    s.migrate()
    return s


def test_embedding_matrix_is_cached_between_reads():
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    s.recall("vehicle")
    cached = s._emb_cache
    assert cached is not None
    s.recall("vehicle")
    assert s._emb_cache is cached, "second read rebuilt the matrix"


def test_write_keeps_the_cache_current_without_a_rescan():
    """#81: a write used to drop the cache, so every remember re-read every
    embedding in the store to rebuild it for kNN wiring. Now the row is
    patched in: after the write the cache is still there, contains the new
    id, and neither the write nor the next read touched the embedding scan."""
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    s.recall("vehicle")
    assert s._emb_cache is not None
    scans = []
    s._conn.set_trace_callback(
        lambda sql: scans.append(sql) if "SELECT id, embedding, type" in sql else None)
    try:
        pid = s.remember("user", "Pizza", "I eat pizza")["id"]
        s.recall("food")
    finally:
        s._conn.set_trace_callback(None)
    assert scans == [], f"write or read rebuilt the matrix from SQL: {scans}"
    ids, mat, _types = s._emb_cache
    assert pid in ids and mat.shape[0] == 2


def test_new_memory_is_semantically_findable_immediately():
    """The invalidation actually matters: a cached matrix that survived a write
    would make the new memory invisible to semantic recall."""
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    s.recall("vehicle")                       # warm the cache
    s.remember("user", "Pizza", "I eat pizza at every meal")
    hits = s.recall("food")
    assert "Pizza" in [h["title"] for h in hits]


def test_forgotten_memory_leaves_semantic_results_immediately():
    """The dangerous direction: a stale matrix would keep serving a memory the
    user just deleted."""
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    pid = s.remember("user", "Pizza", "I eat pizza at every meal")["id"]
    assert "Pizza" in [h["title"] for h in s.recall("food")]
    s.forget(pid)
    assert "Pizza" not in [h["title"] for h in s.recall("food")]


def test_purged_memory_leaves_semantic_results_immediately():
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    pid = s.remember("user", "Pizza", "I eat pizza at every meal")["id"]
    s.recall("food")
    s.purge(pid)
    assert "Pizza" not in [h["title"] for h in s.recall("food")]


def test_restored_memory_returns_to_semantic_results_immediately():
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    pid = s.remember("user", "Pizza", "I eat pizza at every meal")["id"]
    s.forget(pid)
    s.recall("food")                          # warm the cache WITHOUT pizza
    s.restore(pid)
    assert "Pizza" in [h["title"] for h in s.recall("food")]


def test_backfill_leaves_a_fresh_not_stale_cache():
    """backfill_embeddings rewrites vectors wholesale, then hands the rebuilt
    matrix to backfill_semantic - so afterwards the cache is populated rather
    than empty. What matters is that it reflects the NEW vectors: assert
    freshness, not emptiness."""
    s = _cache_store()
    a = s.remember("user", "Car", "I drive my automobile")["id"]
    s.recall("vehicle")
    stale = s._emb_cache
    assert stale is not None

    b = s.remember("user", "Pizza", "I eat pizza at every meal")["id"]
    s._conn.execute("UPDATE memories SET embedding=NULL")
    s._conn.commit()
    s.backfill_embeddings()

    ids, mat, _types = s._embedding_matrix()
    assert s._emb_cache is not stale, "backfill kept the pre-backfill matrix"
    assert set(ids) == {a, b}, f"cache missed a backfilled row: {ids}"
    assert mat.shape[0] == 2
    assert "Pizza" in [h["title"] for h in s.recall("food")]


def test_cached_and_uncached_recall_agree():
    """Equivalence: the cache is an optimization, so a warm store and a store
    forced to rescan on every call must return byte-identical results. This is
    the test that would catch a subtly-wrong filter in the cached path."""
    s = _cache_store()
    for t, title, body in [
            ("user", "Car", "I drive my automobile to work"),
            ("user", "Pizza", "I eat pizza at every meal"),
            ("project", "Tests", "pytest runs the python tests"),
            ("reference", "Driving", "driving a vehicle safely"),
            ("project", "Cooking", "cooking food is a meal skill")]:
        s.remember(t, title, body)

    for query in ("vehicle", "food", "code", "automobile meal", "nothing here"):
        for type_ in (None, "user", "project", "reference"):
            s._invalidate_embedding_cache()
            cold = s._vector_ids(query, type_)
            warm = s._vector_ids(query, type_)      # same call, now cached
            assert cold == warm, f"cache changed results for {query!r}/{type_!r}"


def test_type_filter_is_honored_through_the_cache():
    """The cache is unfiltered and filters in Python, so the type filter is the
    likeliest place for it to go wrong."""
    s = _cache_store()
    s.remember("user", "Car", "I drive my automobile")
    s.remember("project", "Vehicles", "a project about driving a car")
    ids = s._vector_ids("vehicle", "project")
    types = {s._conn.execute("SELECT type FROM memories WHERE id=?", (i,)).fetchone()[0]
             for i in ids}
    assert types <= {"project"}, f"type filter leaked: {types}"


def test_consolidation_swaps_the_superseded_row_in_the_cache():
    s = _cache_store(consolidate=True, dedup_threshold=0.9)
    old = s.remember("user", "Car", "I drive my automobile")["id"]
    s.recall("vehicle")
    r = s.remember("user", "Auto", "I drive my automobile")   # near-duplicate
    assert r["action"] == "consolidated"
    ids, _mat, _types = s._emb_cache
    assert r["id"] in ids and old not in ids


# --- #81: the cache is maintained incrementally, not dropped per write ------

def _fresh_matrix(s):
    """What a from-scratch scan would produce right now."""
    saved = s._emb_cache, s._emb_buf, s._emb_cache_version
    s._invalidate_embedding_cache()
    fresh = s._embedding_matrix()
    s._emb_cache, s._emb_buf, s._emb_cache_version = saved
    return fresh


def _assert_cache_matches_scan(s, where):
    import numpy as np
    assert s._emb_cache is not None, f"{where}: cache was dropped"
    ids, mat, types = s._emb_cache
    fids, fmat, ftypes = _fresh_matrix(s)
    assert ids == fids, f"{where}: ids {ids} != scan {fids}"
    assert types == ftypes, f"{where}: types differ"
    if fmat is None:
        assert mat is None, where
    else:
        assert np.array_equal(mat, fmat), f"{where}: matrix differs from scan"


def test_incremental_cache_matches_a_full_rebuild_after_every_write():
    """Equivalence, the test that matters: after each kind of row-level write
    the patched cache must be byte-identical to a fresh ORDER BY id scan -
    same ids in the same order, same vectors, same types - because ranking
    ties break on row order and a drifted matrix would corrupt recall."""
    s = _cache_store(consolidate=True, dedup_threshold=0.9)
    car = s.remember("user", "Car", "I drive my automobile")["id"]
    s.recall("vehicle")                                     # warm
    _assert_cache_matches_scan(s, "warm")

    pizza = s.remember("user", "Pizza", "I eat pizza")["id"]
    _assert_cache_matches_scan(s, "insert")

    s.remember("user", "Pizza", "I eat pizza and write python code")   # update
    _assert_cache_matches_scan(s, "update (vector replaced)")

    tests = s.remember("project", "Tests", "pytest runs the tests")["id"]
    s.forget(car)
    _assert_cache_matches_scan(s, "forget")

    s.restore(car)                       # id below the others -> sorted insert
    _assert_cache_matches_scan(s, "restore")

    r = s.remember("user", "Auto", "I drive my automobile")  # supersedes car
    assert r["action"] == "consolidated"
    _assert_cache_matches_scan(s, "consolidate")

    s.purge(pizza)
    _assert_cache_matches_scan(s, "purge")

    s.forget(tests)
    s.forget(r["id"])
    _assert_cache_matches_scan(s, "emptied")
    assert s._emb_cache[1] is None


def test_forgetting_sweep_drops_swept_rows_from_the_cache():
    pytest.importorskip("numpy")
    s = make_forget_store(embedder=FakeEmbedder())
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])
    s.recall("T1")                                          # warm the cache
    assert ids[0] in s._emb_cache[0]
    assert s._run_forgetting_sweep() == 1
    assert ids[0] not in s._emb_cache[0]
    _assert_cache_matches_scan(s, "sweep")


def test_write_from_another_connection_is_noticed(tmp_path):
    """The incremental path only sees writes made through THIS store. A CLI
    purge, a second server, or a replica pull commits through another
    connection - PRAGMA data_version catches that and forces a rebuild, so
    the cache never serves a matrix the file has moved past."""
    pytest.importorskip("numpy")
    path = str(tmp_path / "m.db")
    a = Store(sqlite3.connect(path), "a", lambda *x, **k: None,
              embedder=FakeEmbedder(), sync_read_interval=0)
    a.migrate()
    b = Store(sqlite3.connect(path), "b", lambda *x, **k: None,
              embedder=FakeEmbedder(), sync_read_interval=0)
    b.migrate()
    a.remember("user", "Car", "I drive my automobile")
    assert "Car" in [h["title"] for h in a.recall("vehicle")]    # warm a's cache
    pid = b.remember("user", "Pizza", "I eat pizza at every meal")["id"]
    assert "Pizza" in [h["title"] for h in a.recall("food")], \
        "a served a stale matrix after b's write"
    b.purge(pid)
    assert "Pizza" not in [h["title"] for h in a.recall("food")]


def test_cache_put_and_drop_are_noops_without_a_cache():
    s = _cache_store()
    pid = s.remember("user", "Car", "I drive my automobile")["id"]
    assert s._emb_cache is None                 # nothing has read it yet
    s.forget(pid)
    assert s._emb_cache is None
    s.restore(pid)
    assert s._emb_cache is None
    assert "Car" in [h["title"] for h in s.recall("vehicle")]


# --- #83: parallel tool calls share one Store -------------------------------

def test_parallel_calls_on_one_store_are_atomic(tmp_path):
    """#83: the mcp SDK runs sync tool functions on worker threads, so
    parallel tool calls hit one Store and one sqlite3 connection at once.
    Without the store lock this raised "cannot start a transaction within a
    transaction" / "cannot commit - no transaction is active" and reported
    a fresh create as "updated" (last_insert_rowid() is per-connection, and
    recall inserts session rows) - dozens of times per run of this exact
    workload. With it, every call is atomic: no exceptions, right actions."""
    pytest.importorskip("numpy")
    conn = sqlite3.connect(str(tmp_path / "m.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, sync_read_interval=0)
    s.migrate()
    errors, misreports = [], []

    def worker(t):
        for i in range(40):
            try:
                if i % 5 in (0, 1):
                    r = s.remember("user", f"t{t}-{i}", "I drive my car to eat pizza")
                    if r["action"] != "created":          # every title is unique
                        misreports.append((f"t{t}-{i}", r["action"]))
                elif i % 5 == 4:
                    ids = [row[0] for row in conn.execute(
                        "SELECT id FROM memories WHERE valid_to IS NULL LIMIT 2").fetchall()]
                    if len(ids) == 2:
                        s.link(ids[0], ids[1])
                else:
                    s.recall(["car", "pizza", "food"][i % 3], limit=10)
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == [], errors[:5]
    assert misreports == [], misreports[:5]
    n = conn.execute("SELECT COUNT(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
    assert n == 4 * 16                                     # every remember landed


def test_store_lock_is_reentrant_for_import():
    """import_records calls remember()/link() under the lock it already
    holds - an RLock, so that must not deadlock."""
    s = make_store()
    out = s.import_records([
        {"id": 1, "type": "user", "title": "A", "body": "a", "links": [2]},
        {"id": 2, "type": "user", "title": "B", "body": "b"},
    ])
    assert out["created"] == 2 and out["linked"] == 1


# --- #90: porter stemming in the keyword index -------------------------------

def _fts_sql(conn):
    return conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='memories_fts'").fetchone()[0]


def test_new_store_indexes_with_porter_stemming():
    s = make_store()
    assert "porter" in _fts_sql(s._conn)
    s.remember("project", "Runner", "we decided pytest runs the tests")
    assert [h["title"] for h in s.recall("deciding test")] == ["Runner"]


def test_stemming_can_be_turned_off():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, stemming=False)
    s.migrate()
    assert "porter" not in _fts_sql(conn)
    s.remember("project", "Runner", "pytest runs the tests")
    assert s.recall("test") == []                    # no stem -> no match
    assert [h["title"] for h in s.recall("tests")] == ["Runner"]


def test_migrate_rebuilds_a_pre_stemming_index_with_its_rows(tmp_path):
    """A DB built before #90 has a unicode61 table. Opening it with stemming
    on must recreate the index AND re-index the existing rows, so an old
    memory is findable by a stemmed query immediately after upgrade."""
    path = str(tmp_path / "m.db")
    old = Store(sqlite3.connect(path), "d", lambda *a, **k: None, stemming=False)
    old.migrate()
    old.remember("project", "Legacy", "the tests were written before stemming")
    assert "porter" not in _fts_sql(old._conn)
    old._conn.close()

    new = Store(sqlite3.connect(path), "d", lambda *a, **k: None)   # stemming on
    new.migrate()
    assert "porter" in _fts_sql(new._conn)
    assert [h["title"] for h in new.recall("test writing")] == ["Legacy"]
    # and the triggers still keep the rebuilt index in sync
    new.remember("project", "Fresh", "added after the upgrade")
    assert [h["title"] for h in new.recall("adding")] == ["Fresh"]
    # idempotent: a second migrate on a matching index does not touch it
    new.migrate()
    assert [h["title"] for h in new.recall("test writing")] == ["Legacy"]


def test_migrate_rebuilds_back_when_stemming_is_turned_off(tmp_path):
    path = str(tmp_path / "m.db")
    on = Store(sqlite3.connect(path), "d", lambda *a, **k: None)
    on.migrate()
    on.remember("project", "Runner", "pytest runs the tests")
    on._conn.close()
    off = Store(sqlite3.connect(path), "d", lambda *a, **k: None, stemming=False)
    off.migrate()
    assert "porter" not in _fts_sql(off._conn)
    assert off.recall("test") == []
    assert [h["title"] for h in off.recall("tests")] == ["Runner"]


# --- #92: project awareness --------------------------------------------------

def make_project_store(project="tether", **kw):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, device_id="d", sync_now=lambda *a, **k: None,
              project=project, **kw)
    s.migrate()
    return s


def _tags(s, mid):
    return s._conn.execute("SELECT tags FROM memories WHERE id=?", (mid,)).fetchone()[0]


def test_remember_auto_tags_work_memories_with_the_project():
    s = make_project_store()
    p = s.remember("project", "Test runner", "pytest", tags="ci")["id"]
    f = s.remember("feedback", "Terse", "keep answers short")["id"]
    r = s.remember("reference", "Docs", "see wiki")["id"]
    u = s.remember("user", "Name", "Sid")["id"]
    assert _tags(s, p) == "ci,proj:tether"           # existing tags preserved
    assert _tags(s, f) == "proj:tether"
    assert _tags(s, r) == "proj:tether"
    assert _tags(s, u) == ""                          # user memories stay global


def test_remember_respects_an_explicit_project_tag():
    s = make_project_store()
    mid = s.remember("project", "Elsewhere", "x", tags="proj:other,misc")["id"]
    assert _tags(s, mid) == "proj:other,misc"         # not re-stamped


def test_no_project_means_no_tags_and_no_sections():
    s = make_store()                                  # project=None
    mid = s.remember("project", "A", "a")["id"]
    assert _tags(s, mid) == ""
    assert "# This project" not in s.boot_index()


def test_boot_index_leads_with_this_project():
    s = make_project_store()
    a = s.remember("project", "Ours", "x")["id"]
    b = s.remember("project", "Theirs", "y", tags="proj:other")["id"]
    u = s.remember("user", "Global", "z")["id"]
    idx = s.boot_index()
    lines = idx.splitlines()
    assert lines[0] == "# This project (tether)"
    assert f"#{a} " in lines[1]
    assert "# Everything else" in lines
    rest = idx.split("# Everything else")[1]
    assert f"#{b} " in rest and f"#{u} " in rest


def test_boot_index_project_slice_is_half_the_cap_above_it():
    s = make_project_store(assoc=False, boot_index_cap=4)
    mine = [s.remember("project", f"M{i}", "x")["id"] for i in range(6)]
    others = [s.remember("project", f"O{i}", "y", tags="proj:other")["id"] for i in range(6)]
    idx = s.boot_index()
    head, rest = idx.split("# Everything else")
    def ids_in(text):
        return [int(line.split("#")[1].split()[0]) for line in text.splitlines()]

    head_ids = ids_in("\n".join(head.splitlines()[1:]))
    assert head_ids == list(reversed(mine))[:2]       # newest 2 of ours (cap // 2)
    rest_ids = ids_in(rest.strip())
    assert rest_ids == list(reversed(others))[:2]     # remaining budget, newest first
    assert len(head_ids) + len(rest_ids) == 4          # cap still holds overall


def test_recall_prefers_same_project_hit_on_an_equal_match():
    s = make_project_store()
    ours = s.remember("project", "Ours", "pytest runs the suite")["id"]
    theirs = s.remember("project", "Theirs", "pytest runs the suite", tags="proj:other")["id"]
    # `theirs` is newer, so recency alone would rank it first (#92 must win)
    hits = s.recall("pytest runs the suite", budget=0)
    assert [h["id"] for h in hits][:2] == [ours, theirs]


def test_recall_project_bonus_never_beats_a_clearly_better_match():
    s = make_project_store()
    s.remember("project", "Ours", "unrelated note about lunch")
    theirs = s.remember("project", "Theirs", "pytest runs the whole test suite in CI",
                        tags="proj:other")["id"]
    hits = s.recall("pytest test suite CI", budget=0)
    assert hits[0]["id"] == theirs


# --- #38: concurrent link() must not lose updates ----------------------------

def test_concurrent_links_do_not_clobber_each_other(tmp_path):
    """#38: link() used to SELECT the links list into Python, append, and write
    it back. That SELECT ran outside any transaction (sqlite3 only opens one on
    the first DML), so two concurrent link() calls touching the same memory
    both read the same "before" list and the later UPDATE clobbered the
    earlier one's addition.

    This is the issue's own reproduction: 4 memories, all 6 pairwise links
    fired from genuinely simultaneous threads on separate connections, so every
    node is touched by 3 of the 6 calls - the worst case for the race. Against
    the old read-modify-write every node ended up with 1 of its 3 expected
    links; each must now keep all 3.
    """
    import itertools

    db_path = tmp_path / "memory.db"
    seed = _file_store(db_path, assoc=True)
    ids = [seed.remember("user", f"M{i}", f"body {i}")["id"] for i in range(4)]

    pairs = list(itertools.combinations(ids, 2))
    barrier = threading.Barrier(len(pairs))
    errors = []

    def worker(a, b):
        store = _file_store(db_path, assoc=True)
        try:
            barrier.wait(timeout=10)      # fire all six at the same instant
            store.link(a, b)
        except Exception as e:            # surfaced via `errors`, asserted below
            errors.append(e)
        finally:
            store._conn.close()

    threads = [threading.Thread(target=worker, args=p) for p in pairs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, f"link() raised under concurrency: {errors}"
    check = sqlite3.connect(str(db_path))
    for mid in ids:
        stored = set(_json.loads(check.execute(
            "SELECT links FROM memories WHERE id=?", (mid,)).fetchone()[0]))
        assert stored == set(ids) - {mid}, (
            f"memory {mid} lost links: has {sorted(stored)}, "
            f"expected {sorted(set(ids) - {mid})}")
    # the edges table was always correct (atomic upsert); assert it stays so,
    # since memories.links is the only source of truth if edges were rebuilt
    assert check.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='explicit'").fetchone()[0] == len(pairs)


def test_link_is_idempotent_and_does_not_duplicate():
    s = make_store()
    a = s.remember("user", "A", "body a")["id"]
    b = s.remember("user", "B", "body b")["id"]
    s.link(a, b)
    s.link(a, b)                                    # again
    s.link(b, a)                                    # and reversed
    assert _json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (a,)).fetchone()[0]) == [b]
    assert _json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (b,)).fetchone()[0]) == [a]


def test_link_preserves_links_added_by_remember():
    """The SQL union must merge with what's already there, not replace it."""
    s = make_store()
    a = s.remember("user", "A", "body a")["id"]
    b = s.remember("user", "B", "body b")["id"]
    c = s.remember("user", "C", "body c", links=[a])["id"]
    s.link(c, b)
    stored = set(_json.loads(s._conn.execute(
        "SELECT links FROM memories WHERE id=?", (c,)).fetchone()[0]))
    assert stored == {a, b}, f"link() dropped a pre-existing link: {stored}"


# --- #30: snippet payloads + fetch-on-demand ---------------------------------

_LONG = ("Intro line about nothing much. " + "filler filler filler. " * 200
         + " The HEBBIAN wiring detail lives here in the middle. "
         + "trailing trailing trailing. " * 200)


def test_short_bodies_are_returned_whole_and_unmarked():
    """A memory that fits must be byte-identical to pre-#30 output - no
    ellipsis, no truncated flag, no body_chars."""
    s = make_store()
    s.remember("user", "Short", "a brief fact")
    hit = s.recall("brief")[0]
    assert hit["body"] == "a brief fact"
    assert "truncated" not in hit and "body_chars" not in hit


def test_long_bodies_are_excerpted_and_marked():
    s = make_store()
    s.remember("project", "Long", _LONG)
    hit = s.recall("hebbian")[0]
    assert len(hit["body"]) < len(_LONG) / 10
    assert hit["truncated"] is True
    assert hit["body_chars"] == len(_LONG)


def test_excerpt_is_centered_on_the_query_match():
    """The excerpt must show WHY the memory matched, not just its opening -
    the match is deliberately buried in the middle of _LONG."""
    s = make_store()
    s.remember("project", "Long", _LONG)
    body = s.recall("hebbian")[0]["body"]
    assert "HEBBIAN" in body, f"excerpt missed the match: {body[:120]!r}"
    assert body.startswith("…") and body.endswith("…")


def test_excerpt_falls_back_to_the_head_when_nothing_matches():
    """Tag-only lookups have no query at all; take the opening rather than
    returning nothing useful."""
    s = make_store()
    s.remember("project", "Long", _LONG, tags="proj:x")
    hit = s.recall("", tags="proj:x")[0]
    assert hit["body"].startswith("Intro line")
    assert hit["body"].endswith("…")
    assert hit["truncated"] is True


def test_full_true_returns_whole_bodies():
    s = make_store()
    s.remember("project", "Long", _LONG)
    hit = s.recall("hebbian", full=True)[0]
    assert hit["body"] == _LONG
    assert "truncated" not in hit


def test_excerpting_can_be_disabled_entirely():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, excerpt_chars=0,
              sync_read_interval=0)
    s.migrate()
    s.remember("project", "Long", _LONG)
    assert s.recall("hebbian")[0]["body"] == _LONG


def test_get_returns_one_memory_whole():
    """The fetch half of snippet-plus-fetch."""
    s = make_store()
    mid = s.remember("project", "Long", _LONG)["id"]
    assert s.recall("hebbian")[0]["truncated"] is True     # excerpt first...
    assert s.get(mid)["body"] == _LONG                     # ...then fetch whole


def test_get_refuses_missing_and_forgotten_ids():
    s = make_store()
    mid = s.remember("user", "Gone", "body")["id"]
    assert s.get(9999) is None
    s.forget(mid)
    assert s.get(mid) is None, "get() must not resurrect a forgotten memory"


def test_excerpt_never_exceeds_the_configured_width_by_much():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, excerpt_chars=200,
              sync_read_interval=0)
    s.migrate()
    s.remember("project", "Long", _LONG)
    body = s.recall("hebbian")[0]["body"]
    assert len(body) <= 200 + 2, f"excerpt overran its budget: {len(body)}"


def test_excerpt_shrinks_the_payload_substantially():
    """The point of #30, asserted as a property rather than a vibe."""
    conn = sqlite3.connect(":memory:")
    full_store = Store(conn, "d", lambda *a, **k: None, excerpt_chars=0,
                       sync_read_interval=0)
    full_store.migrate()
    s = make_store()
    for st in (full_store, s):
        for i in range(5):
            st.remember("project", f"Doc {i}", _LONG)
    big = len(_json.dumps(full_store.recall("hebbian", limit=5)))
    small = len(_json.dumps(s.recall("hebbian", limit=5)))
    assert small * 10 < big, f"excerpting saved too little: {big} -> {small}"


def test_action_is_correct_when_the_clock_does_not_tick(monkeypatch):
    """#68 (found by the Windows CI job): `action` used to be inferred from
    `created_at == now`, which assumes the clock advances between two writes.
    datetime.now() has ~15.6ms resolution on Windows before Python 3.13, so two
    remembers inside one tick shared a timestamp and an UPDATE reported itself
    as "created". The row was always correct - but `action` is how a caller
    tells "I made a new memory" from "I refined an existing one".

    Freezing the clock reproduces it on any platform.
    """
    from tether import store as store_module

    monkeypatch.setattr(store_module, "_now",
                        lambda: "2026-01-01T00:00:00.000000+00:00")
    s = make_store()
    first = s.remember("user", "Prefers TDD", "Wants tests first.")
    again = s.remember("user", "  prefers   tdd ", "Wants tests first, plus evidence.")

    assert first["action"] == "created"
    assert again["action"] == "updated", "an upsert misreported itself as a create"
    assert again["id"] == first["id"]
    assert s._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert s._conn.execute(
        "SELECT body FROM memories").fetchone()[0] == "Wants tests first, plus evidence."


def test_action_still_reports_created_for_distinct_titles_on_a_frozen_clock(monkeypatch):
    """The mirror case: a frozen clock must not make genuine creates look like
    updates either."""
    from tether import store as store_module

    monkeypatch.setattr(store_module, "_now",
                        lambda: "2026-01-01T00:00:00.000000+00:00")
    s = make_store()
    a = s.remember("user", "First", "body one")
    b = s.remember("user", "Second", "body two")
    assert a["action"] == "created" and b["action"] == "created"
    assert a["id"] != b["id"]


def test_action_is_created_when_another_table_just_took_the_same_rowid():
    """#85: the #68 fix compared last_insert_rowid() before/after the upsert,
    but that counter is connection-wide across ALL tables. recall() inserts
    session_members/edges/meta rows; whenever one of those took the same
    rowid as the next memory id, a genuine create compared equal and came
    back "updated". Reproduced here without recall: park last_insert_rowid()
    on the id the next memory will get, then remember a brand-new title."""
    s = make_store()
    s._conn.execute("INSERT INTO meta(key, value) VALUES ('probe', '1')")   # meta rowid 1
    assert s._conn.execute("SELECT last_insert_rowid()").fetchone()[0] == 1
    r = s.remember("user", "Brand new", "memory id 1")                     # memories id 1
    assert r["id"] == 1
    assert r["action"] == "created", "a fresh create reported as an update (#85)"
    again = s.remember("user", "Brand new", "refined")
    assert again["action"] == "updated" and again["id"] == 1


def test_action_survives_recall_inserting_rows_between_writes():
    """#85 through the real path: with the graph on, recall() inserts session
    rows on the same connection. A create right after must still say so."""
    s = make_b1_store(assoc=True)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(3)]
    for _ in range(3):
        s.recall("T1")                       # session_members / edges / meta inserts
    created = s.remember("user", "T-new", "b")
    assert created["action"] == "created" and created["id"] not in ids
