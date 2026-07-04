import sqlite3

import pytest

from tether.graph import Graph


def make_graph(enabled=True):
    conn = sqlite3.connect(":memory:")
    g = Graph(conn, enabled=enabled)
    g.migrate()
    return g


def test_graph_migrate_creates_tables():
    g = make_graph()
    names = {r[0] for r in g._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"edges", "session_members"} <= names


def test_upsert_edge_canonical_and_modes():
    g = make_graph()
    g._upsert_edge(5, 2, "semantic", 0.3, "t", mode="max")
    g._upsert_edge(2, 5, "semantic", 0.7, "t", mode="max")   # same pair, higher
    row = g._conn.execute(
        "SELECT src, dst, weight FROM edges WHERE kind='semantic'").fetchone()
    assert row == (2, 5, 0.7)                                 # canonical src<dst, max kept
    g._upsert_edge(2, 5, "hebbian", 0.5, "t", mode="add")
    g._upsert_edge(2, 5, "hebbian", 0.5, "t", mode="add")
    w = g._conn.execute(
        "SELECT weight FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert w == 1.0                                           # accumulated


def test_on_forget_deletes_edges_and_session_rows():
    g = make_graph()
    g._upsert_edge(1, 2, "semantic", 0.5, "t", mode="max")
    g._upsert_edge(2, 3, "semantic", 0.5, "t", mode="max")
    g._conn.execute("INSERT INTO session_members VALUES('s', 2, 1.0, 't')")
    g.on_forget(2)
    assert g._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert g._conn.execute("SELECT COUNT(*) FROM session_members").fetchone()[0] == 0


class FakeEmbedder:
    name = "fake-3d"
    dims = 3
    _AXES = [("car", "automobile", "drive"), ("pizza", "food"), ("python", "test")]

    def embed(self, text):
        import math
        t = text.lower()
        v = [float(sum(w in t for w in ax)) for ax in self._AXES]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


def _pack(vec):
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


def _seed_memory(conn, mid, type, title, body, emb):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, type TEXT, "
        "title TEXT, body TEXT, embedding BLOB, valid_to TEXT)")
    conn.execute("INSERT INTO memories(id,type,title,body,embedding,valid_to) "
                 "VALUES(?,?,?,?,?,NULL)", (mid, type, title, body, emb))


def test_on_remember_writes_semantic_edges():
    pytest.importorskip("numpy")
    g = make_graph()
    e = FakeEmbedder()
    a = _pack(e.embed("I drive my car"))          # vehicle axis
    b = _pack(e.embed("driving the automobile"))  # vehicle axis (near a)
    c = _pack(e.embed("pizza and food"))          # food axis (far)
    _seed_memory(g._conn, 1, "user", "A", "car", a)
    _seed_memory(g._conn, 2, "user", "B", "auto", b)
    _seed_memory(g._conn, 3, "user", "C", "lunch", c)
    g.on_remember(2, b)
    # 2 links to 1 (near) with higher weight than to 3 (far)
    w12 = g._conn.execute("SELECT weight FROM edges WHERE src=1 AND dst=2 AND kind='semantic'").fetchone()
    w23 = g._conn.execute("SELECT weight FROM edges WHERE src=2 AND dst=3 AND kind='semantic'").fetchone()
    assert w12 is not None and w12[0] > (w23[0] if w23 else 0.0)


def test_on_link_writes_explicit_edge():
    g = make_graph()
    g.on_link(7, 3)
    row = g._conn.execute("SELECT src, dst, kind, weight FROM edges").fetchone()
    assert row == (3, 7, "explicit", 1.0)
