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
