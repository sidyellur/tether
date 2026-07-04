# tests/test_crystallize.py — new file
import sqlite3
from tether import crystallize


def _store_with_edges(memories, edges):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories(id INTEGER PRIMARY KEY, valid_to TEXT, "
        "title TEXT, tags TEXT DEFAULT '');"
        "CREATE TABLE edges(src INTEGER, dst INTEGER, kind TEXT, weight REAL, "
        "updated_at TEXT, PRIMARY KEY(src,dst,kind));")
    for mid, title in memories:
        conn.execute("INSERT INTO memories(id, title) VALUES(?,?)", (mid, title))
    for src, dst, kind, w in edges:
        conn.execute("INSERT INTO edges VALUES(?,?,?,?, 't')", (src, dst, kind, w))
    return conn


def test_explicit_peak_seeds_and_semantic_expands():
    # explicit peak (1,2); semantic pulls 3 into the cluster; 4 is far (cos 0.1).
    conn = _store_with_edges(
        [(1, "auth bug"), (2, "pool fix"), (3, "rollback rule"), (4, "unrelated")],
        [(1, 2, "explicit", 1.0),
         (2, 3, "semantic", 0.8),      # >= EXPAND_COS -> member
         (3, 4, "semantic", 0.1)])     # < EXPAND_COS -> excluded
    cands = crystallize.candidates(conn)
    assert len(cands) == 1
    assert cands[0]["member_ids"] == [1, 2, 3]
    assert cands[0]["peak_key"] == (1, 2)


def test_crystallized_edges_do_not_seed_peaks():
    # a crystallized hub must NOT re-seed the cluster it named (no recursion).
    conn = _store_with_edges(
        [(1, "principle"), (2, "src a"), (3, "src b")],
        [(1, 2, "crystallized", 1.0), (1, 3, "crystallized", 1.0)])
    assert crystallize.candidates(conn) == []


def test_semantic_only_does_not_seed():
    # no explicit/hebbian peak -> no candidate (uniform semantic floor).
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c")],
        [(1, 2, "semantic", 0.9), (2, 3, "semantic", 0.9)])
    assert crystallize.candidates(conn) == []


def test_min_cluster_filters_small():
    # explicit peak of 2 with no semantic expansion -> below MIN_CLUSTER (3).
    conn = _store_with_edges(
        [(1, "a"), (2, "b")], [(1, 2, "explicit", 1.0)])
    assert crystallize.candidates(conn) == []


def test_dedup_suppresses_recovered_principle_basis():
    # principle 10 crystallizes sources {1,2,3}; a candidate re-covering >=60%
    # of that basis is suppressed.
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c"), (10, "principle")],
        [(1, 2, "explicit", 1.0), (2, 3, "semantic", 0.9),
         (10, 1, "crystallized", 1.0), (10, 2, "crystallized", 1.0),
         (10, 3, "crystallized", 1.0)])
    assert crystallize.candidates(conn) == []       # basis {1,2,3} fully re-covered


def test_dismissed_peak_suppresses_candidate():
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c")],
        [(1, 2, "explicit", 1.0), (2, 3, "semantic", 0.9)])
    assert len(crystallize.candidates(conn)) == 1   # visible first
    conn.execute("CREATE TABLE crystallize_dismissed(src INTEGER, dst INTEGER, "
                 "PRIMARY KEY(src,dst));")
    conn.execute("INSERT INTO crystallize_dismissed VALUES(1,2)")  # dismiss the peak
    assert crystallize.candidates(conn) == []
