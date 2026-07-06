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


def test_unprime_removes_session_rows_but_keeps_edges():
    # #42: a soft-archive (supersede/forget-sweep) must scrub session_members
    # so the memory can't be primed back into recall, but it's a soft delete -
    # unlike on_forget, edges stay (valid_to filtering already hides the node).
    g = make_graph()
    g._upsert_edge(1, 2, "semantic", 0.5, "t", mode="max")
    g._conn.execute("INSERT INTO session_members VALUES('s', 1, 1.0, 't')")
    g._conn.execute("INSERT INTO session_members VALUES('s', 2, 1.0, 't')")
    g.unprime(1)
    rows = {r[0] for r in g._conn.execute(
        "SELECT memory_id FROM session_members").fetchall()}
    assert rows == {2}
    assert g._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


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


def _mem(conn, mid, type="user", valid=True):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, type TEXT, "
        "title TEXT, body TEXT, embedding BLOB, valid_to TEXT)")
    conn.execute("INSERT INTO memories(id,type,valid_to) VALUES(?,?,?)",
                 (mid, type, None if valid else "t"))


def test_spread_reaches_two_hops_within_budget():
    g = make_graph()
    for i in (1, 2, 3):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    g._upsert_edge(2, 3, "hebbian", 1.0, "t", mode="max")
    act0, _ = g.spread({1: 1.0}, budget=0)          # spreading off
    assert set(act0) == {1}
    act2, receipts = g.spread({1: 1.0}, budget=8)   # 1 -> 2 -> 3
    assert 2 in act2 and 3 in act2
    assert act2[1] > act2[2] > act2[3]              # decays with distance
    assert receipts[2]["from"] == 1 and receipts[2]["kind"] == "hebbian"
    assert receipts[3]["hops"] == 2


def test_spread_keeps_seed_above_weak_associate():
    g = make_graph()
    for i in (1, 2): _mem(g._conn, i)
    g._upsert_edge(1, 2, "hebbian", 0.3, "t", mode="max")
    act, _ = g.spread({1: 0.5, 2: 0.5}, budget=8)   # both seeded equally
    assert act[1] > act[2] or act[2] > act[1]        # deterministic, no crash
    # a pure associate never outranks its strong seed source
    act2, _ = g.spread({1: 1.0}, budget=8)
    assert act2[1] > act2[2]


def test_spread_respects_type_and_validity():
    g = make_graph()
    _mem(g._conn, 1, "user"); _mem(g._conn, 2, "project"); _mem(g._conn, 3, "user", valid=False)
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    g._upsert_edge(1, 3, "hebbian", 1.0, "t", mode="max")
    act, _ = g.spread({1: 1.0}, budget=8, type="user")
    assert 2 not in act        # wrong type filtered
    assert 3 not in act        # superseded filtered


def _meta_pair():
    store = {}
    return (lambda k: store.get(k)), (lambda k, v: store.__setitem__(k, str(v)))


def test_resolve_session_uses_param_then_time_buckets():
    g = make_graph()
    get, set_ = _meta_pair()
    assert g.resolve_session("explicit-1", get, set_) == "explicit-1"
    sid1 = g.resolve_session(None, get, set_)
    sid2 = g.resolve_session(None, get, set_)      # immediately after -> same bucket
    assert sid1 == sid2
    # simulate a long gap by rewinding last-activity far into the past
    set_("assoc_last_activity", "2000-01-01T00:00:00+00:00")
    sid3 = g.resolve_session(None, get, set_)
    assert sid3 != sid1


def test_touch_session_primes_and_decays():
    g = make_graph()
    g.touch_session("s", [1, 2])
    a1 = g.session_activation("s")
    # rank-weighted bump: rank0 -> 1.0, rank1 -> 0.5 (B1)
    assert a1[1] == 1.0 and a1[2] == 0.5
    g.touch_session("s", [2])                       # decays 1&2, bumps 2 (rank0)
    a2 = g.session_activation("s")
    assert a2[1] == 0.5 and a2[2] == 1.25


def test_touch_session_lays_hebbian_edges():
    g = make_graph()
    g.touch_session("s", [1, 2, 3])                 # ranks 0,1,2 -> 1.0/0.5/0.25
    w = g._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert w == 3                                   # pairs (1,2),(1,3),(2,3)


def test_touch_session_bump_is_rank_weighted_not_uniform():
    # B1 root cause: a uniform bump over the whole returned list makes
    # activation carry no information (mass ties -> memory_id tie-break decides
    # the Hebbian top-M). The bump must decay geometrically with result rank so
    # the recall's SUBJECT dominates its working-set contribution.
    g = make_graph()
    g.touch_session("s", list(range(1, 21)))        # a full returned list
    a = g.session_activation("s")
    assert a[1] == 1.0 and a[2] == 0.5 and a[3] == 0.25
    # deep-tail padding never enters the working set at all
    assert 10 not in a and 20 not in a
    assert len(set(a.values())) == len(a)           # no ties -> no id squatting


def test_touch_session_wire_floor_excludes_weak_members():
    # members below HEBBIAN_WIRE_FLOOR stay in the session (priming) but are
    # not Hebbian-wired: co-return is not co-use.
    g = make_graph()
    g.touch_session("s", [1, 2, 3, 4])              # rank3 bump 0.125 < floor
    pairs = {(r[0], r[1]) for r in g._conn.execute(
        "SELECT src, dst FROM edges WHERE kind='hebbian'").fetchall()}
    assert pairs == {(1, 2), (1, 3), (2, 3)}        # nothing wired to 4
    assert 4 in g.session_activation("s")           # but 4 still primes


def test_touch_session_old_uniform_rule_is_a_knob_setting(monkeypatch):
    # degrade path: BUMP_DECAY=1.0 + WIRE_FLOOR=0.0 IS the pre-B1 rule.
    import tether.graph as graph_mod
    monkeypatch.setattr(graph_mod, "HEBBIAN_BUMP_DECAY", 1.0)
    monkeypatch.setattr(graph_mod, "HEBBIAN_WIRE_FLOOR", 0.0)
    g = make_graph()
    g.touch_session("s", [1, 2])
    a1 = g.session_activation("s")
    assert a1[1] == 1.0 and a1[2] == 1.0
    g.touch_session("s", [2])
    a2 = g.session_activation("s")
    assert a2[1] == 0.5 and a2[2] == 1.5


def test_degree_map_behavioral_only_excludes_semantic():
    g = make_graph()
    for i in (1, 2, 3):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "semantic", 0.9, "t", mode="max")
    g._upsert_edge(1, 3, "hebbian", 0.5, "t", mode="max")
    g._upsert_edge(2, 3, "explicit", 1.0, "t", mode="max")
    deg = g.degree_map()
    assert deg[1] == 0.5          # only the hebbian edge (semantic ignored)
    assert deg[2] == 1.0          # only the explicit edge
    assert deg[3] == 1.5          # hebbian 0.5 + explicit 1.0


def test_degree_map_emits_zero_for_semantic_only_node():
    g = make_graph()
    for i in (1, 2):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "semantic", 0.9, "t", mode="max")
    assert g.degree_map() == {1: 0.0, 2: 0.0}    # present, but zero


def test_degree_map_ignores_edges_to_noncurrent():
    g = make_graph()
    _mem(g._conn, 1)
    _mem(g._conn, 2, valid=False)               # archived
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    assert g.degree_map() == {1: 0.0}           # node 2 not current; its edge doesn't count


# Tests for crystallized edge kind and config (Task 1)
def _graph():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories(id INTEGER PRIMARY KEY, valid_to TEXT, type TEXT);")
    g = Graph(conn, enabled=True)
    g.migrate()
    return conn, g


def test_crystallized_kind_registered_for_spreading():
    from tether.graph import KIND_W
    assert "crystallized" in KIND_W and KIND_W["crystallized"] > 0


def test_on_crystallize_writes_directional_edges():
    conn, g = _graph()
    for i in (1, 2, 3):
        conn.execute("INSERT INTO memories(id, type) VALUES(?, 'project')", (i,))
    g.on_crystallize(1, [2, 3])                     # principle=1, sources 2,3
    rows = conn.execute(
        "SELECT src, dst, kind FROM edges WHERE kind='crystallized' "
        "ORDER BY dst").fetchall()
    assert rows == [(1, 2, "crystallized"), (1, 3, "crystallized")]  # NOT canonicalized


def test_crystallized_counts_toward_hub_degree():
    conn, g = _graph()
    for i in (1, 2, 3):
        conn.execute("INSERT INTO memories(id, type) VALUES(?, 'project')", (i,))
    g.on_crystallize(1, [2, 3])
    deg = g.degree_map()                            # default kinds
    assert deg[1] > 0 and deg[2] > 0                # principle and sources are hubs


def test_on_crystallize_disabled_is_noop():
    conn, g = _graph()
    g.enabled = False
    conn.execute("INSERT INTO memories(id, type) VALUES(1, 'project')")
    g.on_crystallize(1, [2])
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_dismiss_peak_is_canonical_and_readable():
    conn, g = _graph()
    g.migrate()
    g.dismiss_peak(5, 2)                             # order-independent
    assert g.dismissed_peaks() == {(2, 5)}
