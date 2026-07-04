import math
import sqlite3
from bench import metrics
from bench import corpus as corpus_mod
from tether.store import Store


class FakeEmbedder:
    """Deterministic bag-of-words embedder: one dim per known vocab token.
    Matches the store's embedder contract (.embed(text)->unit vector, .name,
    .dims) — mirrors tests/test_store.py's fake, not a batch .encode API."""
    name = "fake-bench"
    _VOCAB = ["login", "500", "load", "dave", "orm", "neovim", "car",
              "office", "auth", "break", "distrusts", "drive"]
    dims = len(_VOCAB)

    def embed(self, text):
        import math
        t = text.lower()
        v = [1.0 if w in t else 0.0 for w in self._VOCAB]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


def _store(assoc=False, embedder=None):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "dev", lambda *a, **k: None, embedder=embedder, assoc=assoc)
    s.migrate()
    return s


def test_recall_at_k_hit_and_miss():
    assert metrics.recall_at_k([9, 3, 7], {7}, k=3) == 1.0
    assert metrics.recall_at_k([9, 3, 7], {7}, k=2) == 0.0
    assert metrics.recall_at_k([], {7}, k=5) == 0.0
    assert metrics.recall_at_k([1, 2], set(), k=5) == 0.0  # empty gold -> 0


def test_mrr_first_gold_rank():
    assert metrics.mrr([5, 8, 2], {2}) == 1.0 / 3
    assert metrics.mrr([2, 8, 5], {2, 8}) == 1.0        # first position
    assert metrics.mrr([1, 2, 3], {9}) == 0.0           # absent


def test_ndcg_at_k_perfect_and_worse():
    # single gold at rank 1 -> 1.0
    assert metrics.ndcg_at_k([7, 1, 2], {7}, k=3) == 1.0
    # single gold at rank 2 -> 1/log2(3) normalized by ideal (1.0)
    got = metrics.ndcg_at_k([1, 7, 2], {7}, k=3)
    assert math.isclose(got, (1 / math.log2(3)), rel_tol=1e-9)
    assert metrics.ndcg_at_k([1, 2, 3], {9}, k=3) == 0.0
    assert metrics.ndcg_at_k([1, 2], set(), k=2) == 0.0


def test_mini_corpus_shape():
    c = corpus_mod.MINI
    assert len(c.memories) >= 4
    keys = {m.key for m in c.memories}
    # every query's target and golds reference real memory keys
    for q in c.queries:
        assert q.target_key in keys
        assert set(q.gold_keys) <= keys
        assert q.kind in ("graph_only", "control")
    # exactly one of each kind in the mini fixture
    assert len(c.by_kind("graph_only")) == 1
    assert len(c.by_kind("control")) == 1
    # control gold is the target itself (pin: v0.2-findable target, assert-no-demote)
    ctrl = c.by_kind("control")[0]
    assert ctrl.gold_keys == [ctrl.target_key]


def test_loader_maps_keys_to_ids_and_links():
    import pytest
    pytest.importorskip("numpy")
    from bench import loader, corpus as corpus_mod
    s = _store(assoc=True, embedder=FakeEmbedder())
    id_of = loader.load(corpus_mod.MINI, s)
    assert set(id_of) == {"bug", "pref", "editor", "car"}
    assert all(isinstance(v, int) for v in id_of.values())
    # a recall for the target returns it (sanity that memories landed)
    hits = s.recall("login returns 500 under load", limit=5)
    assert id_of["bug"] in [h["id"] for h in hits]


def test_assert_golds_far_passes_and_flags():
    import pytest
    pytest.importorskip("numpy")
    from bench import selfcheck, corpus as corpus_mod
    e = FakeEmbedder()
    # MINI graph_only gold ("dave distrusts the ORM layer") shares no vocab
    # token with query ("why did login break") -> cosine 0 -> passes.
    selfcheck.assert_golds_far(corpus_mod.MINI, e, threshold=0.35)
    # A rigged corpus: gold body shares the query's tokens -> must raise.
    rigged = corpus_mod.Corpus(
        name="rigged",
        memories=[corpus_mod.Memory("a", "note", "t", "login 500 load"),
                  corpus_mod.Memory("b", "note", "t", "login 500 load again")],
        tasks=[corpus_mod.Task("x", ["a", "b"])],
        queries=[corpus_mod.Query("login 500 load", "a", ["b"], "graph_only")],
    )
    with pytest.raises(AssertionError):
        selfcheck.assert_golds_far(rigged, e, threshold=0.35)


def test_assert_targets_found_passes_and_flags():
    import pytest
    pytest.importorskip("numpy")
    from bench import selfcheck, loader, corpus as corpus_mod
    s = _store(assoc=False, embedder=FakeEmbedder())
    id_of = loader.load(corpus_mod.MINI, s)
    selfcheck.assert_targets_found(corpus_mod.MINI, s, id_of, k=10)
    # A query whose target is unfindable by v0.2 must raise.
    bad = corpus_mod.Corpus(
        name="bad", memories=corpus_mod.MINI.memories, tasks=[],
        queries=[corpus_mod.Query("xyzzy nonexistent terms", "car",
                                  ["car"], "control")])
    with pytest.raises(AssertionError):
        selfcheck.assert_targets_found(bad, s, id_of, k=10)


def test_warm_creates_hebbian_edges():
    import pytest
    pytest.importorskip("numpy")
    from bench import warmup, loader, corpus as corpus_mod
    s = _store(assoc=True, embedder=FakeEmbedder())
    id_of = loader.load(corpus_mod.MINI, s)
    # before warm-up: no hebbian edges
    before = s._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert before == 0
    warmup.warm(corpus_mod.MINI, s, id_of, repeats=3)
    after = s._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert after >= 1  # bug <-> pref co-recalled in the "auth" task


def test_build_conditions_edge_states():
    import pytest
    pytest.importorskip("numpy")
    from bench import conditions, corpus as corpus_mod

    def heb(store):
        return store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]

    e = FakeEmbedder()
    s_v2, _ = conditions.build(corpus_mod.MINI, "v2", e)
    assert s_v2._graph.enabled is False

    s_cold, _ = conditions.build(corpus_mod.MINI, "cold", e)
    assert s_cold._graph.enabled is True
    assert heb(s_cold) == 0                      # no usage yet

    s_warm, _ = conditions.build(corpus_mod.MINI, "warmed", e)
    assert heb(s_warm) >= 1                       # warm-up wired co-recalls

    s_or, id_of = conditions.build(corpus_mod.MINI, "oracle", e)
    # full clique of the 2-member "auth" task at max weight
    row = s_or._conn.execute(
        "SELECT weight FROM edges WHERE kind='hebbian' "
        "AND src=? AND dst=?",
        tuple(sorted((id_of["bug"], id_of["pref"])))).fetchone()
    assert row is not None and row[0] == 5.0


def test_run_smoke_all_conditions_both_classes():
    import pytest
    pytest.importorskip("numpy")
    from bench import run, corpus as corpus_mod
    rep = run.run(corpus_mod.MINI, FakeEmbedder(), k=5)
    # all four conditions present, both classes measured
    for cond in ("v2", "cold", "warmed", "oracle"):
        assert cond in rep["conditions"]
        assert "graph_only" in rep["conditions"][cond]
        assert "control" in rep["conditions"][cond]
    # derived numbers exist and are numbers
    assert isinstance(rep["learning_delta_ndcg"], float)
    assert isinstance(rep["headroom_ndcg"], float)
    assert "no_regression" in rep and "passed" in rep["no_regression"]
    # rec 2: held-out (frozen, contamination-free) learning delta present
    ho = rep["held_out"]
    assert isinstance(ho["learning_delta_ndcg"], float)
    assert set(ho["distribution"]) == {"improved", "unchanged", "regressed"}


def test_evaluate_freeze_leaves_graph_unchanged():
    # A frozen eval pass must not persist any eval-time learning: every query
    # sees the identical post-warmup graph. This is what makes the held-out
    # learning delta contamination-free.
    import pytest
    pytest.importorskip("numpy")
    from bench import run, conditions, corpus as corpus_mod
    s, id_of = conditions.build(corpus_mod.MINI, "warmed", FakeEmbedder())
    q = "SELECT src,dst,kind,weight FROM edges ORDER BY 1,2,3,4"
    before = s._conn.execute(q).fetchall()
    run.evaluate(s, id_of, corpus_mod.MINI, "graph_only", k=5, freeze=True)
    assert s._conn.execute(q).fetchall() == before        # rolled back


def test_evaluate_without_freeze_mutates_graph():
    # The contamination the freeze exists to remove: an ordinary eval pass calls
    # touch_session on each recall, so the graph changes as it is measured.
    import pytest
    pytest.importorskip("numpy")
    from bench import run, conditions, corpus as corpus_mod
    s, id_of = conditions.build(corpus_mod.MINI, "warmed", FakeEmbedder())
    q = "SELECT src,dst,kind,weight FROM edges ORDER BY 1,2,3,4"
    before = s._conn.execute(q).fetchall()
    run.evaluate(s, id_of, corpus_mod.MINI, "graph_only", k=5, freeze=False)
    assert s._conn.execute(q).fetchall() != before        # learned during eval


def test_assert_warmup_disjoint_passes_and_flags():
    import pytest
    from bench import selfcheck, corpus as corpus_mod
    # MINI: member titles ("auth 500s", ...) are disjoint from the eval queries.
    selfcheck.assert_warmup_disjoint(corpus_mod.MINI)
    # rigged: an eval query IS a member title -> warmup would train on the exact
    # eval query -> not held out -> must raise.
    rigged = corpus_mod.Corpus(
        name="rig",
        memories=[corpus_mod.Memory("a", "project",
                                    "login returns 500 under load", "x"),
                  corpus_mod.Memory("b", "user", "pref", "y")],
        tasks=[corpus_mod.Task("t", ["a", "b"])],
        queries=[corpus_mod.Query("login returns 500 under load", "a",
                                  ["b"], "graph_only")])
    with pytest.raises(AssertionError):
        selfcheck.assert_warmup_disjoint(rigged)


def test_crystallized_condition_wires_hub_edges():
    import pytest
    pytest.importorskip("numpy")
    from bench import conditions, corpus as corpus_mod
    s, id_of = conditions.build(corpus_mod.MINI, "crystallized", FakeEmbedder())
    # MINI's 'auth' task has 2 members -> a principle crystallized over both
    # writes 2 directional principle->source edges.
    n = s._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='crystallized'").fetchone()[0]
    assert n == 2


def test_crystallized_detected_condition_builds():
    # MINI has no explicit links, and hebbian peaks are W_b=0, so the real
    # detector finds no clusters -> zero principles, condition == warmed. The
    # point is that build() runs the detector path without error.
    import pytest
    pytest.importorskip("numpy")
    from bench import conditions, corpus as corpus_mod
    s, id_of = conditions.build(
        corpus_mod.MINI, "crystallized_detected", FakeEmbedder())
    n = s._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='crystallized'").fetchone()[0]
    assert n == 0


def test_assert_principles_far_passes_and_flags():
    import pytest
    pytest.importorskip("numpy")
    from bench import selfcheck, conditions, corpus as corpus_mod
    e = FakeEmbedder()
    # neutral principle bodies share no vocab with MINI queries/golds -> pass.
    selfcheck.assert_principles_far(corpus_mod.MINI, e, threshold=0.35)
    # monkeypatch a riggable (topical) principle body -> must raise.
    orig = conditions.principle_body
    conditions.principle_body = lambda i: "login returns 500 under load"
    try:
        with pytest.raises(AssertionError):
            selfcheck.assert_principles_far(corpus_mod.MINI, e, threshold=0.35)
    finally:
        conditions.principle_body = orig


def test_run_reports_crystallization_block():
    import pytest
    pytest.importorskip("numpy")
    from bench import run, corpus as corpus_mod
    rep = run.run(corpus_mod.MINI, FakeEmbedder(), k=5)
    cz = rep["crystallization"]
    for key in ("cold_frozen_ndcg", "warmed_frozen_ndcg", "oracle_frozen_ndcg",
                "crystallized_frozen_ndcg", "crystallized_detected_frozen_ndcg",
                "delta_vs_cold", "headroom_recovered", "detected_delta_vs_warmed"):
        assert key in cz and isinstance(cz[key], float)


def test_distribution_counts():
    from bench import run
    base = [{"ndcg": 0.5}, {"ndcg": 0.5}, {"ndcg": 0.5}]
    cond = [{"ndcg": 0.9}, {"ndcg": 0.5}, {"ndcg": 0.1}]  # up / flat / down
    d = run.distribution(base, cond, eps=0.02)
    assert d == {"improved": 1, "unchanged": 1, "regressed": 1}
