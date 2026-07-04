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
