import math
import sqlite3
from bench import metrics
from bench import corpus as corpus_mod
from tether.store import Store


class FakeEmbedder:
    """Deterministic bag-of-words-ish vector: one dim per known token."""
    _VOCAB = ["login", "500", "load", "dave", "orm", "neovim", "car",
              "office", "auth", "break", "distrusts", "drive"]

    def encode(self, texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.array([1.0 if w in t.lower() else 0.0 for w in self._VOCAB])
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


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
