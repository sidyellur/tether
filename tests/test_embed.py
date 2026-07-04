import math
import sys

import pytest


def test_get_embedder_returns_none_when_model2vec_missing(monkeypatch):
    # Poison the import: `from model2vec import StaticModel` will raise.
    monkeypatch.setitem(sys.modules, "model2vec", None)
    from tether import embed
    assert embed.get_embedder("anything") is None


def test_real_embedder_produces_unit_normalized_vectors():
    pytest.importorskip("model2vec")
    from tether import embed
    e = embed.get_embedder()
    if e is None:
        pytest.skip("model could not be loaded (likely offline)")
    v = e.embed("hello world")
    assert e.dims > 0 and len(v) == e.dims
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-4
    assert isinstance(e.name, str) and e.name
