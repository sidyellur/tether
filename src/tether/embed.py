"""embed.py - optional local semantic embeddings.

Turns memory text into a small dense vector with a STATIC embedding model
(Model2Vec): a tokenize-lookup-pool table, no neural forward pass, CPU-only,
no network at inference. This honors tether's contract - embedding can never
hang or reach the network on the hot path.

Everything here degrades to "disabled": if the optional deps are absent or the
model fails to load, get_embedder() returns None and tether runs pure-FTS5.
Semantic recall is an ADDITIVE boost, never a hard dependency.
"""

import math

_DEFAULT_MODEL = "minishlab/potion-base-8M"


class Embedder:
    """Wraps a Model2Vec StaticModel and returns unit-normalized vectors."""

    def __init__(self, model, name: str):
        self._model = model
        self.name = name
        self.dims = len(self._encode_raw("dimension probe"))

    def _encode_raw(self, text: str) -> list:
        # StaticModel.encode takes a list of strings, returns a 2-D array-like.
        row = self._model.encode([text])[0]
        return [float(x) for x in row]

    def embed(self, text: str) -> list:
        v = self._encode_raw(text or "")
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            return v
        return [x / norm for x in v]


def get_embedder(model_name: str = _DEFAULT_MODEL):
    """Return an Embedder, or None if semantic support is unavailable.

    Never raises: a missing model2vec install, a failed download, or any load
    error degrades to None (the caller falls back to keyword-only recall).
    """
    try:
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(model_name)
        return Embedder(model, model_name)
    except Exception:
        return None
