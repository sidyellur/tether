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
import os

_DEFAULT_MODEL = "minishlab/potion-base-8M"


def _is_cached(model_name: str) -> bool:
    """Best-effort check for the HF hub's on-disk cache layout. Used only to
    decide whether skipping the hub's connectivity check is safe (#28) - a
    miss here just means we fall through to the normal online load."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    repo_dir = "models--" + model_name.replace("/", "--")
    return os.path.isdir(os.path.join(cache_dir, repo_dir))


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
        if _is_cached(model_name):
            # Already have it locally - skip the hub's connectivity/metadata
            # round-trip on every process start (#28: it can stall for
            # seconds on a slow/unreachable network and buys nothing once
            # the model is cached). setdefault so an explicit user setting
            # always wins.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from model2vec import StaticModel

        model = StaticModel.from_pretrained(model_name)
        return Embedder(model, model_name)
    except Exception:
        return None
