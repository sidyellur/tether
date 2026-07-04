"""Build a fresh Store for each measured condition from the same corpus."""
import sqlite3
from itertools import combinations

from tether.store import Store
from tether.graph import HEBBIAN_CAP
from bench import loader, warmup


def _fresh(embedder, assoc):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "bench", lambda *a, **k: None, embedder=embedder, assoc=assoc)
    s.migrate()
    return s


def build(corpus, condition, embedder):
    if condition == "v2":
        s = _fresh(embedder, assoc=False)
        return s, loader.load(corpus, s)

    s = _fresh(embedder, assoc=True)
    id_of = loader.load(corpus, s)   # semantic + explicit edges built on load

    if condition == "cold":
        return s, id_of
    if condition == "warmed":
        warmup.warm(corpus, s, id_of, repeats=3)
        return s, id_of
    if condition == "oracle":
        _wire_oracle(s, corpus, id_of)
        return s, id_of
    raise ValueError(f"unknown condition {condition!r}")


def _wire_oracle(store, corpus, id_of):
    """Ideal graph: every within-task pair wired hebbian at max weight."""
    now = store._conn.execute("SELECT datetime('now')").fetchone()[0]
    for task in corpus.tasks:
        ids = [id_of[k] for k in task.member_keys]
        for a, b in combinations(ids, 2):
            store._graph._upsert_edge(a, b, "hebbian", HEBBIAN_CAP, now, mode="max")
    store._conn.commit()
