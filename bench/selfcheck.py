"""Pre-flight assertions that make the benchmark honest:
  - assert_golds_far: graph_only golds must be semantically FAR from their
    query, so any lift is the graph's, not smuggled semantic overlap.
  - assert_targets_found: every query's target must be retrievable by v0.2,
    so a condition never underperforms merely because the SEED was missed."""


def _cos(embedder, a, b):
    # embedder.embed returns unit-normalized vectors, so dot == cosine.
    va, vb = embedder.embed(a), embedder.embed(b)
    return float(sum(x * y for x, y in zip(va, vb)))


def assert_golds_far(corpus, embedder, threshold=0.35):
    body_of = {m.key: m.body for m in corpus.memories}
    for q in corpus.by_kind("graph_only"):
        for g in q.gold_keys:
            sim = _cos(embedder, q.query, body_of[g])
            assert sim < threshold, (
                f"corpus riggable: graph_only gold {g!r} is cos={sim:.3f} "
                f">= {threshold} to query {q.query!r} — semantic search could "
                f"surface it, so lift would not be attributable to the graph.")


def assert_targets_found(corpus, store_v2, id_of, k=10):
    for q in corpus.queries:
        got = [h["id"] for h in store_v2.recall(q.query, limit=k)]
        assert id_of[q.target_key] in got, (
            f"seed unfindable: target {q.target_key!r} not in v0.2 top-{k} "
            f"for query {q.query!r} — the graph has no valid seed to spread "
            f"from, so this query would measure a missed seed, not the graph.")
