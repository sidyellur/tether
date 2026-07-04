"""Pre-flight assertions that make the benchmark honest:
  - assert_golds_far: graph_only golds must be semantically FAR from their
    query, so any lift is the graph's, not smuggled semantic overlap.
  - assert_targets_found: every query's target must be retrievable by v0.2,
    so a condition never underperforms merely because the SEED was missed.
  - assert_warmup_disjoint: the queries that WARM the graph must not overlap
    the queries we EVALUATE on, so the learning delta measures transfer to
    held-out queries, not memorization of the eval set."""


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


def assert_principles_far(corpus, embedder, threshold=0.35):
    """The synthesized crystallization principles must be semantically FAR from
    every query AND every graph_only gold. This rules out the two ways a
    principle could smuggle a non-structural lift: (a) matching the query so it
    is seeded directly, or (b) matching a gold so its semantic kNN edge — not its
    crystallized edge — carries the activation. If the neutral principle text is
    ever edited to include topical vocabulary, this fails loudly."""
    from bench import conditions
    bodies = [conditions.principle_body(i) for i in range(len(corpus.tasks))]
    targets = [q.query for q in corpus.queries]
    gold_keys = {g for q in corpus.by_kind("graph_only") for g in q.gold_keys}
    body_of = {m.key: m.body for m in corpus.memories}
    targets += [body_of[g] for g in gold_keys]
    for body in bodies:
        for text in targets:
            sim = _cos(embedder, body, text)
            assert sim < threshold, (
                f"principle text riggable: {body!r} is cos={sim:.3f} >= "
                f"{threshold} to {text!r} — a crystallization lift could then be "
                f"semantic, not structural. Keep principle bodies topic-free.")


def assert_warmup_disjoint(corpus):
    """warmup.warm recalls each task member by its TITLE; eval recalls by
    q.query. If any eval query equals a warmup query, that query was trained on
    directly and the learning delta overstates transfer. Assert no overlap."""
    member_keys = {k for t in corpus.tasks for k in t.member_keys}
    title_of = {m.key: m.title for m in corpus.memories}
    warm_qs = {title_of[k].strip().lower() for k in member_keys}
    for q in corpus.queries:
        assert q.query.strip().lower() not in warm_qs, (
            f"not held out: eval query {q.query!r} is also a warmup query "
            f"(a task member's title) — the graph was trained on this exact "
            f"query, so its learning delta is memorization, not transfer.")
