"""Run the four conditions over both query classes and print the report.

Headline numbers:
  learning_delta = warmed - cold      (graph_only nDCG)  -- what usage added
  headroom       = oracle - warmed    (graph_only nDCG)  -- mechanism vs learning
  no_regression  = warmed >= v2 - eps AND zero regressed (control class)
Distributions ({improved/unchanged/regressed}) accompany every comparison so a
mean can't hide a single-query swing (small N)."""
from bench import metrics, selfcheck, conditions


def _snapshot_graph(conn):
    return (conn.execute(
                "SELECT src,dst,kind,weight,updated_at FROM edges").fetchall(),
            conn.execute("SELECT session_id,memory_id,activation,updated_at "
                         "FROM session_members").fetchall())


def _restore_graph(conn, snap):
    edges, members = snap
    conn.execute("DELETE FROM edges")
    conn.executemany("INSERT INTO edges(src,dst,kind,weight,updated_at) "
                     "VALUES (?,?,?,?,?)", edges)
    conn.execute("DELETE FROM session_members")
    conn.executemany("INSERT INTO session_members"
                     "(session_id,memory_id,activation,updated_at) "
                     "VALUES (?,?,?,?)", members)
    conn.commit()


def evaluate(store, id_of, corpus, kind, k=10, budget=None, freeze=False):
    # `freeze`: hold the graph exactly as warm-up left it. recall() learns into
    # the edge table (touch_session), so an ordinary pass mutates the graph as
    # it measures it — query N reshapes what query N+1 sees. When freeze=True we
    # snapshot the graph and restore it after every query, so each query is
    # scored against the identical post-warmup graph (contamination-free).
    snap = _snapshot_graph(store._conn) if freeze else None
    per_query = []
    for i, q in enumerate(corpus.by_kind(kind)):
        # Each query gets its OWN fresh session id. recall() primes from and
        # learns into the session working set, so a shared session would let
        # query N's results reorder query N+1 (priming carryover) — measuring
        # eval-order, not the graph. A unique per-query session isolates the
        # measurement to the warmed graph alone. (The design's "fresh session
        # for the headline" pin, applied per query.)
        ranked = [h["id"] for h in store.recall(
            q.query, limit=k, budget=budget, session=f"eval-{kind}-{i}")]
        if freeze:
            _restore_graph(store._conn, snap)
        gold = {id_of[g] for g in q.gold_keys}
        per_query.append({
            "query": q.query,
            "recall_at_k": metrics.recall_at_k(ranked, gold, k),
            "mrr": metrics.mrr(ranked, gold),
            "ndcg": metrics.ndcg_at_k(ranked, gold, k),
        })
    n = len(per_query) or 1
    mean = {m: sum(r[m] for r in per_query) / n
            for m in ("recall_at_k", "mrr", "ndcg")}
    return {"per_query": per_query, "mean": mean}


def distribution(base_per_query, cond_per_query, eps=0.02):
    out = {"improved": 0, "unchanged": 0, "regressed": 0}
    for b, c in zip(base_per_query, cond_per_query):
        delta = c["ndcg"] - b["ndcg"]
        if delta > eps:
            out["improved"] += 1
        elif delta < -eps:
            out["regressed"] += 1
        else:
            out["unchanged"] += 1
    return out


def run(corpus, embedder, k=10, eps=0.02):
    # 1. build v2 first (needed by the seed-findability check)
    stores = {c: conditions.build(corpus, c, embedder)
              for c in ("v2", "cold", "warmed", "oracle")}
    # 2. self-checks (fail loudly before any number is trusted)
    selfcheck.assert_golds_far(corpus, embedder)
    selfcheck.assert_warmup_disjoint(corpus)
    s_v2, id_v2 = stores["v2"]
    selfcheck.assert_targets_found(corpus, s_v2, id_v2, k=k)

    # 3. evaluate both classes for every condition
    report = {"corpus": corpus.name, "k": k, "conditions": {}}
    for cond, (store, id_of) in stores.items():
        report["conditions"][cond] = {
            "graph_only": evaluate(store, id_of, corpus, "graph_only", k=k),
            "control": evaluate(store, id_of, corpus, "control", k=k),
        }

    go = report["conditions"]  # shorthand
    report["learning_delta_ndcg"] = (
        go["warmed"]["graph_only"]["mean"]["ndcg"]
        - go["cold"]["graph_only"]["mean"]["ndcg"])
    report["headroom_ndcg"] = (
        go["oracle"]["graph_only"]["mean"]["ndcg"]
        - go["warmed"]["graph_only"]["mean"]["ndcg"])

    # 4. no-regression guard on the control class (default budget)
    ctrl_v2 = go["v2"]["control"]["per_query"]
    ctrl_warm = go["warmed"]["control"]["per_query"]
    dist = distribution(ctrl_v2, ctrl_warm, eps=eps)
    mean_ok = (go["warmed"]["control"]["mean"]["ndcg"]
               >= go["v2"]["control"]["mean"]["ndcg"] - eps)
    report["no_regression"] = {
        "distribution": dist,
        "mean_ok": mean_ok,
        "passed": mean_ok and dist["regressed"] == 0,
    }
    # learning-delta distribution too (graph_only, warmed vs cold)
    report["learning_distribution"] = distribution(
        go["cold"]["graph_only"]["per_query"],
        go["warmed"]["graph_only"]["per_query"], eps=eps)

    # 5. held-out (frozen) learning delta — the contamination-free proof.
    # The main loop above evaluated cold/warmed WITHOUT freeze, so those stores
    # learned during eval; rebuild fresh ones and measure them frozen, so every
    # query is scored against the identical post-warmup graph. Same corpus, same
    # warm-up, disjoint eval queries (asserted) -> the delta is transfer, not
    # eval-time memorization.
    cold_s, cold_id = conditions.build(corpus, "cold", embedder)
    warm_s, warm_id = conditions.build(corpus, "warmed", embedder)
    cold_fz = evaluate(cold_s, cold_id, corpus, "graph_only", k=k, freeze=True)
    warm_fz = evaluate(warm_s, warm_id, corpus, "graph_only", k=k, freeze=True)
    report["held_out"] = {
        "cold_frozen_ndcg": cold_fz["mean"]["ndcg"],
        "warmed_frozen_ndcg": warm_fz["mean"]["ndcg"],
        "learning_delta_ndcg": (warm_fz["mean"]["ndcg"]
                                - cold_fz["mean"]["ndcg"]),
        "distribution": distribution(cold_fz["per_query"],
                                     warm_fz["per_query"], eps=eps),
    }
    return report


_HONESTY = (
    "Existence proof on a controlled corpus; small N; NOT a generalization "
    "claim (see corpora B/C, out of scope). The self-check guards semantic "
    "smuggling, not the author shaping task structure toward what Hebbian "
    "captures. A single green number here is evidence, not proof.")


def _print(report):
    print(f"\n=== bench: {report['corpus']} (k={report['k']}) ===")
    print(_HONESTY + "\n")
    for cond in ("v2", "cold", "warmed", "oracle"):
        c = report["conditions"][cond]
        print(f"{cond:>7}  graph_only nDCG={c['graph_only']['mean']['ndcg']:.3f}"
              f"  MRR={c['graph_only']['mean']['mrr']:.3f}"
              f"  R@k={c['graph_only']['mean']['recall_at_k']:.3f}"
              f"   | control nDCG={c['control']['mean']['ndcg']:.3f}")
    print(f"\nlearning delta (warmed-cold, graph_only nDCG): "
          f"{report['learning_delta_ndcg']:+.3f}  "
          f"dist={report['learning_distribution']}")
    ho = report["held_out"]
    print(f"held-out learning delta (FROZEN graph, warmed-cold): "
          f"{ho['learning_delta_ndcg']:+.3f}  "
          f"(cold {ho['cold_frozen_ndcg']:.3f} -> warmed "
          f"{ho['warmed_frozen_ndcg']:.3f})  dist={ho['distribution']}")
    print(f"headroom (oracle-warmed, graph_only nDCG): "
          f"{report['headroom_ndcg']:+.3f}")
    nr = report["no_regression"]
    print(f"no-regression guard (control, default budget): "
          f"{'PASS' if nr['passed'] else 'FAIL'}  dist={nr['distribution']}")


def main():
    # opt-in real run: needs the Model2Vec static model.
    try:
        import model2vec  # noqa: F401
    except ImportError:
        raise SystemExit(
            "bench real run needs model2vec — `pip install model2vec` "
            "(the hermetic suite in tests/test_bench.py needs no model).")
    from tether.embed import get_embedder
    from bench.corpus import SCENARIO
    embedder = get_embedder()
    if embedder is None:
        raise SystemExit("embedder unavailable: Model2Vec model failed to load.")
    report = run(SCENARIO, embedder)
    _print(report)
    return report


if __name__ == "__main__":
    main()
