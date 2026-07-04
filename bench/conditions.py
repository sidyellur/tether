"""Build a fresh Store for each measured condition from the same corpus."""
import sqlite3
from itertools import combinations

from tether.store import Store
from tether.graph import HEBBIAN_CAP
from bench import loader, warmup


def _fresh(embedder, assoc, crystallize=False):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "bench", lambda *a, **k: None, embedder=embedder,
              assoc=assoc, crystallize=crystallize)
    s.migrate()
    return s


# --- crystallization: neutral principle text, deliberately free of any corpus
# vocabulary, so a crystallized principle can help ONLY by structural routing
# (seed -> hub -> distant gold), never by semantically matching the query or the
# gold. assert_principles_far enforces that neutrality against the real embedder.
def principle_title(i):
    return f"principle {i}"


def principle_body(i):
    return (f"Principle {i}: a recurring pattern abstracted from several earlier "
            "notes, kept deliberately free of any specific topic or terminology.")


def build(corpus, condition, embedder):
    if condition == "v2":
        s = _fresh(embedder, assoc=False)
        return s, loader.load(corpus, s)

    crystallize = condition in ("crystallized", "crystallized_detected")
    s = _fresh(embedder, assoc=True, crystallize=crystallize)
    id_of = loader.load(corpus, s)   # semantic + explicit edges built on load

    if condition == "cold":
        return s, id_of
    if condition == "warmed":
        warmup.warm(corpus, s, id_of, repeats=3)
        return s, id_of
    if condition == "oracle":
        _wire_oracle(s, corpus, id_of)
        return s, id_of
    if condition == "crystallized":
        _crystallize_tasks(s, corpus, id_of)
        return s, id_of
    if condition == "crystallized_detected":
        # detection seeds from explicit/hebbian PEAKS; warm the graph first so
        # there is usage structure to detect (base = warmed, not cold).
        warmup.warm(corpus, s, id_of, repeats=3)
        _crystallize_detected(s, embedder)
        return s, id_of
    raise ValueError(f"unknown condition {condition!r}")


def _crystallize_tasks(store, corpus, id_of):
    """Oracle-style: name one principle per ground-truth task and crystallize it
    over that task's members. Bypasses DETECTION to isolate the recall-via-hub
    mechanism — the crystallization analog of _wire_oracle bypassing warm-up.
    This is the ceiling: if hub-routing can reach the topically-distant golds at
    all, it shows up here."""
    for i, task in enumerate(corpus.tasks):
        members = [id_of[k] for k in task.member_keys]
        store.remember("reference", principle_title(i), principle_body(i),
                       crystallizes=members)


def _crystallize_detected(store, embedder):
    """Run the REAL detector on the warmed graph and crystallize a principle over
    each cluster it actually finds. Tests the shipped pipeline, not the ideal —
    detection clusters by explicit-peak + semantic expansion, so this reveals
    whether the detector's clusters align with where the recall value lives.
    Returns the number of principles named."""
    from tether import crystallize
    cands = crystallize.candidates(store._conn, embedder)
    for i, c in enumerate(cands):
        store.remember("reference", principle_title(i), principle_body(i),
                       crystallizes=c["member_ids"])
    return len(cands)


def _wire_oracle(store, corpus, id_of):
    """Ideal graph: every within-task pair wired hebbian at max weight."""
    now = store._conn.execute("SELECT datetime('now')").fetchone()[0]
    for task in corpus.tasks:
        ids = [id_of[k] for k in task.member_keys]
        for a, b in combinations(ids, 2):
            store._graph._upsert_edge(a, b, "hebbian", HEBBIAN_CAP, now, mode="max")
    store._conn.commit()
