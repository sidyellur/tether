# Associative Recall Evaluation Harness (`bench/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `bench/`, a deterministic harness that measures whether associative recall improves with use — reporting a learning delta, a headroom-to-oracle, and a no-regression guard, over a controlled corpus.

**Architecture:** A new additive `bench/` package that imports `tether` and drives `Store`/`Graph` directly (not via MCP). Pure-function metrics + self-checks, a real-recall-path warm-up, four condition builders, and a runner that prints a report. Nothing in `src/tether/` changes.

**Tech Stack:** Python 3, `sqlite3` (in-memory DBs), `numpy` (cosine in the self-check), Model2Vec (`minishlab/potion-base-8M`) for the real run, `pytest` for the hermetic smoke/unit tests with a `FakeEmbedder`.

**Design spec:** `docs/superpowers/specs/2026-07-04-tether-bench-associative-eval-design.md`

## Global Constraints

- **Additive only** — no file under `src/tether/` is modified. The harness is read-only against the public API, plus `Graph._upsert_edge` for the oracle (the seam the existing tests already use).
- **Deterministic & reproducible** — no `Date.now()`/random ordering in scored paths; warm-up replay is a fixed sequence; static embeddings have no randomness. Same corpus + code ⇒ same numbers.
- **Measurement, not a CI gate** — the full real-corpus run is opt-in (`importorskip("model2vec")`); only a hermetic smoke test + unit tests run in the default suite.
- **Hermetic tests** — every test in `tests/test_bench.py` uses the in-file `FakeEmbedder` (no model download, no network). Guard numpy-dependent paths with `pytest.importorskip("numpy")`.
- **Tier A API (verbatim, do not re-derive):**
  - `Store(conn, device_id, sync_now, embedder=None, author="", consolidate=False, dedup_threshold=0.92, decay_half_life_days=None, assoc=False, recall_budget=24, ...)`.
  - `store.migrate()`; `store.remember(type, title, body, tags="") -> dict` (has `["id"]`); `store.link(id_a, id_b)`; `store.recall(query, type=None, limit=20, budget=None, session=None) -> list[dict]` where each hit is `{"id","type","title","body","tags","updated_at", "via"?}`.
  - `store._graph` is a `Graph`; `Graph._upsert_edge(a, b, kind, weight, now, mode="max")`; kinds `"semantic"|"explicit"|"hebbian"`.
  - `from tether.graph import HEBBIAN_CAP` (== 5.0). Default recall budget `24`.
- **Corpus authoring gate:** the real corpus is accepted only when both self-checks pass (golds far, targets found) — that is its definition of done, not a line count (though target ≥ 30 memories).

## File Structure

- `bench/__init__.py` — empty package marker.
- `bench/metrics.py` — pure ranking metrics: `recall_at_k`, `mrr`, `ndcg_at_k`. No tether imports.
- `bench/corpus.py` — dataclasses (`Memory`, `Task`, `Query`, `Corpus`) + the `MINI` test fixture + (Task 7) the real `SCENARIO` corpus. Data only.
- `bench/loader.py` — `load(corpus, store) -> dict[str,int]`: remembers each memory, applies links, returns a corpus-key → db-id map.
- `bench/selfcheck.py` — `assert_golds_far`, `assert_targets_found`.
- `bench/warmup.py` — `warm(store, corpus, id_of, repeats=3)`.
- `bench/conditions.py` — `build(corpus, condition, embedder) -> tuple[Store, dict[str,int]]` for `"v2"|"cold"|"warmed"|"oracle"`.
- `bench/run.py` — `evaluate`, `run`, `main`; the report.
- `tests/test_bench.py` — hermetic unit + smoke tests.

Helpers shared by tests (a fresh in-memory store) live inline in `tests/test_bench.py`, mirroring the existing `tests/test_store.py` pattern.

---

### Task 1: Pure ranking metrics

**Files:**
- Create: `bench/__init__.py`, `bench/metrics.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `recall_at_k(ranked: list[int], gold: set[int], k: int) -> float` — 1.0 if any gold in first `k`, else 0.0.
  - `mrr(ranked: list[int], gold: set[int]) -> float` — reciprocal rank of the first gold (0.0 if none).
  - `ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float` — binary-gain nDCG over top `k`, normalized by the ideal ordering.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench.py
import math
from bench import metrics


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench'`.

- [ ] **Step 3: Implement the metrics**

```python
# bench/__init__.py
```

```python
# bench/metrics.py
"""Pure ranking metrics. No tether/embedding imports — plain arithmetic over
a ranked list of ids and a gold set."""
import math


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    return 1.0 if any(mid in gold for mid in ranked[:k]) else 0.0


def mrr(ranked, gold):
    if not gold:
        return 0.0
    for i, mid in enumerate(ranked):
        if mid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    dcg = 0.0
    for i, mid in enumerate(ranked[:k]):
        if mid in gold:
            dcg += 1.0 / math.log2(i + 2)  # rank i (0-based) -> discount log2(i+2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bench/__init__.py bench/metrics.py tests/test_bench.py
git commit -m "feat(bench): pure ranking metrics (recall@k, mrr, ndcg)"
```

---

### Task 2: Corpus schema + mini fixture

**Files:**
- Create: `bench/corpus.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass Memory(key: str, type: str, title: str, body: str)`
  - `@dataclass Task(key: str, member_keys: list[str])`
  - `@dataclass Query(query: str, target_key: str, gold_keys: list[str], kind: str)` — `kind` in `{"graph_only", "control"}`.
  - `@dataclass Corpus(name: str, memories: list[Memory], tasks: list[Task], links: list[tuple[str, str]], queries: list[Query])`
  - `Corpus.by_kind(kind) -> list[Query]`, `Corpus.member_of_task(mem_key) -> set[str]`
  - `MINI: Corpus` — a tiny fixture (4 memories, 1 task, 1 graph_only query, 1 control query) used by hermetic tests. Its graph_only gold shares no obvious tokens with its query; its control gold IS the directly-matched target.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_bench.py
from bench import corpus as corpus_mod


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py::test_mini_corpus_shape -v`
Expected: FAIL with `ImportError` / `AttributeError: MINI`.

- [ ] **Step 3: Implement the schema + MINI fixture**

```python
# bench/corpus.py
"""Corpus data model + the MINI hermetic fixture.

A corpus is authored with string `key`s (stable, human-readable); the loader
maps keys -> db ids at load time. Two query classes:
  - graph_only: gold is relevant but semantically FAR from the query
    (reachable only via behavioral edges) -> measures UPSIDE.
  - control:    gold IS the directly-matched target that v0.2 ranks well
    -> measures NO-REGRESSION (assoc must not demote it)."""
from dataclasses import dataclass, field


@dataclass
class Memory:
    key: str
    type: str
    title: str
    body: str


@dataclass
class Task:
    key: str
    member_keys: list


@dataclass
class Query:
    query: str
    target_key: str
    gold_keys: list
    kind: str  # "graph_only" | "control"


@dataclass
class Corpus:
    name: str
    memories: list
    tasks: list
    links: list = field(default_factory=list)   # list of (key_a, key_b)
    queries: list = field(default_factory=list)

    def by_kind(self, kind):
        return [q for q in self.queries if q.kind == kind]

    def member_of_task(self, mem_key):
        out = set()
        for t in self.tasks:
            if mem_key in t.member_keys:
                out.update(t.member_keys)
        out.discard(mem_key)
        return out


# --- MINI: hermetic fixture. Keep tiny; the FakeEmbedder makes cosine trivial,
# so "far" here just means "distinct key vectors", not real semantics. ---
MINI = Corpus(
    name="mini",
    memories=[
        Memory("bug", "note", "auth 500s", "login returns 500 under load"),
        Memory("pref", "note", "dave orm stance", "dave distrusts the ORM layer"),
        Memory("editor", "note", "editor choice", "team standardized on neovim"),
        Memory("car", "note", "commute", "i drive my car to the office"),
    ],
    tasks=[Task("auth", ["bug", "pref"])],
    links=[],
    queries=[
        Query("why did login break", "bug", ["pref"], "graph_only"),
        Query("login returns 500 under load", "bug", ["bug"], "control"),
    ],
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py::test_mini_corpus_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/corpus.py tests/test_bench.py
git commit -m "feat(bench): corpus schema (Memory/Task/Query/Corpus) + MINI fixture"
```

---

### Task 3: Corpus loader

**Files:**
- Create: `bench/loader.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Corpus` (Task 2); `Store` (Tier A).
- Produces: `load(corpus: Corpus, store: Store) -> dict[str, int]` — remembers every memory (in list order), applies every link, returns `{memory_key: db_id}`. Assumes `store.migrate()` already called.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench.py
import sqlite3
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py::test_loader_maps_keys_to_ids_and_links -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.loader'`.

- [ ] **Step 3: Implement the loader**

```python
# bench/loader.py
"""Load a Corpus into a Store, returning a corpus-key -> db-id map."""


def load(corpus, store):
    id_of = {}
    for m in corpus.memories:
        rec = store.remember(m.type, m.title, m.body)
        id_of[m.key] = rec["id"]
    for (a, b) in corpus.links:
        store.link(id_of[a], id_of[b])
    return id_of
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py::test_loader_maps_keys_to_ids_and_links -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/loader.py tests/test_bench.py
git commit -m "feat(bench): corpus loader (keys -> db ids, applies links)"
```

---

### Task 4: Self-checks (anti-rigging + seed-findability)

**Files:**
- Create: `bench/selfcheck.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Corpus`, an embedder with `.encode([...]) -> np.ndarray` of unit vectors, a v0.2 `Store`, a loaded `id_of` map.
- Produces:
  - `assert_golds_far(corpus, embedder, threshold=0.35) -> None` — for every `graph_only` (query, gold) pair, cosine(query, gold.body) must be `< threshold`, else `raise AssertionError` naming the pair.
  - `assert_targets_found(corpus, store_v2, id_of, k=10) -> None` — for every query (both kinds), `id_of[target]` must appear in `store_v2.recall(query, limit=k)`, else `raise AssertionError` naming the query.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_bench.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py -k assert_ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.selfcheck'`.

- [ ] **Step 3: Implement the self-checks**

```python
# bench/selfcheck.py
"""Pre-flight assertions that make the benchmark honest:
  - assert_golds_far: graph_only golds must be semantically FAR from their
    query, so any lift is the graph's, not smuggled semantic overlap.
  - assert_targets_found: every query's target must be retrievable by v0.2,
    so a condition never underperforms merely because the SEED was missed."""


def _cos(embedder, a, b):
    import numpy as np
    va, vb = embedder.encode([a, b])
    return float(np.dot(va, vb))


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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py -k assert_ -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bench/selfcheck.py tests/test_bench.py
git commit -m "feat(bench): symmetric self-checks (golds far, targets found)"
```

---

### Task 5: Warm-up via the real recall path

**Files:**
- Create: `bench/warmup.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: a warmed-capable `Store` (`assoc=True`), a `Corpus`, its `id_of` map.
- Produces: `warm(store, corpus, id_of, repeats=3) -> None` — for each task, `repeats` times, opens a task-specific session and issues one `recall(query=<member title>, session=<task session>)` per member so the members co-occur and get Hebbian-wired by `touch_session`. Deterministic; uses session ids of the form `f"warm:{task.key}"` — never the eval session.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py::test_warm_creates_hebbian_edges -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.warmup'`.

- [ ] **Step 3: Implement warm-up**

```python
# bench/warmup.py
"""Warm the usage graph through the REAL recall path: replay each task as a
session of recalls so its members co-occur and get Hebbian-wired by
touch_session. No hand-wiring — this is the mechanism under test."""


def warm(corpus, store, id_of, repeats=3):
    title_of = {m.key: m.title for m in corpus.memories}
    for _ in range(repeats):
        for task in corpus.tasks:
            sid = f"warm:{task.key}"
            for mkey in task.member_keys:
                # a query that directly surfaces this member, in the task
                # session, so all members land in the session working set.
                store.recall(title_of[mkey], session=sid)
```

Note the arg order: `warm(corpus, store, id_of, repeats)` — corpus first (matches the test).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py::test_warm_creates_hebbian_edges -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/warmup.py tests/test_bench.py
git commit -m "feat(bench): real-recall-path warm-up (co-recall -> hebbian edges)"
```

---

### Task 6: Condition builders

**Files:**
- Create: `bench/conditions.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Corpus`, `load` (Task 3), `warm` (Task 5), `Store`/`Graph`, `HEBBIAN_CAP`.
- Produces: `build(corpus, condition, embedder) -> tuple[Store, dict[str, int]]` for `condition` in `{"v2", "cold", "warmed", "oracle"}`:
  - `"v2"` — `assoc=False`. Baseline.
  - `"cold"` — `assoc=True`, loaded (semantic+explicit edges auto-built), no warm-up.
  - `"warmed"` — `"cold"` then `warm(...)`.
  - `"oracle"` — `assoc=True`, loaded, then hand-wire a full within-task Hebbian clique at `HEBBIAN_CAP` (no cross-task edges).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py::test_build_conditions_edge_states -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.conditions'`.

- [ ] **Step 3: Implement condition builders**

```python
# bench/conditions.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py::test_build_conditions_edge_states -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/conditions.py tests/test_bench.py
git commit -m "feat(bench): four condition builders (v2/cold/warmed/oracle)"
```

---

### Task 7: Runner, report, and the no-regression guard

**Files:**
- Create: `bench/run.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `evaluate(store, id_of, corpus, kind, k=10, budget=None) -> dict` — for each query of `kind`, runs `store.recall(q.query, limit=k, budget=budget)`, maps its target/golds via `id_of`, and returns `{"per_query": [ {query, recall_at_k, mrr, ndcg} ], "mean": {recall_at_k, mrr, ndcg}}`.
  - `distribution(base_per_query, cond_per_query, eps=0.02) -> dict` — `{improved, unchanged, regressed}` counts comparing per-query nDCG (paired by query index).
  - `run(corpus, embedder, k=10, eps=0.02) -> dict` — runs self-checks, builds the four conditions, evaluates both classes, computes learning delta / headroom / no-regression guard + distributions, returns a structured report.
  - `main()` — builds the real embedder (opt-in), runs `SCENARIO`, prints the report with the honesty header.
- **Pins:** the no-regression guard runs at the product default budget (`budget=None` ⇒ `recall_budget=24`); control gold is the v0.2-findable target, so the guard asserts assoc does not *demote* it (warmed nDCG ≥ v2 nDCG − eps, **and** zero regressed in the distribution).

- [ ] **Step 1: Write the failing tests (hermetic smoke + guard)**

```python
# add to tests/test_bench.py
def test_run_smoke_all_conditions_both_classes():
    import pytest
    pytest.importorskip("numpy")
    from bench import run, corpus as corpus_mod
    rep = run.run(corpus_mod.MINI, FakeEmbedder(), k=5)
    # all four conditions present, both classes measured
    for cond in ("v2", "cold", "warmed", "oracle"):
        assert cond in rep["conditions"]
        assert "graph_only" in rep["conditions"][cond]
        assert "control" in rep["conditions"][cond]
    # derived numbers exist and are numbers
    assert isinstance(rep["learning_delta_ndcg"], float)
    assert isinstance(rep["headroom_ndcg"], float)
    assert "no_regression" in rep and "passed" in rep["no_regression"]


def test_distribution_counts():
    from bench import run
    base = [{"ndcg": 0.5}, {"ndcg": 0.5}, {"ndcg": 0.5}]
    cond = [{"ndcg": 0.9}, {"ndcg": 0.5}, {"ndcg": 0.1}]  # up / flat / down
    d = run.distribution(base, cond, eps=0.02)
    assert d == {"improved": 1, "unchanged": 1, "regressed": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bench.py -k "run_smoke or distribution_counts" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.run'`.

- [ ] **Step 3: Implement the runner**

```python
# bench/run.py
"""Run the four conditions over both query classes and print the report.

Headline numbers:
  learning_delta = warmed - cold      (graph_only nDCG)  -- what usage added
  headroom       = oracle - warmed    (graph_only nDCG)  -- mechanism vs learning
  no_regression  = warmed >= v2 - eps AND zero regressed (control class)
Distributions ({improved/unchanged/regressed}) accompany every comparison so a
mean can't hide a single-query swing (small N)."""
from bench import metrics, selfcheck, conditions


def evaluate(store, id_of, corpus, kind, k=10, budget=None):
    per_query = []
    for q in corpus.by_kind(kind):
        ranked = [h["id"] for h in store.recall(q.query, limit=k, budget=budget)]
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
    print(f"headroom (oracle-warmed, graph_only nDCG): "
          f"{report['headroom_ndcg']:+.3f}")
    nr = report["no_regression"]
    print(f"no-regression guard (control, default budget): "
          f"{'PASS' if nr['passed'] else 'FAIL'}  dist={nr['distribution']}")


def main():
    import pytest
    pytest.importorskip("model2vec")  # opt-in real run
    from tether.embed import Embedder
    from bench.corpus import SCENARIO
    report = run(SCENARIO, Embedder())
    _print(report)
    return report


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bench.py -k "run_smoke or distribution_counts" -v`
Expected: PASS.

- [ ] **Step 5: Run the full hermetic suite**

Run: `pytest tests/test_bench.py -v`
Expected: PASS (all tasks' tests green).

- [ ] **Step 6: Commit**

```bash
git add bench/run.py tests/test_bench.py
git commit -m "feat(bench): runner + report + no-regression guard + distributions"
```

---

### Task 8: Author the real scenario corpus, run it, record the numbers

**Files:**
- Modify: `bench/corpus.py` (add `SCENARIO: Corpus`)
- Docs: append results to the tether blog journal (via tether, in a local session)

**Interfaces:**
- Consumes: the whole harness.
- Produces: `SCENARIO: Corpus` — the real hand-authored corpus, and a recorded run.

**This is data authoring, gated by the self-checks — not free-form.** Requirements:
- **≥ 30 memories**, real prose, one coherent fictional software team (decisions, bugs, preferences, people, infra). Distinct vocabulary across topics so semantics are honest.
- **Tasks** grouping memories used together (≥ 5 tasks, 2–5 members each). A memory may be in multiple tasks.
- **graph_only queries** (≥ 10): target a memory directly; golds are the *other* task members, authored so cosine(query, gold.body) `< 0.35` under the **real** embedder (the acceptance gate — `assert_golds_far` must pass).
- **control queries** (≥ 10): `gold_keys == [target_key]` (the pin) — a memory v0.2 already ranks in top-k; used to prove assoc does not demote it. `assert_targets_found` must pass for *all* queries.
- Optional `links`: only where a human would genuinely `link()` two memories.
- A comment block documents each task's intent and why its graph_only golds are non-matching. Authored from the scenario, **not tuned to the result**.

- [ ] **Step 1: Draft `SCENARIO` in `bench/corpus.py`** following the requirements above (schema identical to `MINI`). Keep the vocab-per-topic distinct so far-golds are achievable.

- [ ] **Step 2: Verify the self-checks pass on the real corpus**

Run:
```bash
python -c "from tether.embed import Embedder; from bench import selfcheck, loader, conditions; from bench.corpus import SCENARIO; \
e=Embedder(); s,ido=conditions.build(SCENARIO,'v2',e); \
selfcheck.assert_golds_far(SCENARIO,e); selfcheck.assert_targets_found(SCENARIO,s,ido); print('self-checks OK')"
```
Expected: `self-checks OK`. If either raises, revise the offending memory/query (this is the gate — do not weaken the thresholds to pass).

- [ ] **Step 3: Run the harness and capture the report**

Run: `python -m bench.run`
Expected: a report table + learning delta + headroom + no-regression guard. Record the exact numbers.

- [ ] **Step 4: Commit the corpus**

```bash
git add bench/corpus.py
git commit -m "feat(bench): real scenario corpus (self-checks passing)"
```

- [ ] **Step 5: Record the measured numbers in the tether blog journal**

In a local session (tether reachable), recall the `tether blog journal`, append a dated entry with: learning delta, headroom, no-regression result, the {improved/unchanged/regressed} distributions, and an honest read (does the central bet hold on this corpus?). Remember it back (full body — never a delta).

---

## Self-Review

**1. Spec coverage:**
- Corpus (two query classes, real embedder, frozen/documented) → Tasks 2, 8. ✅
- Anti-rigging + seed-findability self-checks → Task 4. ✅
- Real-recall-path warm-up + fresh eval session → Task 5 (warm uses `warm:<task>` sessions; eval in Task 7 uses no session ⇒ neutral, so priming can't leak). ✅
- Four conditions incl. precisely-defined oracle (full within-task clique at `HEBBIAN_CAP`) → Task 6. ✅
- Metrics (Recall@k, MRR, nDCG) → Task 1. ✅
- Learning delta / headroom / no-regression guard → Task 7. ✅
- Distribution ({improved/unchanged/regressed}) → Task 7 (`distribution`, printed for both learning and control). ✅
- Honesty header (both caveats) → Task 7 (`_HONESTY`). ✅
- Smoke + unit tests, hermetic, opt-in real run → Tasks 1–7 tests + `main()` `importorskip`. ✅
- Additive, nothing in `src/tether/` changes → all tasks create under `bench/` / `tests/`. ✅
- **Plan pins:** control gold = target (Task 2 test asserts `gold_keys == [target]`; Task 8 requires it); guard at default budget (Task 7 `budget=None`). ✅

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code. Task 8 is data-authoring with an explicit acceptance gate (self-checks pass) rather than inlined prose — the one place inlining 30+ memories is neither possible nor useful; the gate makes "done" objective.

**3. Type consistency:** `warm(corpus, store, id_of, repeats)` arg order consistent between Task 5 impl/test and Task 6 caller. `build(corpus, condition, embedder) -> (Store, id_of)` consistent Tasks 6–7. `evaluate(...) -> {"per_query","mean"}` and `distribution(base, cond, eps)` consistent Task 7. `id_of` is `{key:int}` throughout. Metric names (`recall_at_k`/`mrr`/`ndcg_at_k`) consistent; note the per-query dict key is `"ndcg"` (short) while the metric function is `ndcg_at_k` — intentional and used consistently in Task 7.

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute here with batch checkpoints via executing-plans.

Which approach?
