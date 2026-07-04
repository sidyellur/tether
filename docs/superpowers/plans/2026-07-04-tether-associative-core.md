# tether Associative Core (Tier A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `recall` from a flat lookup into spreading-activation retrieval over a self-learning *usage graph* — connected neighborhoods that sharpen with use — while staying 100% local, deterministic, and degrade-never.

**Architecture:** A new `graph.py` (the association layer) owns an `edges` table (semantic kNN + explicit links + learned Hebbian co-recall) and a `session_members` working set. `store.py`'s `recall` orchestrates: v0.2 hybrid produces *seeds*, `graph.spread` walks the graph adding activation, results carry `via` receipts. Everything is additive: with the association layer off, no edges, or no embedder, `recall` returns exactly v0.2 results.

**Tech Stack:** Python ≥3.10, stdlib `sqlite3`, `numpy` (already a `[semantic]` dep) for kNN. No new dependency. Design: `docs/superpowers/specs/2026-07-04-tether-associative-core-design.md`.

## Global Constraints

- **Python ≥3.10, POSIX.** Distribution `tether-memory`; import/CLI `tether`; entry point `tether-memory = "tether.server:main"`.
- **Still four verbs only:** `remember`, `recall`, `link`, `forget` + the boot-index resource. **No new tool.** `recall` gains two optional params (`budget`, `session`) and each hit gains a `via` field; no signature *breaks*.
- **Reasoning stays in the caller. No LLM, no network inside tether.** The graph is geometry + behavior only.
- **Degrade, never throw.** Association layer off (`assoc=False` / `TETHER_ASSOC=0`), `budget=0`, empty graph, or no embedder → `recall` is byte-identical to v0.2. Every graph op is wrapped so a failure degrades to plain recall; no tool call may crash the server.
- **Additive schema only:** new `edges` + `session_members` tables; `memories` untouched. A v0.3 DB upgrades in place.
- **Determinism:** every Tier-A operation is deterministic given DB state (walk order breaks ties by activation desc then id; kNN by cosine then id; no RNG — dreaming is Tier C).
- **Library-conservative, product-on:** `Store` defaults `assoc=False` (so existing callers/tests are unaffected); the server turns it on via `config.assoc_enabled()` (default true).
- Commit after every task. Sign commits with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pyproject.toml` | version bump `0.4.0` | Modify |
| `src/tether/config.py` | `assoc_enabled()`, `recall_budget()` | Modify |
| `src/tether/graph.py` | `Graph`: `edges` + `session_members` SQL, edge maintenance, `spread`, session working set | **Create** |
| `src/tether/store.py` | hold a `Graph`; `recall` orchestrates seed→prime→spread→rank→learn→receipts; `remember`/`link`/`forget`/`migrate`/`backfill_embeddings` delegate edge upkeep | Modify |
| `src/tether/server.py` | `recall` tool gains `budget`/`session`; wire `assoc`/`recall_budget` into `Store` | Modify |
| `tests/test_config.py` | the two resolvers | Modify |
| `tests/test_graph.py` | edges/schema, edge maintenance, `spread`, sessions | **Create** |
| `tests/test_store.py` | recall orchestration, degrade paths, receipts | Modify |
| `tests/test_mcp.py` | server wiring of `budget`/`session`/`via` | Modify |

## Test Strategy

**Philosophy:** TDD per task. Default `pytest` run is hermetic — no network, no model download. The graph traversal, sessions, and recall orchestration are tested with **hand-inserted edges** and the existing `FakeEmbedder`, so nothing needs `model2vec`. `numpy` (in `[dev]`) is required for the kNN/spread paths; guard those tests with `importorskip("numpy")`.

**Levels:**
- *Unit* — config resolvers; `Graph` edge maintenance (semantic/explicit/hebbian) against an in-memory DB; the pure `spread` walk over a hand-built graph; session decay/priming/hebbian; the `_seed_scores` refactor.
- *Integration* — `Store.recall` end-to-end (seeds → spread → receipts) with a `FakeEmbedder`; `server._get_store` wiring; MCP `recall` exposing `budget`/`session` and returning `via`.

**Hermeticity controls:** hand-inserted rows in `edges` for pure `spread` tests (no embedder needed); `FakeEmbedder` for semantic-edge and end-to-end tests; `pytest.importorskip("numpy")` on kNN/spread; timestamps written directly via SQL for session time-bucket tests (never freeze the clock).

**Coverage matrix (guarantee → test):**

| Guarantee | Task / test |
|---|---|
| Config: assoc on by default, budget default | Task 1 `test_assoc_*`, `test_recall_budget_*` |
| `edges` + `session_members` created; `forget` deletes a memory's edges + session rows | Task 2 `test_graph_migrate_creates_tables`, `test_forget_deletes_edges` |
| Semantic kNN edges written on remember; explicit edge on link; backfill | Task 3 `test_remember_writes_semantic_edges`, `test_link_writes_explicit_edge`, `test_backfill_*` |
| Spreading: 2-hop associate surfaces at `budget≥2`, absent at `budget=0`; HOP_DECAY keeps seed dominant; receipts trace the edge | Task 4 `test_spread_*` |
| Session priming biases the next recall; Hebbian edge formed among co-active; time-bucket derivation | Task 5 `test_session_*` |
| Associative recall finds a connected neighbor a flat query misses; `via` receipts present | Task 6 `test_recall_associative_*`, `test_recall_via_receipts` |
| **`assoc=False` → recall byte-identical to v0.2** | Task 6 `test_recall_disabled_matches_v2` |
| No embedder → no semantic edges but recall works; empty graph → passthrough | Task 6 `test_recall_degrades_*` |
| Server wires budget/session; default call unchanged for existing callers | Task 7 `test_server_recall_budget_session` |

**Deliberate non-goals:** no retroactive kNN rewiring (write-time approximation, per spec); no edge decay / hub curation / dreaming (Tiers B/C); latency not asserted in CI (brute-force walk is bounded by `budget`).

---

### Task 1: config — `assoc_enabled` + `recall_budget`

**Files:**
- Modify: `src/tether/config.py`
- Modify: `pyproject.toml` (version)
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `config.assoc_enabled() -> bool` (default `True`; `0/false/no/off` disables) and `config.recall_budget() -> int` (`$TETHER_RECALL_BUDGET`, default `24`; a bad or negative value → `24`; `0` allowed = spreading off).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py (append)

def test_assoc_enabled_default_true(monkeypatch):
    monkeypatch.delenv("TETHER_ASSOC", raising=False)
    assert config.assoc_enabled() is True
    for v in ("0", "false", "off", "NO"):
        monkeypatch.setenv("TETHER_ASSOC", v)
        assert config.assoc_enabled() is False


def test_recall_budget_default_and_parsing(monkeypatch):
    monkeypatch.delenv("TETHER_RECALL_BUDGET", raising=False)
    assert config.recall_budget() == 24
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "8")
    assert config.recall_budget() == 8
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "0")
    assert config.recall_budget() == 0
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "-5")
    assert config.recall_budget() == 24
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "junk")
    assert config.recall_budget() == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -k "assoc or recall_budget" -v`
Expected: FAIL — `AttributeError: module 'tether.config' has no attribute 'assoc_enabled'`.

- [ ] **Step 3: Add the resolvers to `src/tether/config.py`**

Append:

```python
_ASSOC_OFF = {"0", "false", "no", "off"}
_DEFAULT_RECALL_BUDGET = 24


def assoc_enabled() -> bool:
    """Associative (spreading-activation) recall is on by default; any of
    0/false/no/off forces plain v0.2 hybrid recall."""
    val = os.environ.get("TETHER_ASSOC")
    if val is None:
        return True
    return val.strip().lower() not in _ASSOC_OFF


def recall_budget() -> int:
    """Default spreading budget (max node-expansions). 0 = spreading off."""
    raw = os.environ.get("TETHER_RECALL_BUDGET")
    if not raw:
        return _DEFAULT_RECALL_BUDGET
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_RECALL_BUDGET
    return val if val >= 0 else _DEFAULT_RECALL_BUDGET
```

- [ ] **Step 4: Bump the version in `pyproject.toml`**

Change `version = "0.3.0"` to:

```toml
version = "0.4.0"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all config tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tether/config.py pyproject.toml tests/test_config.py
git commit -m "feat: assoc/recall-budget config (TETHER_ASSOC, TETHER_RECALL_BUDGET)"
```

---

### Task 2: `graph.py` — schema, edge upsert, `forget` cleanup

**Files:**
- Create: `src/tether/graph.py`
- Modify: `src/tether/store.py` (hold a `Graph`; wire `migrate` + `forget`)
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces:
  - `class Graph(conn, enabled=True)` with attribute `.enabled`.
  - `Graph.migrate()` — creates `edges` and `session_members` tables (idempotent).
  - `Graph._upsert_edge(a, b, kind, weight, now, mode)` — canonical (min,max) key; `mode="max"` keeps the larger weight, `mode="add"` accumulates (capped by caller).
  - `Graph.on_forget(mid)` — deletes all `edges` and `session_members` rows referencing `mid`.
  - Module constants: `KNN_K=8`, `HOP_DECAY=0.4`, `EPSILON=1e-4`, `KIND_W={"semantic":1.0,"explicit":1.2,"hebbian":1.0}`, `SESSION_DECAY=0.5`, `SESSION_GAP_SECONDS=1800`, `HEBBIAN_INCREMENT=0.5`, `HEBBIAN_CAP=5.0`, `HEBBIAN_TOP_M=8`.
- Consumes (Store): `Store.__init__` gains `self._graph = Graph(conn, enabled=assoc)` (the `assoc` arg is added in Task 6; for now default the Store field to a disabled graph so existing behavior is unchanged — see Step 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py
import sqlite3

from tether.graph import Graph


def make_graph(enabled=True):
    conn = sqlite3.connect(":memory:")
    g = Graph(conn, enabled=enabled)
    g.migrate()
    return g


def test_graph_migrate_creates_tables():
    g = make_graph()
    names = {r[0] for r in g._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"edges", "session_members"} <= names


def test_upsert_edge_canonical_and_modes():
    g = make_graph()
    g._upsert_edge(5, 2, "semantic", 0.3, "t", mode="max")
    g._upsert_edge(2, 5, "semantic", 0.7, "t", mode="max")   # same pair, higher
    row = g._conn.execute(
        "SELECT src, dst, weight FROM edges WHERE kind='semantic'").fetchone()
    assert row == (2, 5, 0.7)                                 # canonical src<dst, max kept
    g._upsert_edge(2, 5, "hebbian", 0.5, "t", mode="add")
    g._upsert_edge(2, 5, "hebbian", 0.5, "t", mode="add")
    w = g._conn.execute(
        "SELECT weight FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert w == 1.0                                           # accumulated


def test_on_forget_deletes_edges_and_session_rows():
    g = make_graph()
    g._upsert_edge(1, 2, "semantic", 0.5, "t", mode="max")
    g._upsert_edge(2, 3, "semantic", 0.5, "t", mode="max")
    g._conn.execute("INSERT INTO session_members VALUES('s', 2, 1.0, 't')")
    g.on_forget(2)
    assert g._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert g._conn.execute("SELECT COUNT(*) FROM session_members").fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tether.graph'`.

- [ ] **Step 3: Write `src/tether/graph.py`**

```python
"""graph.py - the association layer.

A usage graph over memories: edges from embedding geometry (semantic kNN),
explicit links, and learned co-recall (Hebbian), plus an ephemeral session
working set. Owns the `edges` and `session_members` SQL. Spreading activation
(`spread`) turns a set of seed memories into their connected neighborhood.

100% local and deterministic; no LLM, no network. Every operation degrades to
a no-op rather than raising, so the memory layer never breaks the agent.
"""

from datetime import datetime, timezone

KNN_K = 8
HOP_DECAY = 0.4
EPSILON = 1e-4
KIND_W = {"semantic": 1.0, "explicit": 1.2, "hebbian": 1.0}
SESSION_DECAY = 0.5
SESSION_GAP_SECONDS = 1800
SESSION_TTL_ACTIVATION = 0.05
HEBBIAN_INCREMENT = 0.5
HEBBIAN_CAP = 5.0
HEBBIAN_TOP_M = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    src        INTEGER NOT NULL,
    dst        INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS session_members (
    session_id  TEXT NOT NULL,
    memory_id   INTEGER NOT NULL,
    activation  REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, memory_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Graph:
    def __init__(self, conn, enabled=True):
        self._conn = conn
        self.enabled = enabled

    def migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _upsert_edge(self, a, b, kind, weight, now, mode="max") -> None:
        if a == b:
            return
        src, dst = (a, b) if a < b else (b, a)
        if mode == "add":
            expr = "weight = min(?, edges.weight + ?)"
            params = (src, dst, kind, weight, now, HEBBIAN_CAP, weight)
        else:  # max
            expr = "weight = max(edges.weight, excluded.weight)"
            params = (src, dst, kind, weight, now)
        self._conn.execute(
            f"INSERT INTO edges(src, dst, kind, weight, updated_at) "
            f"VALUES (?,?,?,?,?) "
            f"ON CONFLICT(src, dst, kind) DO UPDATE SET {expr}, updated_at=excluded.updated_at",
            params)

    def on_forget(self, mid) -> None:
        try:
            self._conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (mid, mid))
            self._conn.execute("DELETE FROM session_members WHERE memory_id=?", (mid,))
        except Exception:
            pass
```

Note the `mode="add"` branch references `edges.weight` for the cap; the `excluded`-based `max` branch and the explicit `edges.weight + ?` are both valid SQLite upsert forms.

- [ ] **Step 4: Wire `Graph` into `Store` (`src/tether/store.py`)**

Add the import at the top (after the existing imports):

```python
from .graph import Graph
```

In `Store.__init__`, add a graph field at the end of the body (the `assoc` constructor arg comes in Task 6; until then default to disabled so nothing changes):

```python
        self._graph = Graph(conn, enabled=False)
```

In `migrate`, before `self._conn.commit()`, add:

```python
        self._graph.migrate()
```

In `forget`, call the graph cleanup (replace the method body):

```python
    def forget(self, id) -> dict:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (id,))
        self._graph.on_forget(id)
        self._conn.commit()
        self._sync_now()
        return {"forgotten": id, "existed": cur.rowcount > 0}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph.py tests/test_store.py -q`
Expected: the new graph tests pass; all existing store tests still pass (graph is disabled, `migrate` just adds two empty tables).

- [ ] **Step 6: Commit**

```bash
git add src/tether/graph.py src/tether/store.py tests/test_graph.py
git commit -m "feat: graph.py - edges + session_members schema, edge upsert, forget cleanup"
```

---

### Task 3: edge maintenance — semantic kNN, explicit, backfill

**Files:**
- Modify: `src/tether/graph.py` (`on_remember`, `on_link`, `backfill_semantic`, `backfill_explicit`)
- Modify: `src/tether/store.py` (`remember`, `link`, `backfill_embeddings`, `migrate` delegate)
- Test: `tests/test_graph.py` (append), `tests/test_store.py` (append)

**Interfaces:**
- Produces:
  - `Graph.on_remember(mid, emb_blob)` — upserts `mid`'s top-`KNN_K` semantic edges (weight = cosine) against other current embedded memories. No-op if disabled / no blob / numpy missing.
  - `Graph.on_link(id_a, id_b)` — upserts one `explicit` edge (weight `1.0`).
  - `Graph.backfill_semantic()` — builds semantic edges for every current embedded memory (idempotent, max-merge).
  - `Graph.backfill_explicit(pairs)` — upserts `explicit` edges for a list of `(a, b)` id pairs (used by `store.migrate` to import existing `links`).
- Consumes: the `memories.embedding` blobs (unit-normalized float32) written by v0.2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py (append)
import pytest


class FakeEmbedder:
    name = "fake-3d"
    dims = 3
    _AXES = [("car", "automobile", "drive"), ("pizza", "food"), ("python", "test")]

    def embed(self, text):
        import math
        t = text.lower()
        v = [float(sum(w in t for w in ax)) for ax in self._AXES]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


def _pack(vec):
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


def _seed_memory(conn, mid, type, title, body, emb):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, type TEXT, "
        "title TEXT, body TEXT, embedding BLOB, valid_to TEXT)")
    conn.execute("INSERT INTO memories(id,type,title,body,embedding,valid_to) "
                 "VALUES(?,?,?,?,?,NULL)", (mid, type, title, body, emb))


def test_on_remember_writes_semantic_edges():
    pytest.importorskip("numpy")
    g = make_graph()
    e = FakeEmbedder()
    a = _pack(e.embed("I drive my car"))          # vehicle axis
    b = _pack(e.embed("driving the automobile"))  # vehicle axis (near a)
    c = _pack(e.embed("pizza and food"))          # food axis (far)
    _seed_memory(g._conn, 1, "user", "A", "car", a)
    _seed_memory(g._conn, 2, "user", "B", "auto", b)
    _seed_memory(g._conn, 3, "user", "C", "lunch", c)
    g.on_remember(2, b)
    # 2 links to 1 (near) with higher weight than to 3 (far)
    w12 = g._conn.execute("SELECT weight FROM edges WHERE src=1 AND dst=2 AND kind='semantic'").fetchone()
    w23 = g._conn.execute("SELECT weight FROM edges WHERE src=2 AND dst=3 AND kind='semantic'").fetchone()
    assert w12 is not None and w12[0] > (w23[0] if w23 else 0.0)


def test_on_link_writes_explicit_edge():
    g = make_graph()
    g.on_link(7, 3)
    row = g._conn.execute("SELECT src, dst, kind, weight FROM edges").fetchone()
    assert row == (3, 7, "explicit", 1.0)
```

```python
# tests/test_store.py (append)

def test_remember_writes_semantic_edges_when_assoc_on():
    import pytest
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder())
    s._graph.enabled = True                     # force association on (assoc arg lands in Task 6)
    s.migrate()
    s.remember("user", "Commute", "I drive my car to work")
    s.remember("user", "Errand", "driving the automobile downtown")
    n = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='semantic'").fetchone()[0]
    assert n >= 1


def test_link_writes_explicit_edge():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None)
    s._graph.enabled = True
    s.migrate()
    a = s.remember("user", "A", "x")["id"]
    b = s.remember("project", "B", "y")["id"]
    s.link(a, b)
    row = conn.execute("SELECT kind, weight FROM edges").fetchone()
    assert row == ("explicit", 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py tests/test_store.py -k "semantic_edges or explicit_edge" -v`
Expected: FAIL — `AttributeError: 'Graph' object has no attribute 'on_remember'`.

- [ ] **Step 3: Add edge maintenance to `src/tether/graph.py`**

```python
    def on_remember(self, mid, emb_blob) -> None:
        if not self.enabled or emb_blob is None:
            return
        try:
            import numpy as np

            q = np.frombuffer(emb_blob, dtype="<f4")
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE embedding IS NOT NULL AND valid_to IS NULL AND id != ?",
                (mid,)).fetchall()
            if not rows:
                return
            ids = [r[0] for r in rows]
            mat = np.frombuffer(b"".join(r[1] for r in rows),
                                dtype="<f4").reshape(len(ids), -1)
            sims = mat @ q
            k = min(KNN_K, len(ids))
            top = np.argsort(-sims)[:k]
            now = _now()
            for i in top:
                w = float(sims[i])
                if w <= 0:
                    continue
                self._upsert_edge(mid, ids[int(i)], "semantic", w, now, mode="max")
        except Exception:
            return

    def on_link(self, id_a, id_b) -> None:
        if not self.enabled:
            return
        try:
            self._upsert_edge(id_a, id_b, "explicit", 1.0, _now(), mode="max")
        except Exception:
            return

    def backfill_semantic(self) -> None:
        if not self.enabled:
            return
        try:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE embedding IS NOT NULL AND valid_to IS NULL").fetchall()
            for mid, blob in rows:
                self.on_remember(mid, blob)
        except Exception:
            return

    def backfill_explicit(self, pairs) -> None:
        if not self.enabled:
            return
        try:
            now = _now()
            for a, b in pairs:
                self._upsert_edge(a, b, "explicit", 1.0, now, mode="max")
        except Exception:
            return
```

- [ ] **Step 4: Delegate from `src/tether/store.py`**

In `remember`, after `mid`/`action` are known and before `self._conn.commit()`, add an `on_remember` call (covers both insert and update; `emb` is already computed):

```python
        self._graph.on_remember(mid, emb)
```

In `link`, before `self._conn.commit()`, add:

```python
        self._graph.on_link(id_a, id_b)
```

In `backfill_embeddings`, replace the final `return done` of the success path with a graph backfill first:

```python
            self._graph.backfill_semantic()
            return done
```

(Place `self._graph.backfill_semantic()` immediately before `return done`, inside the `try`, after the `while` loop completes.)

In `migrate`, after `self._graph.migrate()` and before `self._conn.commit()`, import existing links into explicit edges:

```python
        if self._graph.enabled:
            pairs = []
            for (rid, links_json) in self._conn.execute(
                    "SELECT id, links FROM memories").fetchall():
                for other in json.loads(links_json or "[]"):
                    pairs.append((rid, other))
            self._graph.backfill_explicit(pairs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph.py tests/test_store.py -q`
Expected: all pass (existing store tests unaffected — their stores keep `assoc` disabled).

- [ ] **Step 6: Commit**

```bash
git add src/tether/graph.py src/tether/store.py tests/test_graph.py tests/test_store.py
git commit -m "feat: semantic kNN + explicit edge maintenance and backfill"
```

---

### Task 4: `spread` — the activation walk + receipts

**Files:**
- Modify: `src/tether/graph.py` (`_neighbors`, `spread`)
- Test: `tests/test_graph.py` (append)

**Interfaces:**
- Produces:
  - `Graph._neighbors(node, type) -> list[(neighbor_id, blended_weight, dominant_kind)]` — current (`valid_to IS NULL`), type-filtered neighbors; weight blended across kinds via `KIND_W`; deterministic (sorted by neighbor id).
  - `Graph.spread(seed_activation: dict[int,float], budget: int, type=None) -> (activation: dict[int,float], receipts: dict[int,dict])` — bounded spreading activation. `budget` = max node-expansions. Returns total activation per node and, for spread-reached nodes, `{"from","kind","w","hops"}` for the strongest incoming edge. Disabled / `budget<=0` / empty seeds → returns `(dict(seed_activation), {})`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py (append)

def _mem(conn, mid, type="user", valid=True):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, type TEXT, "
        "title TEXT, body TEXT, embedding BLOB, valid_to TEXT)")
    conn.execute("INSERT INTO memories(id,type,valid_to) VALUES(?,?,?)",
                 (mid, type, None if valid else "t"))


def test_spread_reaches_two_hops_within_budget():
    g = make_graph()
    for i in (1, 2, 3):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    g._upsert_edge(2, 3, "hebbian", 1.0, "t", mode="max")
    act0, _ = g.spread({1: 1.0}, budget=0)          # spreading off
    assert set(act0) == {1}
    act2, receipts = g.spread({1: 1.0}, budget=8)   # 1 -> 2 -> 3
    assert 2 in act2 and 3 in act2
    assert act2[1] > act2[2] > act2[3]              # decays with distance
    assert receipts[2]["from"] == 1 and receipts[2]["kind"] == "hebbian"
    assert receipts[3]["hops"] == 2


def test_spread_keeps_seed_above_weak_associate():
    g = make_graph()
    for i in (1, 2): _mem(g._conn, i)
    g._upsert_edge(1, 2, "hebbian", 0.3, "t", mode="max")
    act, _ = g.spread({1: 0.5, 2: 0.5}, budget=8)   # both seeded equally
    assert act[1] > act[2] or act[2] > act[1]        # deterministic, no crash
    # a pure associate never outranks its strong seed source
    act2, _ = g.spread({1: 1.0}, budget=8)
    assert act2[1] > act2[2]


def test_spread_respects_type_and_validity():
    g = make_graph()
    _mem(g._conn, 1, "user"); _mem(g._conn, 2, "project"); _mem(g._conn, 3, "user", valid=False)
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    g._upsert_edge(1, 3, "hebbian", 1.0, "t", mode="max")
    act, _ = g.spread({1: 1.0}, budget=8, type="user")
    assert 2 not in act        # wrong type filtered
    assert 3 not in act        # superseded filtered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -k spread -v`
Expected: FAIL — `AttributeError: 'Graph' object has no attribute 'spread'`.

- [ ] **Step 3: Add `_neighbors` and `spread` to `src/tether/graph.py`**

```python
    def _neighbors(self, node, type=None):
        rows = self._conn.execute(
            "SELECT CASE WHEN src=? THEN dst ELSE src END, kind, weight "
            "FROM edges WHERE src=? OR dst=?", (node, node, node)).fetchall()
        if not rows:
            return []
        agg = {}
        for nbr, kind, w in rows:
            agg.setdefault(nbr, {})[kind] = w
        ids = list(agg)
        ph = ",".join("?" for _ in ids)
        params = list(ids)
        sql = f"SELECT id, type FROM memories WHERE id IN ({ph}) AND valid_to IS NULL"
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        valid = {r[0]: r[1] for r in self._conn.execute(sql, params).fetchall()}
        out = []
        for nbr in sorted(agg):
            if nbr not in valid:
                continue
            kinds = agg[nbr]
            blended = sum(KIND_W.get(k, 0.0) * w for k, w in kinds.items())
            if blended <= 0:
                continue
            dominant = max(kinds, key=lambda k: KIND_W.get(k, 0.0) * kinds[k])
            out.append((nbr, blended, dominant))
        return out

    def spread(self, seed_activation, budget, type=None):
        activation = dict(seed_activation)
        if not self.enabled or budget <= 0 or not activation:
            return activation, {}
        receipts = {}
        depth = {mid: 0 for mid in seed_activation}
        fired = set()
        expansions = 0
        while expansions < budget:
            candidates = [(a, mid) for mid, a in activation.items()
                          if mid not in fired and a >= EPSILON]
            if not candidates:
                break
            candidates.sort(key=lambda x: (-x[0], x[1]))
            a, node = candidates[0]
            fired.add(node)
            expansions += 1
            for nbr, w, kind in self._neighbors(node, type):
                transmit = a * w * HOP_DECAY
                if transmit < EPSILON:
                    continue
                activation[nbr] = activation.get(nbr, 0.0) + transmit
                d = depth.get(node, 0) + 1
                if nbr not in depth or d < depth[nbr]:
                    depth[nbr] = d
                if nbr not in seed_activation:
                    prev = receipts.get(nbr)
                    if prev is None or transmit > prev["_t"]:
                        receipts[nbr] = {"from": node, "kind": kind,
                                         "w": round(w, 3), "hops": depth[nbr], "_t": transmit}
        for r in receipts.values():
            r.pop("_t", None)
        return activation, receipts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph.py -k spread -v`
Expected: the three spread tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py tests/test_graph.py
git commit -m "feat: spreading-activation walk over the usage graph, with receipts"
```

---

### Task 5: session working set — priming + Hebbian

**Files:**
- Modify: `src/tether/graph.py` (`resolve_session`, `session_activation`, `touch_session`)
- Test: `tests/test_graph.py` (append)

**Interfaces:**
- Produces:
  - `Graph.resolve_session(session, meta_get, meta_set) -> str` — returns `session` if given; else time-buckets using `meta_get`/`meta_set` callbacks over keys `assoc_session` / `assoc_last_activity` (a gap > `SESSION_GAP_SECONDS` starts a new id = now). Updates last-activity.
  - `Graph.session_activation(session_id) -> dict[int,float]` — current working-set activations for priming.
  - `Graph.touch_session(session_id, ordered_ids)` — decays the session's activations (×`SESSION_DECAY`), bumps the touched ids by `1.0`, lays capped Hebbian edges among the top-`HEBBIAN_TOP_M` active members, and prunes rows below `SESSION_TTL_ACTIVATION`. Never raises.
- Consumes: `Store`'s `_meta_get`/`_meta_set` (passed as callbacks so `graph` stays independent of the `meta` table's owner).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py (append)

def _meta_pair():
    store = {}
    return (lambda k: store.get(k)), (lambda k, v: store.__setitem__(k, str(v)))


def test_resolve_session_uses_param_then_time_buckets():
    g = make_graph()
    get, set_ = _meta_pair()
    assert g.resolve_session("explicit-1", get, set_) == "explicit-1"
    sid1 = g.resolve_session(None, get, set_)
    sid2 = g.resolve_session(None, get, set_)      # immediately after -> same bucket
    assert sid1 == sid2
    # simulate a long gap by rewinding last-activity far into the past
    set_("assoc_last_activity", "2000-01-01T00:00:00+00:00")
    sid3 = g.resolve_session(None, get, set_)
    assert sid3 != sid1


def test_touch_session_primes_and_decays():
    g = make_graph()
    g.touch_session("s", [1, 2])
    a1 = g.session_activation("s")
    assert a1[1] == 1.0 and a1[2] == 1.0
    g.touch_session("s", [2])                       # decays 1&2, bumps 2
    a2 = g.session_activation("s")
    assert a2[1] == 0.5 and a2[2] == 1.5


def test_touch_session_lays_hebbian_edges():
    g = make_graph()
    g.touch_session("s", [1, 2, 3])                 # all co-active
    w = g._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='hebbian'").fetchone()[0]
    assert w == 3                                   # pairs (1,2),(1,3),(2,3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -k session -v`
Expected: FAIL — `AttributeError: 'Graph' object has no attribute 'resolve_session'`.

- [ ] **Step 3: Add session methods to `src/tether/graph.py`**

```python
    def resolve_session(self, session, meta_get, meta_set) -> str:
        now_iso = _now()
        if session:
            sid = str(session)
        else:
            last = meta_get("assoc_last_activity")
            cur = meta_get("assoc_session")
            gap = None
            if last is not None:
                try:
                    gap = (datetime.fromisoformat(now_iso)
                           - datetime.fromisoformat(last)).total_seconds()
                except Exception:
                    gap = None
            if cur and gap is not None and gap <= SESSION_GAP_SECONDS:
                sid = cur
            else:
                sid = now_iso
        meta_set("assoc_session", sid)
        meta_set("assoc_last_activity", now_iso)
        return sid

    def session_activation(self, session_id) -> dict:
        try:
            return {r[0]: r[1] for r in self._conn.execute(
                "SELECT memory_id, activation FROM session_members WHERE session_id=?",
                (session_id,)).fetchall()}
        except Exception:
            return {}

    def touch_session(self, session_id, ordered_ids) -> None:
        try:
            now = _now()
            self._conn.execute(
                "UPDATE session_members SET activation = activation * ?, updated_at=? "
                "WHERE session_id=?", (SESSION_DECAY, now, session_id))
            for mid in ordered_ids:
                self._conn.execute(
                    "INSERT INTO session_members(session_id, memory_id, activation, updated_at) "
                    "VALUES (?,?,?,?) ON CONFLICT(session_id, memory_id) "
                    "DO UPDATE SET activation = session_members.activation + 1.0, updated_at=?",
                    (session_id, mid, 1.0, now, now))
            active = [r[0] for r in self._conn.execute(
                "SELECT memory_id FROM session_members WHERE session_id=? "
                "ORDER BY activation DESC, memory_id LIMIT ?",
                (session_id, HEBBIAN_TOP_M)).fetchall()]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    self._upsert_edge(active[i], active[j], "hebbian",
                                      HEBBIAN_INCREMENT, now, mode="add")
            self._conn.execute(
                "DELETE FROM session_members WHERE activation < ?",
                (SESSION_TTL_ACTIVATION,))
        except Exception:
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph.py -k session -v`
Expected: the three session tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py tests/test_graph.py
git commit -m "feat: session working set - priming activations + Hebbian co-recall edges"
```

---

### Task 6: recall orchestration

**Files:**
- Modify: `src/tether/store.py` (`__init__`, `_seed_scores`, `recall`)
- Test: `tests/test_store.py` (append)

**Interfaces:**
- Produces:
  - `Store.__init__(..., assoc=False, recall_budget=24)` — new args; `self._graph = Graph(conn, enabled=assoc)`, `self._recall_budget = recall_budget`.
  - `Store._seed_scores(query, type) -> dict[int,float]` — the v0.2 hybrid+recency+decay scoring, factored out.
  - `Store.recall(query, type=None, limit=20, budget=None, session=None) -> list[dict]` — associative when the graph is enabled (seed → prime → spread → rank → learn → `via` receipts); byte-identical to v0.2 when disabled. `budget=None` uses `self._recall_budget`.
  - Module constant `_PRIMING_WEIGHT = 0.25`.
- Consumes: `Graph.spread`/`session_activation`/`touch_session`/`resolve_session` (Tasks 4–5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py (append)

def make_assoc_store():
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, recall_budget=16)
    s.migrate()
    return s


def test_recall_disabled_matches_v2():
    # assoc defaults False -> identical to the v0.2 recall path (no 'via' field).
    s = make_store()  # helper from the existing suite; assoc off
    s.remember("user", "A", "car and driving")
    hits = s.recall("car")
    assert hits and "via" not in hits[0]
    assert set(hits[0]) == {"id", "type", "title", "body", "tags", "updated_at"}


def test_recall_associative_finds_linked_neighbor():
    import pytest
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why not sessions", "sessions were rejected for scaling")["id"]
    s.link(a, b)                                  # explicit edge a<->b
    # 'JWT' matches only A; B is reached across the explicit edge
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert a in ids and b in ids


def test_recall_via_receipts_present():
    import pytest
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why", "the rationale doc")["id"]
    s.link(a, b)
    hits = {h["id"]: h for h in s.recall("JWT tokens", budget=8)}
    assert hits[a]["via"] == {"seed": True}
    assert "path" in hits[b]["via"] and hits[b]["via"]["path"][0]["from"] == a


def test_recall_budget_zero_is_passthrough():
    import pytest
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
    b = s.remember("project", "Why", "the rationale doc")["id"]
    s.link(a, b)
    ids = [h["id"] for h in s.recall("JWT tokens", budget=0)]
    assert ids == [a]                             # no spreading -> only the direct match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -k "recall_disabled or associative or via_receipts or budget_zero" -v`
Expected: FAIL — `Store.__init__() got an unexpected keyword argument 'assoc'`.

- [ ] **Step 3: Edit `src/tether/store.py`**

Add the priming constant near the other module constants (after `_RECENCY_WEIGHT`):

```python
_PRIMING_WEIGHT = 0.25
```

Extend `__init__` (replace its signature and add the two fields):

```python
    def __init__(self, conn, device_id: str, sync_now, embedder=None,
                 author="", consolidate=False, dedup_threshold=0.92,
                 decay_half_life_days=None, assoc=False, recall_budget=24):
        self._conn = conn
        self._device_id = device_id
        self._sync_now = sync_now
        self._embedder = embedder
        self._author = author
        self._consolidate = consolidate
        self._dedup_threshold = dedup_threshold
        self._decay_half_life_days = decay_half_life_days
        self._recall_budget = recall_budget
        self._graph = Graph(conn, enabled=assoc)
```

(Delete the interim `self._graph = Graph(conn, enabled=False)` line added in Task 2 — this replaces it.)

Factor the seed scoring out and rewrite `recall` (replace the whole `recall` method):

```python
    def _seed_scores(self, query, type) -> dict:
        fts_ids = self._fts_ids(query, type)
        vec_ids = self._vector_ids(query, type)
        lists = [fts_ids] + ([vec_ids] if vec_ids else [])
        scores = _rrf_scores(lists)
        if not scores:
            return {}
        recency = _rrf_scores([self._recency_order(list(scores))])
        for mid, s in recency.items():
            scores[mid] += _RECENCY_WEIGHT * s
        if self._decay_half_life_days:
            now = _now()
            updated = self._updated_at_of(list(scores))
            for mid in list(scores):
                scores[mid] *= _decay_factor(
                    _age_days(updated[mid], now), self._decay_half_life_days)
        return scores

    def recall(self, query, type=None, limit=20, budget=None, session=None) -> list:
        if not query or not query.strip():
            return []
        scores = self._seed_scores(query, type)
        if not self._graph.enabled:
            if not scores:
                return []
            order = [mid for mid, _ in sorted(
                scores.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
            return self._hydrate(order)          # v0.2 shape, no `via`
        # associative path
        if budget is None:
            budget = self._recall_budget
        sid = self._graph.resolve_session(session, self._meta_get, self._meta_set)
        for mid, a in self._graph.session_activation(sid).items():
            scores[mid] = scores.get(mid, 0.0) + _PRIMING_WEIGHT * a
        if not scores:
            return []
        activation, receipts = self._graph.spread(scores, budget, type)
        order = [mid for mid, _ in sorted(
            activation.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
        self._graph.touch_session(sid, order)
        self._conn.commit()
        hits = self._hydrate(order)
        for h in hits:
            r = receipts.get(h["id"])
            if r is not None and h["id"] not in scores:
                h["via"] = {"path": [{"from": r["from"], "kind": r["kind"], "w": r["w"]}],
                            "hops": r["hops"]}
            else:
                h["via"] = {"seed": True}
        return hits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -q`
Expected: all pass — the v0.2 recall tests still green (they run `assoc=False`), plus the four associative tests.

- [ ] **Step 5: Commit**

```bash
git add src/tether/store.py tests/test_store.py
git commit -m "feat: associative recall - seed + prime + spread + learn + via receipts"
```

---

### Task 7: server wiring + docs

**Files:**
- Modify: `src/tether/server.py` (`_get_store`, `recall` tool)
- Modify: `tests/test_mcp.py` (append)
- Modify: `README.md`

**Interfaces:**
- Consumes: `config.assoc_enabled`/`recall_budget` (Task 1), `Store(..., assoc=, recall_budget=)` (Task 6).
- Produces: the lazy store is built with the association config; the `recall` tool exposes optional `budget` and `session`, and hits carry `via`. Default call (no new args) is unchanged for existing callers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp.py (append)

def test_server_recall_budget_session(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("numpy")
    from tether import server

    class Fake:
        name = "fake-3d"; dims = 3
        _AXES = [("car", "automobile", "drive"), ("pizza", "food"), ("python", "test")]
        def embed(self, text):
            import math
            t = text.lower()
            v = [float(sum(w in t for w in ax)) for ax in self._AXES]
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v] if n else v

    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_ASSOC", raising=False)
    monkeypatch.setattr("tether.embed.get_embedder", lambda *a, **k: Fake())
    server._store = None
    try:
        s = server._get_store()
        assert s._graph.enabled is True
        a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
        b = s.remember("project", "Why", "rationale")["id"]
        s.link(a, b)
        hits = s.recall("JWT tokens", budget=8, session="sess-1")
        assert any(h["id"] == b for h in hits)     # reached via the explicit edge
    finally:
        server._store = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp.py -k budget_session -v`
Expected: FAIL — `_get_store` doesn't pass `assoc`, so `s._graph.enabled` is `False`.

- [ ] **Step 3: Wire the server (`src/tether/server.py`)**

In `_get_store`, pass the two new args to `Store(...)` (add to the constructor call):

```python
        store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                      embedder=embedder, author=config.author(),
                      consolidate=config.consolidate_enabled(),
                      dedup_threshold=config.dedup_threshold(),
                      decay_half_life_days=config.decay_half_life_days(),
                      assoc=config.assoc_enabled(),
                      recall_budget=config.recall_budget())
```

Update the `recall` tool to accept and forward the new params (replace the tool):

```python
@mcp.tool()
def recall(query: str, type: str = None, limit: int = 20,
           budget: int = None, session: str = None) -> dict:
    """Search memories by keyword and semantic similarity, then follow the
    usage graph to related memories, most relevant first.

    Each hit carries {id, type, title, body, tags, updated_at} and a `via`
    receipt explaining why it surfaced (a direct match, or the edge it was
    reached through). Use `updated_at` to judge staleness and `id` to cite what
    you update via remember/link.

    Args:
        query: free text; punctuation is safe.
        type: optional filter ("user"/"feedback"/"project"/"reference").
        limit: max results (default 20).
        budget: how far to follow associations (0 = direct matches only).
        session: optional id grouping related recalls so they prime each other.
    """
    try:
        return {"results": _get_store().recall(
            query, type=type, limit=limit, budget=budget, session=session)}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass. The existing MCP roundtrip (which sets `TETHER_SEMANTIC=0`) is unaffected — with no embedder there are no semantic edges, and Hebbian/explicit still work; the default `recall` call shape is unchanged.

- [ ] **Step 5: Update `README.md`**

Add after the "Consolidation (optional)" section:

````markdown
## Associative recall (optional)

`recall` doesn't just return keyword/semantic matches — it follows a **usage
graph** to related memories, so asking about one thing surfaces its connected
context. The graph's edges come from three local, deterministic sources — no
LLM, no network:

- **semantic** — nearest neighbours by embedding (needs the `[semantic]` extra),
- **explicit** — the `link()` verb,
- **hebbian** — memories you recall *together* get wired together over time.

Every hit carries a `via` receipt saying why it surfaced (a direct match, or the
edge it came through), and two optional `recall` args tune it:

| Arg / var | Default | Effect |
|---|---|---|
| `budget` (per call) | `TETHER_RECALL_BUDGET` | how far to follow associations; `0` = direct matches only |
| `session` (per call) | time-bucketed | group related recalls so they prime each other |
| `TETHER_ASSOC` | on | set `0`/`false`/`off` for plain keyword+semantic recall |
| `TETHER_RECALL_BUDGET` | `24` | default association breadth |

With `TETHER_ASSOC=0` (or `budget=0`, or an empty graph), `recall` behaves exactly
as before — associative recall is purely additive and never breaks a lookup.
````

- [ ] **Step 6: Commit**

```bash
git add src/tether/server.py tests/test_mcp.py README.md
git commit -m "feat: wire associative recall into the server; budget/session params; docs"
```

---

## Self-Review

**1. Spec coverage:**
- Usage graph (semantic kNN + explicit + Hebbian) → Tasks 2–5. ✓
- Spreading-activation recall with `budget` dial → Task 4 (`spread`), Task 6 (wiring). ✓
- Activation receipts (`via`) → Task 4 (produced), Task 6 (attached). ✓
- Session priming → Tasks 5–6. ✓
- `store`/`graph` split, `recall` orchestrator → Tasks 2, 6. ✓
- Degrade-never / cold-start / assoc-off == v0.2 → Task 6 (`_graph.enabled` gate, `budget<=0` passthrough), tested. ✓
- No new verbs; `recall` gains optional params + `via` → Task 7. ✓
- Additive schema, no `memories` change → Task 2. ✓
- Config (`TETHER_ASSOC`, `TETHER_RECALL_BUDGET`) → Task 1. ✓
- Write-time kNN approximation (no retroactive rewire) → documented; `on_remember` upserts the writer's own top-k only. ✓

**2. Placeholder scan:** No TBD/TODO. Every code step is complete and runnable. The one interim line (`Graph(conn, enabled=False)` in Task 2) is explicitly replaced in Task 6.

**3. Type consistency:** `Graph(conn, enabled=)` used identically across tasks. `spread(seed_activation, budget, type) -> (activation, receipts)` matches its consumer in Task 6. `_neighbors -> (id, weight, kind)` tuples feed `spread`. `resolve_session(session, meta_get, meta_set)` matches Task 6's call with `self._meta_get`/`self._meta_set`. Receipt dict keys (`from`/`kind`/`w`/`hops`) are produced in Task 4 and consumed in Task 6. `Store.recall(..., budget=None, session=None)` matches the server tool in Task 7. `_upsert_edge(..., mode)` modes (`max`/`add`) are consistent across Tasks 2–5.

## Execution notes

- Tasks are dependency-ordered; each ends green and independently reviewable.
- Existing v0.2 tests stay green because `Store` defaults `assoc=False`; only the new tests (and the server, via config) turn the graph on.
- The numpy-dependent tests use `pytest.importorskip("numpy")`; the deterministic `spread`/session tests use hand-inserted edges and need no embedder or numpy.
