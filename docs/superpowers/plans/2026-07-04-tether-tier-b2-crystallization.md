# Tier B2 — Crystallization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let tether detect dense clusters of related memories, surface them to the calling agent as candidate "principles," and — when the agent names one — write the principle and auto-link it over its sources.

**Architecture:** A new `crystallize.py` computes candidate clusters as a *read-time derived view* (seed-from-peak over explicit/hebbian edges, membership expanded along semantic edges) — no daemon, no maintained table. `remember(crystallizes=[…])` writes the principle + a new directional `crystallized` edge kind. A pull-only MCP resource surfaces candidates; a `dismiss_cluster` tool records rejected nuclei. Everything is opt-in and degrade-never.

**Tech Stack:** Python 3.10+, SQLite (`edges` table), numpy (optional, guarded), Model2Vec embeddings (optional), FastMCP.

## Global Constraints

- 100% local, deterministic, **no LLM inside tether**, no network. Naming is done by the *calling* agent.
- **Degrade-never:** every operation no-ops rather than raises; feature is **off by default** (`TETHER_CRYSTALLIZE`, like `TETHER_CONSOLIDATE`/`TETHER_FORGET`).
- **`crystallized` edge-kind role matrix:** included in recall spreading (`KIND_W`) and in `degree_map` default kinds (hubs); **excluded from peak-seeding** (prevents principles-of-principles recursion).
- **Crystallized edges are stored directionally** (`src`=principle, `dst`=source) — the one kind not canonicalized — so `principle.sources` is recoverable for dedup. All read paths (`spread`, `degree_map`, `_neighbors`) already read both endpoints, so bidirectional activation is unaffected.
- **Tuning constants are module-level** (matching `graph.py`'s `KNN_K`/`HEBBIAN_*` precedent). Only the master enable is an env var. `W_b` (behavioral peak weight) ships at **0**.
- **No new memory verb.** `remember(crystallizes=)` extends the existing verb (typed links). `dismiss_cluster` is a reflection-loop control, not a memory operation.
- **Dedup by basis-recovery overlap:** suppress when `|candidate.members ∩ principle.sources| / |principle.sources| ≥ DEDUP_OVERLAP`.
- **Dismissed-set keyed on the peak edge** (stable id), not the member set.
- Hermetic tests: `FakeEmbedder` + hand-inserted edges; numpy guarded via `pytest.importorskip`.
- The bench no-regression guard must include a **crystallized-hub control case** (#25 back-door).

---

### Task 1: `crystallized` edge kind + config opt-in

**Files:**
- Modify: `src/tether/graph.py` (add kind to `KIND_W`, `degree_map` default, add `on_crystallize`)
- Modify: `src/tether/config.py` (add `crystallize_enabled`)
- Test: `tests/test_graph.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `Graph._upsert_edge`, `Graph.spread`, `Graph.degree_map`, `Graph._neighbors`, `KIND_W`.
- Produces: `KIND_W["crystallized"] = 1.2`; `degree_map()` default kinds include `"crystallized"`; `Graph.on_crystallize(principle_id, source_ids)` inserts **directional** `crystallized` edges (principle→source), no-op when `not self.enabled`; `config.crystallize_enabled() -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph.py — append
import sqlite3
from tether.graph import Graph, KIND_W

def _graph():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories(id INTEGER PRIMARY KEY, valid_to TEXT, type TEXT);")
    g = Graph(conn, enabled=True)
    g.migrate()
    return conn, g

def test_crystallized_kind_registered_for_spreading():
    assert "crystallized" in KIND_W and KIND_W["crystallized"] > 0

def test_on_crystallize_writes_directional_edges():
    conn, g = _graph()
    for i in (1, 2, 3):
        conn.execute("INSERT INTO memories(id, type) VALUES(?, 'project')", (i,))
    g.on_crystallize(1, [2, 3])                     # principle=1, sources 2,3
    rows = conn.execute(
        "SELECT src, dst, kind FROM edges WHERE kind='crystallized' "
        "ORDER BY dst").fetchall()
    assert rows == [(1, 2, "crystallized"), (1, 3, "crystallized")]  # NOT canonicalized

def test_crystallized_counts_toward_hub_degree():
    conn, g = _graph()
    for i in (1, 2, 3):
        conn.execute("INSERT INTO memories(id, type) VALUES(?, 'project')", (i,))
    g.on_crystallize(1, [2, 3])
    deg = g.degree_map()                            # default kinds
    assert deg[1] > 0 and deg[2] > 0                # principle and sources are hubs

def test_on_crystallize_disabled_is_noop():
    conn, g = _graph()
    g.enabled = False
    conn.execute("INSERT INTO memories(id, type) VALUES(1, 'project')")
    g.on_crystallize(1, [2])
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
```

```python
# tests/test_config.py — append
def test_crystallize_off_by_default(monkeypatch):
    monkeypatch.delenv("TETHER_CRYSTALLIZE", raising=False)
    assert config.crystallize_enabled() is False
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("TETHER_CRYSTALLIZE", v)
        assert config.crystallize_enabled() is True
    monkeypatch.setenv("TETHER_CRYSTALLIZE", "0")
    assert config.crystallize_enabled() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_graph.py -k crystalliz tests/test_config.py -k crystallize -q`
Expected: FAIL (`KeyError: 'crystallized'`, `AttributeError: on_crystallize`, `AttributeError: crystallize_enabled`).

- [ ] **Step 3: Implement the graph changes**

```python
# src/tether/graph.py — change KIND_W (line 17)
KIND_W = {"semantic": 1.0, "explicit": 1.2, "hebbian": 1.0, "crystallized": 1.2}
```

```python
# src/tether/graph.py — change degree_map default (line 140)
    def degree_map(self, kinds=("explicit", "hebbian", "crystallized")) -> dict:
```

```python
# src/tether/graph.py — add method after on_link
    def on_crystallize(self, principle_id, source_ids) -> None:
        """Directional crystallized edges principle->source (the one kind NOT
        canonicalized, so provenance is recoverable for dedup). Read paths
        (_neighbors/spread/degree_map) read both endpoints, so activation stays
        bidirectional. No-op when disabled; never raises."""
        if not self.enabled:
            return
        try:
            now = _now()
            for src in source_ids:
                if src == principle_id:
                    continue
                self._conn.execute(
                    "INSERT INTO edges(src, dst, kind, weight, updated_at) "
                    "VALUES (?,?,'crystallized',1.0,?) "
                    "ON CONFLICT(src, dst, kind) DO UPDATE SET updated_at=excluded.updated_at",
                    (principle_id, src, now))
        except Exception:
            return
```

```python
# src/tether/config.py — add after crystallize constants near _ASSOC_OFF block
_CRYSTALLIZE_ON = {"1", "true", "yes", "on"}


def crystallize_enabled() -> bool:
    """Crystallization (agent-in-the-loop principle detection) is opt-in, off by
    default."""
    val = os.environ.get("TETHER_CRYSTALLIZE")
    if val is None:
        return False
    return val.strip().lower() in _CRYSTALLIZE_ON
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_graph.py -k crystalliz tests/test_config.py -k crystallize -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py src/tether/config.py tests/test_graph.py tests/test_config.py
git commit -m "feat(b2): crystallized edge kind (spreading+hubs, directional) + opt-in config"
```

---

### Task 2: `remember(crystallizes=…)` write-back

**Files:**
- Modify: `src/tether/store.py` (`Store.__init__` add `crystallize` flag; `remember` add `crystallizes` param)
- Modify: `src/tether/server.py` (`_get_store` wire `crystallize=config.crystallize_enabled()`; `remember` tool add `crystallizes` param)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Graph.on_crystallize` (Task 1), `config.crystallize_enabled` (Task 1).
- Produces: `Store(..., crystallize=False)`; `Store.remember(type, title, body, tags=None, links=None, crystallizes=None) -> {"id","action"}` — when `crystallize` and graph enabled and `crystallizes` given, links the new/updated memory to each source id via `crystallized` edges.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py — append
def test_remember_crystallizes_links_sources_when_enabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    a = s.remember("project", "auth outage", "login 500s under load")["id"]
    b = s.remember("project", "pool fix", "raised the connection pool ceiling")["id"]
    p = s.remember("reference", "principle: fail fast on saturation",
                   "cap the pool and time out", crystallizes=[a, b])["id"]
    rows = conn.execute(
        "SELECT src, dst FROM edges WHERE kind='crystallized' ORDER BY dst").fetchall()
    assert rows == [(p, a), (p, b)]

def test_remember_crystallizes_ignored_when_disabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=False)          # feature off
    s.migrate()
    a = s.remember("project", "x", "y")["id"]
    s.remember("reference", "p", "z", crystallizes=[a])
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='crystallized'").fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_store.py -k crystalliz -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'crystallize'`).

- [ ] **Step 3: Implement**

```python
# src/tether/store.py — Store.__init__ signature: add crystallize=False after protect_head/seed_floor
                 protect_head=_PROTECT_HEAD, seed_floor=_SEED_FLOOR,
                 crystallize=False,
                 boot_index_cap=50, forget=False, forget_age_days=90,
```

```python
# src/tether/store.py — in __init__ body, after self._seed_floor = seed_floor
        self._crystallize = crystallize
```

```python
# src/tether/store.py — remember signature + body (line 286)
    def remember(self, type, title, body, tags=None, links=None,
                 crystallizes=None) -> dict:
```
Then, in `remember`, replace the final block:
```python
        self._graph.on_remember(mid, emb)
        if self._crystallize and crystallizes:
            self._graph.on_crystallize(mid, crystallizes)
        self._conn.commit()
        self._sync_now()
        self._maybe_forget()
        return {"id": mid, "action": action}
```

```python
# src/tether/server.py — _get_store, add to Store(...) kwargs
                      seed_floor=config.seed_floor(),
                      crystallize=config.crystallize_enabled(),
                      boot_index_cap=config.boot_index_cap(),
```

```python
# src/tether/server.py — remember tool signature + call
@mcp.tool()
def remember(type: str, title: str, body: str,
             tags: str = "", links: list = None,
             crystallizes: list = None) -> dict:
    """... (existing docstring) ...
        crystallizes: optional list of source memory ids this memory abstracts;
            links it over them as a crystallized principle (needs TETHER_CRYSTALLIZE).
    """
    try:
        return _get_store().remember(type, title, body, tags=tags, links=links,
                                     crystallizes=crystallizes)
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_store.py -k crystalliz -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tether/store.py src/tether/server.py tests/test_store.py
git commit -m "feat(b2): remember(crystallizes=) writes principle->source edges"
```

---

### Task 3: Detection core — seed-from-peak + semantic-expand

**Files:**
- Create: `src/tether/crystallize.py`
- Test: `tests/test_crystallize.py`

**Interfaces:**
- Consumes: the `edges` table (`explicit`/`hebbian`/`semantic`/`crystallized` kinds), `memories(id, valid_to, title, tags)`.
- Produces: `crystallize.candidates(conn, embedder=None) -> list[dict]`, each `{"peak_key": (a,b), "member_ids": [..sorted..], "member_titles": [..], "why": [(a,b,kind),..], "descriptor": str}`. This task ignores dedup/dismissed/fallback (added in Tasks 4–5); it returns raw candidates of size ≥ `MIN_CLUSTER`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crystallize.py — new file
import sqlite3
from tether import crystallize


def _store_with_edges(memories, edges):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE memories(id INTEGER PRIMARY KEY, valid_to TEXT, "
        "title TEXT, tags TEXT DEFAULT '');"
        "CREATE TABLE edges(src INTEGER, dst INTEGER, kind TEXT, weight REAL, "
        "updated_at TEXT, PRIMARY KEY(src,dst,kind));")
    for mid, title in memories:
        conn.execute("INSERT INTO memories(id, title) VALUES(?,?)", (mid, title))
    for src, dst, kind, w in edges:
        conn.execute("INSERT INTO edges VALUES(?,?,?,?, 't')", (src, dst, kind, w))
    return conn


def test_explicit_peak_seeds_and_semantic_expands():
    # explicit peak (1,2); semantic pulls 3 into the cluster; 4 is far (cos 0.1).
    conn = _store_with_edges(
        [(1, "auth bug"), (2, "pool fix"), (3, "rollback rule"), (4, "unrelated")],
        [(1, 2, "explicit", 1.0),
         (2, 3, "semantic", 0.8),      # >= EXPAND_COS -> member
         (3, 4, "semantic", 0.1)])     # < EXPAND_COS -> excluded
    cands = crystallize.candidates(conn)
    assert len(cands) == 1
    assert cands[0]["member_ids"] == [1, 2, 3]
    assert cands[0]["peak_key"] == (1, 2)


def test_crystallized_edges_do_not_seed_peaks():
    # a crystallized hub must NOT re-seed the cluster it named (no recursion).
    conn = _store_with_edges(
        [(1, "principle"), (2, "src a"), (3, "src b")],
        [(1, 2, "crystallized", 1.0), (1, 3, "crystallized", 1.0)])
    assert crystallize.candidates(conn) == []


def test_semantic_only_does_not_seed():
    # no explicit/hebbian peak -> no candidate (uniform semantic floor).
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c")],
        [(1, 2, "semantic", 0.9), (2, 3, "semantic", 0.9)])
    assert crystallize.candidates(conn) == []


def test_min_cluster_filters_small():
    # explicit peak of 2 with no semantic expansion -> below MIN_CLUSTER (3).
    conn = _store_with_edges(
        [(1, "a"), (2, "b")], [(1, 2, "explicit", 1.0)])
    assert crystallize.candidates(conn) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'tether.crystallize'`).

- [ ] **Step 3: Implement**

```python
# src/tether/crystallize.py — new file
"""crystallize.py - Tier B2 candidate detection (read-time derived view).

Seed-from-peak (explicit + hebbian density) + semantic-expand (membership),
NOT global community detection. Deterministic, local, degrade-never: any
failure returns [] rather than raising. No LLM; the calling agent names the
clusters. See the B2 design doc.
"""

# Tuning constants (module-level, like graph.py's KNN_K/HEBBIAN_*; eval-tuned).
PEAK_KINDS = ("explicit", "hebbian")      # crystallized & semantic excluded from seeding
PEAK_W = {"explicit": 1.0, "hebbian": 0.0}  # W_b = 0 at launch (behavioral parked)
PEAK_FLOOR = 0.5                          # boundness a peak edge must clear to seed
EXPAND_COS = 0.5                          # semantic edge weight (== cosine) to admit a member
EXPAND_HOPS = 1                           # membership hop cap (precision boundary)
MIN_CLUSTER = 3                           # min members to surface


def _current_ids(conn):
    return {r[0] for r in conn.execute(
        "SELECT id FROM memories WHERE valid_to IS NULL")}


def _peak_edges(conn, current):
    ph = ",".join("?" for _ in PEAK_KINDS)
    rows = conn.execute(
        f"SELECT src, dst, kind, weight FROM edges WHERE kind IN ({ph})",
        PEAK_KINDS).fetchall()
    out = []
    for src, dst, kind, w in rows:
        if src in current and dst in current:
            if PEAK_W.get(kind, 0.0) * w >= PEAK_FLOOR:
                a, b = (src, dst) if src < dst else (dst, src)
                out.append((a, b))
    return sorted(set(out))


def _components(peaks):
    """Union-find over peak edges. Deterministic: smaller id is always the root.
    Returns (root_of, members_by_root)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in peaks:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return find, groups


def _expand(members, conn, current):
    frontier = set(members)
    members = set(members)
    for _ in range(EXPAND_HOPS):
        if not frontier:
            break
        ph = ",".join("?" for _ in frontier)
        rows = conn.execute(
            f"SELECT src, dst FROM edges WHERE kind='semantic' AND weight >= ? "
            f"AND (src IN ({ph}) OR dst IN ({ph}))",
            (EXPAND_COS, *frontier, *frontier)).fetchall()
        new = set()
        for src, dst in rows:
            for a, b in ((src, dst), (dst, src)):
                if a in frontier and b in current and b not in members:
                    new.add(b)
        members |= new
        frontier = new
    return members


def _descriptor(conn, member_ids):
    rows = conn.execute(
        "SELECT title, tags FROM memories WHERE id IN (%s)"
        % ",".join("?" for _ in member_ids), member_ids).fetchall()
    titles = "; ".join(r[0] for r in rows if r[0])
    tags = sorted({t for r in rows for t in (r[1] or "").split(",") if t})
    return titles + (f" [tags: {', '.join(tags)}]" if tags else "")


def candidates(conn, embedder=None):
    """Raw candidate clusters (no dedup/dismissed/fallback yet — Tasks 4-5)."""
    try:
        current = _current_ids(conn)
        if not current:
            return []
        peaks = _peak_edges(conn, current)
        if not peaks:
            return []
        find, groups = _components(peaks)
        peaks_by_root = {}
        for a, b in peaks:
            peaks_by_root.setdefault(find(a), []).append((a, b))
        out = []
        for root, seed in sorted(groups.items()):
            members = _expand(seed, conn, current)
            if len(members) < MIN_CLUSTER:
                continue
            member_ids = sorted(members)
            root_peaks = sorted(peaks_by_root.get(root, []))
            out.append({
                "peak_key": root_peaks[0],
                "member_ids": member_ids,
                "member_titles": [r[0] for r in conn.execute(
                    "SELECT title FROM memories WHERE id IN (%s) ORDER BY id"
                    % ",".join("?" for _ in member_ids), member_ids).fetchall()],
                "why": [(a, b, "peak") for a, b in root_peaks],
                "descriptor": _descriptor(conn, member_ids),
            })
        return out
    except Exception:
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tether/crystallize.py tests/test_crystallize.py
git commit -m "feat(b2): crystallization detection — seed-from-peak + semantic-expand"
```

---

### Task 4: Dedup (basis-recovery) + dismissed-set

**Files:**
- Modify: `src/tether/graph.py` (schema: `crystallize_dismissed` table; `dismiss_peak`, `dismissed_peaks`)
- Modify: `src/tether/crystallize.py` (`candidates` suppresses crystallized-overlap + dismissed peaks)
- Modify: `src/tether/store.py` (`dismiss_cluster`)
- Modify: `src/tether/server.py` (`dismiss_cluster` tool)
- Test: `tests/test_crystallize.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: `crystallize.candidates` (Task 3), directional `crystallized` edges (Task 1).
- Produces: `Graph.dismiss_peak(a, b)` / `Graph.dismissed_peaks() -> set[(a,b)]` (canonical); `crystallize.candidates` now suppresses (a) candidates whose basis-recovery overlap vs any crystallized principle's sources ≥ `DEDUP_OVERLAP`, (b) candidates whose `peak_key` is dismissed; `Store.dismiss_cluster(a, b) -> {"dismissed":[a,b]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crystallize.py — append
def test_dedup_suppresses_recovered_principle_basis():
    # principle 10 crystallizes sources {1,2,3}; a candidate re-covering >=60%
    # of that basis is suppressed.
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c"), (10, "principle")],
        [(1, 2, "explicit", 1.0), (2, 3, "semantic", 0.9),
         (10, 1, "crystallized", 1.0), (10, 2, "crystallized", 1.0),
         (10, 3, "crystallized", 1.0)])
    assert crystallize.candidates(conn) == []       # basis {1,2,3} fully re-covered

def test_dismissed_peak_suppresses_candidate():
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c")],
        [(1, 2, "explicit", 1.0), (2, 3, "semantic", 0.9)])
    assert len(crystallize.candidates(conn)) == 1   # visible first
    conn.execute("CREATE TABLE crystallize_dismissed(src INTEGER, dst INTEGER, "
                 "PRIMARY KEY(src,dst));")
    conn.execute("INSERT INTO crystallize_dismissed VALUES(1,2)")  # dismiss the peak
    assert crystallize.candidates(conn) == []
```

```python
# tests/test_graph.py — append
def test_dismiss_peak_is_canonical_and_readable():
    conn, g = _graph()
    g.migrate()
    g.dismiss_peak(5, 2)                             # order-independent
    assert g.dismissed_peaks() == {(2, 5)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py tests/test_graph.py -k "dedup or dismiss" -q`
Expected: FAIL (`no such table: crystallize_dismissed`; `AttributeError: dismiss_peak`).

- [ ] **Step 3: Implement**

```python
# src/tether/graph.py — add table to _SCHEMA (inside the triple-quoted string)
CREATE TABLE IF NOT EXISTS crystallize_dismissed (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    PRIMARY KEY (src, dst)
);
```

```python
# src/tether/graph.py — add methods
    def dismiss_peak(self, a, b) -> None:
        try:
            src, dst = (a, b) if a < b else (b, a)
            self._conn.execute(
                "INSERT OR IGNORE INTO crystallize_dismissed(src, dst) VALUES(?,?)",
                (src, dst))
            self._conn.commit()
        except Exception:
            return

    def dismissed_peaks(self) -> set:
        try:
            return {(r[0], r[1]) for r in self._conn.execute(
                "SELECT src, dst FROM crystallize_dismissed").fetchall()}
        except Exception:
            return set()
```

```python
# src/tether/crystallize.py — add constant
DEDUP_OVERLAP = 0.6                        # basis-recovery fraction to suppress
```

```python
# src/tether/crystallize.py — add helpers
def _principle_bases(conn, current):
    """{principle_id: frozenset(source_ids)} from directional crystallized edges,
    restricted to still-current memories. on_forget deletes a node's edges, but a
    soft-forgotten (valid_to set, edges retained) principle or source must not
    suppress a live candidate via stale basis-recovery overlap."""
    bases = {}
    for src, dst in conn.execute(
            "SELECT src, dst FROM edges WHERE kind='crystallized'").fetchall():
        if src in current and dst in current:
            bases.setdefault(src, set()).add(dst)
    return {p: frozenset(s) for p, s in bases.items()}


def _covered(member_ids, bases):
    members = set(member_ids)
    for sources in bases.values():
        if sources and len(members & sources) / len(sources) >= DEDUP_OVERLAP:
            return True
    return False


def _dismissed(conn):
    try:
        return {(r[0], r[1]) for r in conn.execute(
            "SELECT src, dst FROM crystallize_dismissed").fetchall()}
    except Exception:
        return set()
```

```python
# src/tether/crystallize.py — in candidates(), before appending each cluster:
        bases = _principle_bases(conn, current)
        dismissed = _dismissed(conn)
        out = []
        for root, seed in sorted(groups.items()):
            members = _expand(seed, conn, current)
            if len(members) < MIN_CLUSTER:
                continue
            member_ids = sorted(members)
            root_peaks = sorted(peaks_by_root.get(root, []))
            if root_peaks[0] in dismissed:           # dismissed nucleus
                continue
            if _covered(member_ids, bases):          # basis already crystallized
                continue
            out.append({ ... })                       # (unchanged emit block)
```

```python
# src/tether/store.py — add method near link()
    def dismiss_cluster(self, id_a, id_b) -> dict:
        self._graph.dismiss_peak(id_a, id_b)
        return {"dismissed": [id_a, id_b]}
```

```python
# src/tether/server.py — add tool
@mcp.tool()
def dismiss_cluster(id_a: int, id_b: int) -> dict:
    """Reflection control: dismiss the crystallization candidate nucleated by the
    peak edge (id_a, id_b) so it is not re-surfaced. Not a memory operation."""
    try:
        return _get_store().dismiss_cluster(id_a, id_b)
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py tests/test_graph.py -k "dedup or dismiss" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py src/tether/crystallize.py src/tether/store.py src/tether/server.py tests/test_crystallize.py tests/test_graph.py
git commit -m "feat(b2): basis-recovery dedup + peak-keyed dismissed-set"
```

---

### Task 5: Cold-start semantic-density fallback (gated off)

**Files:**
- Modify: `src/tether/crystallize.py` (relative-outlier semantic seeding, off by default)
- Test: `tests/test_crystallize.py`

**Interfaces:**
- Consumes: `crystallize.candidates` (Tasks 3-4), `semantic` edges (weights == cosine).
- Produces: `crystallize.candidates(conn, embedder=None, fallback=False)` — when `fallback=True`, additionally seeds a cluster from a memory whose local semantic neighborhood is an outlier (`mean + FALLBACK_Z·σ`) against the store's own per-node mean semantic weight, computed **only within materialized kNN neighborhoods** (never all-pairs). Off by default.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crystallize.py — append
def test_fallback_seeds_tight_neighborhood_only_when_enabled():
    # No explicit/hebbian peaks. Node 1 is a genuine semantic-density OUTLIER:
    # strength 1.9 (two 0.95 edges) against a diffuse baseline of eight nodes at
    # 0.4. Its neighbours 2,3 (strength 0.95) are NOT outliers themselves, so
    # only node 1 seeds — then _expand pulls its tight neighbours in.
    # NOTE: the seed must clear mean + FALLBACK_Z*std. A *balanced* bimodal split
    # (equal-size tight vs loose groups) puts the tight nodes at only ~+1 std and
    # can never satisfy Z=2.0 — the tight cluster must be a minority against a
    # larger diffuse background for the threshold to mean anything.
    conn = _store_with_edges(
        [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e"),
         (6, "f"), (7, "g"), (8, "h"), (9, "i"), (10, "j"), (11, "k")],
        [(1, 2, "semantic", 0.95), (1, 3, "semantic", 0.95),
         (4, 5, "semantic", 0.40), (6, 7, "semantic", 0.40),
         (8, 9, "semantic", 0.40), (10, 11, "semantic", 0.40)])
    assert crystallize.candidates(conn) == []                 # off by default
    cands = crystallize.candidates(conn, fallback=True)
    assert len(cands) == 1 and cands[0]["member_ids"] == [1, 2, 3]
```

> **Fixture math (why Z=2.0 holds here):** node strengths are
> `[1.9, 0.95, 0.95, 0.4×8]` → mean ≈ 0.636, std ≈ 0.451, threshold =
> mean + 2·std ≈ **1.54**. Only node 1 (1.9) clears it; nodes 2,3 (0.95) do not.
> Node 1's strong neighbours {2,3} satisfy `FALLBACK_MIN_DEG=2`, so the seed
> expands to `{1,2,3}`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py -k fallback -q`
Expected: FAIL (`candidates() got an unexpected keyword argument 'fallback'`).

- [ ] **Step 3: Implement**

```python
# src/tether/crystallize.py — add constants
FALLBACK_Z = 2.0                          # outlier threshold: mean + Z*std of node semantic weight
FALLBACK_MIN_DEG = 2                      # a fallback seed needs at least this many tight neighbors
```

```python
# src/tether/crystallize.py — add helper
def _semantic_outlier_seeds(conn, current):
    """Nodes whose summed semantic neighborhood weight is an outlier vs the
    store's own baseline. Computed only over materialized semantic edges
    (never all-pairs)."""
    node_w = {}
    for src, dst, w in conn.execute(
            "SELECT src, dst, weight FROM edges WHERE kind='semantic'").fetchall():
        if src in current and dst in current:
            node_w.setdefault(src, []).append((dst, w))
            node_w.setdefault(dst, []).append((src, w))
    if len(node_w) < 3:
        return []
    strengths = [sum(w for _, w in nbrs) for nbrs in node_w.values()]
    mean = sum(strengths) / len(strengths)
    var = sum((s - mean) ** 2 for s in strengths) / len(strengths)
    std = var ** 0.5
    threshold = mean + FALLBACK_Z * std
    seeds = []
    for node, nbrs in sorted(node_w.items()):
        strong = [d for d, w in nbrs if w >= EXPAND_COS]
        if sum(w for _, w in nbrs) >= threshold and len(strong) >= FALLBACK_MIN_DEG:
            seeds.append((node, {node, *strong}))
    return seeds
```

```python
# src/tether/crystallize.py — candidates() signature + fallback wiring
def candidates(conn, embedder=None, fallback=False):
    ...
        # after the peak-seeded groups loop builds `out`:
        if fallback:
            seen = {m for c in out for m in c["member_ids"]}
            for node, members in _semantic_outlier_seeds(conn, current):
                members = _expand(members, conn, current)
                member_ids = sorted(members)
                if len(member_ids) < MIN_CLUSTER:
                    continue
                if any(m in seen for m in member_ids):     # don't duplicate peak clusters
                    continue
                if _covered(member_ids, bases):
                    continue
                out.append({
                    "peak_key": (node, node),              # semantic seed: self-key
                    "member_ids": member_ids,
                    "member_titles": [r[0] for r in conn.execute(
                        "SELECT title FROM memories WHERE id IN (%s) ORDER BY id"
                        % ",".join("?" for _ in member_ids), member_ids).fetchall()],
                    "why": [("semantic-density", node, "fallback")],
                    "descriptor": _descriptor(conn, member_ids),
                })
                seen.update(member_ids)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crystallize.py -q`
Expected: PASS (all crystallize tests).

- [ ] **Step 5: Commit**

```bash
git add src/tether/crystallize.py tests/test_crystallize.py
git commit -m "feat(b2): gated semantic-density cold-start fallback (relative outlier)"
```

---

### Task 6: Memoized `crystallization_candidates()` + pull-only resource

**Files:**
- Modify: `src/tether/store.py` (`crystallization_candidates()` memoized on a cheap graph signature)
- Modify: `src/tether/server.py` (`tether://crystallization` resource)
- Test: `tests/test_store.py`, `tests/test_mcp.py`

**Interfaces:**
- Consumes: `crystallize.candidates` (Tasks 3-5), `config.crystallize_enabled`.
- Produces: `Store.crystallization_candidates() -> list[dict]` — `[]` when `crystallize` disabled; otherwise `crystallize.candidates(...)`, **process-memoized** on `(edge_count, max(updated_at))` so repeated reads in one pass don't recompute. Server resource `tether://crystallization` returns JSON of the candidates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py — append
def test_crystallization_candidates_empty_when_disabled():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=False)
    s.migrate()
    assert s.crystallization_candidates() == []

def test_crystallization_candidates_memoized_until_write():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    first = s.crystallization_candidates()
    # same signature -> same object identity (cache hit, no recompute)
    assert s.crystallization_candidates() is first

def test_dismiss_invalidates_candidate_memo():
    # Regression: dismiss_cluster writes crystallize_dismissed (not edges), so an
    # edges-only memo signature would NOT recompute and the dismissal would no-op.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True)
    s.migrate()
    s.crystallization_candidates()                  # populate the memo signature
    sig_before = s._cryst_sig
    s.dismiss_cluster(1, 2)                          # writes crystallize_dismissed
    s.crystallization_candidates()                  # must recompute
    assert s._cryst_sig != sig_before               # signature reflects the dismissal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_store.py -k "crystallization_candidates or dismiss_invalidates" -q`
Expected: FAIL (`AttributeError: crystallization_candidates`).

- [ ] **Step 3: Implement**

```python
# src/tether/store.py — in __init__, add cache slots after self._crystallize
        self._cryst_sig = None
        self._cryst_cache = []
```

```python
# src/tether/store.py — add method near boot_index()
    def crystallization_candidates(self) -> list:
        """Read-time derived view of principle candidates. [] when disabled.
        Process-memoized on a cheap graph signature (adjustment C) so repeated
        reads in one reflection pass don't recompute. Never raises.

        The signature MUST include the dismissed-set count: dismiss_cluster writes
        to crystallize_dismissed, NOT edges, so an edges-only signature would keep
        serving a dismissed candidate from cache for the life of the process
        (dismissal silently no-ops). Naming a principle self-invalidates because it
        adds crystallized edges."""
        if not self._crystallize:
            return []
        try:
            from . import crystallize
            sig = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM edges), "
                "(SELECT COALESCE(MAX(updated_at), '') FROM edges), "
                "(SELECT COUNT(*) FROM crystallize_dismissed)").fetchone()
            if sig != self._cryst_sig:
                self._cryst_cache = crystallize.candidates(self._conn, self._embedder)
                self._cryst_sig = sig
            return self._cryst_cache
        except Exception:
            return []
```

```python
# src/tether/server.py — add resource
@mcp.resource("tether://crystallization")
def crystallization() -> str:
    """Pull-only reflection view: candidate clusters that may want a name. Read
    it during a reflection pass (NOT auto-loaded). For each cluster, name it via
    remember(..., crystallizes=member_ids) or drop it via dismiss_cluster(peak).
    """
    try:
        return json.dumps({"candidates": _get_store().crystallization_candidates()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_store.py -k "crystallization_candidates or dismiss_invalidates" tests/test_mcp.py -q`
Expected: PASS (3 tests: disabled-empty, memoized-hit, dismiss-invalidates + the resource test).

- [ ] **Step 5: Commit**

```bash
git add src/tether/store.py src/tether/server.py tests/test_store.py tests/test_mcp.py
git commit -m "feat(b2): memoized crystallization_candidates + pull-only resource"
```

---

### Task 7: Recall no-regression — the #25 crystallized-hub guard

**Files:**
- Modify: `tests/test_store.py` (crystallized-hub recall spreading + no-demote)
- Modify: `bench/corpus.py` or `bench/conditions.py` (a crystallized-hub control case, if the real bench is run)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `remember(crystallizes=)` (Task 2), recall spreading (`KIND_W["crystallized"]`, Task 1).
- Produces: a passing hermetic guarantee that (a) recalling a source surfaces its principle across the crystallized edge, and (b) a max-fan-out crystallized hub does NOT bury a direct hit (the #25 shape).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py — append
def test_crystallized_edge_surfaces_principle_from_source():
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True, recall_budget=16)
    s.migrate()
    a = s.remember("project", "Auth", "we switched to JWT tokens")["id"]
    p = s.remember("reference", "Principle", "fail fast under load",
                   crystallizes=[a])["id"]
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert a in ids and p in ids                    # principle reached from its source

def test_crystallized_hub_does_not_bury_direct_hit():
    # #25 back-door: a max-fan-out principle must not outrank a query's own hit.
    pytest.importorskip("numpy")
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, embedder=FakeEmbedder(),
              assoc=True, crystallize=True, recall_budget=16)
    s.migrate()
    hits = [s.remember("project", f"n{i}", "quarterly pizza budget review")["id"]
            for i in range(6)]
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]  # the direct hit
    p = s.remember("reference", "Principle", "a big fan-out principle",
                   crystallizes=hits + [a])["id"]   # hub over everything incl. a
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert ids[0] == a                              # seed still dominates the hub
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_store.py -k "crystallized_edge_surfaces or crystallized_hub" -q`
Expected: BOTH PASS. The first must pass — `KIND_W["crystallized"] = 1.2 > 0` guarantees spreading reaches the principle from its source across the (bidirectionally-read) crystallized edge; if it fails, the edge isn't being written or read on both endpoints (revisit Task 1). The second must pass because the #25 protect-head guard locks the top v0.2 seeds above any spread-reached hub. If the second FAILS, lower `KIND_W["crystallized"]` (Task 1) until the seed dominates, then re-run.

- [ ] **Step 3: Implement / tune**

If `test_crystallized_hub_does_not_bury_direct_hit` fails, reduce `KIND_W["crystallized"]` in `src/tether/graph.py` (e.g. 1.2 → 1.0) — the protect-head guard from #25 locks the top v0.2 seeds, so this should already hold; the tune only matters if a crystallized hub reshuffles the tail enough to matter. Document the chosen value with a one-line comment tying it to this guard.

- [ ] **Step 4: Run the full recall + bench guard**

Run: `.venv/bin/python -m pytest tests/test_store.py -q && HF_HUB_OFFLINE=1 .venv/bin/python -m bench.run 2>&1 | tail -3`
Expected: recall tests PASS; bench `no-regression guard ... PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py tests/test_store.py
git commit -m "test(b2): crystallized-hub recall no-regression (#25 back-door guard)"
```

---

### Task 8: Docs + honest note

**Files:**
- Modify: `README.md` (crystallization section under "Self-organizing store")
- Test: none (docs)

**Interfaces:** consumes the whole feature; produces user-facing docs.

- [ ] **Step 1: Add the README section**

```markdown
## Crystallization (optional, off by default)

With `TETHER_CRYSTALLIZE=1`, tether reflects: it detects dense clusters of
related memories and offers them for naming. Read `tether://crystallization`
during a reflection pass (it is pull-only, never auto-loaded) to get candidate
clusters; name a real principle with `remember(..., crystallizes=[source_ids])`
— which writes the principle and links it over its sources — or drop a candidate
with `dismiss_cluster(id_a, id_b)`. Clusters are seeded by explicit links +
usage (semantic similarity fills out membership), so this finds *"these belong
together"* structure, not mere topical similarity. tether finds the structure;
your agent supplies the words.

A crystallized principle becomes a boot-index hub and is reachable from its
sources in recall. Note: this makes "named" a third importance signal alongside
"used" and "linked" — deliberate, since an agent judging something
principle-worthy is a strong signal.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(b2): crystallization README section + third-importance-axis note"
```

---

## Self-Review

**1. Spec coverage:**
- Foundation (semantic+explicit, behavioral booster W_b=0) → Task 3 (`PEAK_W`, `PEAK_KINDS`). ✓
- Seed-from-peak + bounded semantic-expand → Task 3. ✓
- Cold-start relative-outlier fallback, gated off → Task 5. ✓
- Pull-only memoized resource → Task 6. ✓
- `remember(crystallizes=)` → Task 2. ✓
- `crystallized` role matrix (spreading + hubs in, peak-seeding out; directional) → Task 1 + Task 3 (`PEAK_KINDS` excludes it). ✓
- Basis-recovery dedup → Task 4. ✓
- Peak-keyed dismissed-set → Task 4. ✓
- Opt-in / degrade-never → Tasks 1, 2, 6. ✓
- #25 back-door guard → Task 7. ✓
- Third-importance-axis note → Task 8. ✓
- Provenance-preservation (protect sources) — already holds via directional edges counting in `degree_map`; forgetting protection is emergent, no code needed. ✓

**2. Placeholder scan:** the emit block in Task 4 uses `out.append({ ... })` as shorthand for the Task-3 block — the implementer must reuse Task 3's exact emit dict. Flagged here so it isn't read as a literal placeholder; every other step carries complete code.

**3. Type consistency:** candidate dict shape `{peak_key, member_ids, member_titles, why, descriptor}` is identical across Tasks 3/5/6. `dismiss_peak`/`dismissed_peaks` (graph) vs `dismiss_cluster` (store/server) names are intentional (layer-appropriate). `crystallize.candidates(conn, embedder=None, fallback=False)` signature is stable from Task 5 on.

## Execution Handoff

Two execution options — see below.
