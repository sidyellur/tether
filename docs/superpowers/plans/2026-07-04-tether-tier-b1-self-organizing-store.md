# tether Tier B1 — Self-Organizing Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the store self-organizing — the boot-index surfaces the *load-bearing* memories at scale, and old + behaviorally-isolated memories can fade (reversibly, opt-in) — over Tier A's usage graph.

**Architecture:** One new graph primitive, `Graph.degree_map()` (weighted **behavioral** degree — `explicit` + `hebbian` only, never `semantic`), powers both features. `boot_index()` consumes it read-time to rank a bounded two-slice index; a bounded, opt-in `_run_forgetting_sweep()` consumes it to soft-archive faded memories by reusing the existing `valid_to`/`superseded_by` columns, triggered amortized every N writes.

**Tech Stack:** Python ≥3.10, stdlib `sqlite3`. No numpy, no embedder on any B1 path (degree is pure SQL/Python over `edges`). Design: `docs/superpowers/specs/2026-07-04-tether-tier-b1-self-organizing-store-design.md`.

## ⛔ Hard dependency — Tier A must be merged first

This plan builds directly on **Tier A: Associative Core** (issue #16, plan `docs/superpowers/plans/2026-07-04-tether-associative-core.md`). Before starting, confirm on `main`:
- `src/tether/graph.py` exists with `class Graph(conn, enabled=)`, `.enabled`, `.migrate()`, `._upsert_edge(a,b,kind,weight,now,mode)`, and the `edges` table (`src, dst, kind, weight, updated_at`; PK `(src,dst,kind)`; canonical `src < dst`; kinds `semantic`/`explicit`/`hebbian`).
- `Store.__init__` already has `assoc=False, recall_budget=24` and sets `self._graph = Graph(conn, enabled=assoc)`; `Store.migrate()` calls `self._graph.migrate()`.
- `tests/test_graph.py` defines the helpers `make_graph()` and `_mem(conn, mid, type="user", valid=True)`.

If any of these is absent, **stop** — Tier A is not merged and B1 has no substrate.

## Global Constraints

- **Python ≥3.10, POSIX.** Distribution `tether-memory`; import/CLI `tether`.
- **Still four verbs only.** No new tool. B1 changes only the boot-index *content* and adds an internal maintenance sweep; the MCP surface is unchanged.
- **Behavioral degree, never semantic.** `degree_map` counts `explicit` + `hebbian` edges only. Semantic similarity must not confer degree (it would neuter forgetting and make hubs mean "redundant").
- **Reuse, don't add.** No new verb, no new module, **no schema migration** — forgetting archives via the existing `valid_to`/`superseded_by` columns (already present from bet 2) and a `meta` counter.
- **Opt-in for anything destructive.** Forgetting is off by default (`TETHER_FORGET`), like consolidation/dedup/decay. Hub-curation is on by default but only activates with a graph present and `count > CAP`.
- **Degrade-never.** No graph (`assoc` off) ⇒ boot-index is today's unbounded newest-first and forgetting is a hard no-op. `TETHER_FORGET` off ⇒ `recall` and forgetting byte-identical to pre-B1.
- **Determinism.** Every ordering breaks ties explicitly (degree desc → `updated_at` desc → id desc). No RNG.
- Commit after every task. Sign commits with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pyproject.toml` | version bump `0.5.0` (assumes Tier A's `0.4.0` is merged) | Modify |
| `src/tether/config.py` | `boot_index_cap`, `forget_enabled`, `forget_age_days`, `forget_interval`, `forget_max_per_sweep` | Modify |
| `src/tether/graph.py` | `Graph.degree_map()` — behavioral weighted degree, emits zeros | Modify |
| `src/tether/store.py` | hub-curated `boot_index()`; `_run_forgetting_sweep()` + amortized trigger; new constructor fields | Modify |
| `src/tether/server.py` | `_get_store()` wires B1 config into `Store` | Modify |
| `tests/test_config.py` | the five resolvers | Modify |
| `tests/test_graph.py` | `degree_map` | Modify |
| `tests/test_store.py` | boot-index curation, forgetting sweep, trigger | Modify |
| `tests/test_mcp.py` | server wiring | Modify |
| `README.md` | "Self-organizing store (optional)" section | Modify |

## Test Strategy

**Philosophy:** TDD per task. **Every B1 test is fully hermetic — no embedder, no numpy, no network.** `degree_map` is pure SQL/Python over `edges`, so tests hand-insert `edges` rows of specific kinds and set `updated_at` directly via SQL to simulate age (never freeze the clock). This is the whole point of the "behavioral degree" decision being SQL-only.

**Levels:**
- *Unit* — the five config resolvers; `Graph.degree_map` against hand-built memories + edges; `_curated_index` ordering; each forgetting gate and safety rail in isolation.
- *Integration* — `Store.boot_index()` end-to-end across all four regimes; `_run_forgetting_sweep()` archive mechanics + reversibility; the amortized trigger firing through `remember()`; `server._get_store` wiring.

**Hermeticity controls:** in-memory SQLite; `assoc=True` to enable the graph without any embedder; hand-inserted `edges`; `updated_at` aged via `UPDATE`; small caps/intervals passed to the constructor so tests stay tiny.

**Coverage matrix (guarantee → test):**

| Guarantee | Task / test |
|---|---|
| Config defaults + parsing (incl. `<1` → default) | Task 1 `test_boot_index_cap_*`, `test_forget_*` |
| Behavioral degree excludes semantic; emits zeros; current-only | Task 2 `test_degree_map_*` |
| Small store unchanged; curated two-slice above CAP; recent-only when no hubs; unbounded when graph off | Task 3 `test_boot_index_*` |
| Only old+behaviorally-isolated archived; semantic doesn't protect | Task 4 `test_forgetting_archives_*`, `test_forgetting_keeps_*`, `test_forgetting_semantic_only_*` |
| Safety rails (disabled, no-behavioral-graph, size-floor, per-sweep cap) | Task 4 `test_forgetting_noop_*`, `test_forgetting_respects_size_floor`, `test_forgetting_bounded_per_sweep` |
| Archive reversible + audit-preserving; edges retained | Task 4 `test_forgetting_is_reversible` |
| Amortized trigger fires at interval; no counter when disabled | Task 5 `test_forget_trigger_*` |
| Server wires B1 config | Task 6 `test_server_wires_forget_config` |
| `TETHER_FORGET` off ⇒ remember untouched | Task 5 `test_forget_trigger_disabled_never_fires` |

**Deliberate non-goals:** PageRank centrality; relaxed low-degree forgetting; an un-forget verb; edge decay; crystallization (B2).

---

### Task 1: config — the five B1 resolvers + version bump

**Files:**
- Modify: `src/tether/config.py` (append)
- Modify: `pyproject.toml`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `config.boot_index_cap() -> int` (`TETHER_BOOT_INDEX_CAP`, default 50; `<1` or unparseable → 50), `config.forget_enabled() -> bool` (`TETHER_FORGET`, default False; `1/true/yes/on` enables), `config.forget_age_days() -> int` (default 90), `config.forget_interval() -> int` (default 20), `config.forget_max_per_sweep() -> int` (default 10). The three numeric forget resolvers share the `<1` → default rule.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py (append)

def test_boot_index_cap_default_and_parsing(monkeypatch):
    monkeypatch.delenv("TETHER_BOOT_INDEX_CAP", raising=False)
    assert config.boot_index_cap() == 50
    monkeypatch.setenv("TETHER_BOOT_INDEX_CAP", "10")
    assert config.boot_index_cap() == 10
    monkeypatch.setenv("TETHER_BOOT_INDEX_CAP", "0")
    assert config.boot_index_cap() == 50          # <1 → default
    monkeypatch.setenv("TETHER_BOOT_INDEX_CAP", "junk")
    assert config.boot_index_cap() == 50


def test_forget_enabled_default_off(monkeypatch):
    monkeypatch.delenv("TETHER_FORGET", raising=False)
    assert config.forget_enabled() is False
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("TETHER_FORGET", v)
        assert config.forget_enabled() is True
    monkeypatch.setenv("TETHER_FORGET", "0")
    assert config.forget_enabled() is False


def test_forget_numeric_configs(monkeypatch):
    cases = [("TETHER_FORGET_AGE_DAYS", config.forget_age_days, 90),
             ("TETHER_FORGET_INTERVAL", config.forget_interval, 20),
             ("TETHER_FORGET_MAX_PER_SWEEP", config.forget_max_per_sweep, 10)]
    for name, fn, default in cases:
        monkeypatch.delenv(name, raising=False)
        assert fn() == default
        monkeypatch.setenv(name, "5")
        assert fn() == 5
        monkeypatch.setenv(name, "0")
        assert fn() == default                    # <1 → default
        monkeypatch.setenv(name, "x")
        assert fn() == default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -k "boot_index or forget" -v`
Expected: FAIL — `AttributeError: module 'tether.config' has no attribute 'boot_index_cap'`.

- [ ] **Step 3: Append the resolvers to `src/tether/config.py`**

```python
_FORGET_ON = {"1", "true", "yes", "on"}
_DEFAULT_BOOT_INDEX_CAP = 50
_DEFAULT_FORGET_AGE_DAYS = 90
_DEFAULT_FORGET_INTERVAL = 20
_DEFAULT_FORGET_MAX_PER_SWEEP = 10


def _pos_int(env: str, default: int) -> int:
    """A positive integer from the env, or the default (also on <1/unparseable)."""
    raw = os.environ.get(env)
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val >= 1 else default


def boot_index_cap() -> int:
    """Boot-index size above which hub-curation kicks in (needs a graph)."""
    return _pos_int("TETHER_BOOT_INDEX_CAP", _DEFAULT_BOOT_INDEX_CAP)


def forget_enabled() -> bool:
    """Forgetting sweep is opt-in, off by default."""
    val = os.environ.get("TETHER_FORGET")
    if val is None:
        return False
    return val.strip().lower() in _FORGET_ON


def forget_age_days() -> int:
    return _pos_int("TETHER_FORGET_AGE_DAYS", _DEFAULT_FORGET_AGE_DAYS)


def forget_interval() -> int:
    return _pos_int("TETHER_FORGET_INTERVAL", _DEFAULT_FORGET_INTERVAL)


def forget_max_per_sweep() -> int:
    return _pos_int("TETHER_FORGET_MAX_PER_SWEEP", _DEFAULT_FORGET_MAX_PER_SWEEP)
```

- [ ] **Step 4: Bump the version in `pyproject.toml`**

Change `version = "0.4.0"` to:

```toml
version = "0.5.0"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all config tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tether/config.py pyproject.toml tests/test_config.py
git commit -m "feat: B1 config resolvers (boot-index cap + forgetting knobs)"
```

---

### Task 2: `Graph.degree_map` — behavioral weighted degree

**Files:**
- Modify: `src/tether/graph.py` (add method to `Graph`)
- Test: `tests/test_graph.py` (append)

**Interfaces:**
- Consumes (Tier A): `Graph._conn`, the `edges` table, `_mem(conn, mid, type=, valid=)` test helper.
- Produces: `Graph.degree_map(kinds=("explicit", "hebbian")) -> dict[int, float]` — for every current (`valid_to IS NULL`) memory, its summed incident edge `weight` over edges of the given `kinds` whose **other endpoint is also current**; nodes with no such edge appear with `0.0`. `semantic` is excluded by default. Deterministic; O(memories + edges); any failure → `{}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py (append)

def test_degree_map_behavioral_only_excludes_semantic():
    g = make_graph()
    for i in (1, 2, 3):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "semantic", 0.9, "t", mode="max")
    g._upsert_edge(1, 3, "hebbian", 0.5, "t", mode="max")
    g._upsert_edge(2, 3, "explicit", 1.0, "t", mode="max")
    deg = g.degree_map()
    assert deg[1] == 0.5          # only the hebbian edge (semantic ignored)
    assert deg[2] == 1.0          # only the explicit edge
    assert deg[3] == 1.5          # hebbian 0.5 + explicit 1.0


def test_degree_map_emits_zero_for_semantic_only_node():
    g = make_graph()
    for i in (1, 2):
        _mem(g._conn, i)
    g._upsert_edge(1, 2, "semantic", 0.9, "t", mode="max")
    assert g.degree_map() == {1: 0.0, 2: 0.0}    # present, but zero


def test_degree_map_ignores_edges_to_noncurrent():
    g = make_graph()
    _mem(g._conn, 1)
    _mem(g._conn, 2, valid=False)               # archived
    g._upsert_edge(1, 2, "hebbian", 1.0, "t", mode="max")
    assert g.degree_map() == {1: 0.0}           # node 2 not current; its edge doesn't count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -k degree_map -v`
Expected: FAIL — `AttributeError: 'Graph' object has no attribute 'degree_map'`.

- [ ] **Step 3: Add `degree_map` to `src/tether/graph.py`**

```python
    def degree_map(self, kinds=("explicit", "hebbian")) -> dict:
        """Behavioral weighted degree for every current memory (semantic edges
        excluded by default). Emits explicit 0.0 for isolated current nodes -
        forgetting needs the degree-0 set, which a pure edge-scan can't produce.
        Only edges between two current nodes count. Never raises."""
        try:
            current = {r[0] for r in self._conn.execute(
                "SELECT id FROM memories WHERE valid_to IS NULL").fetchall()}
            deg = {mid: 0.0 for mid in current}
            if not kinds:
                return deg
            ph = ",".join("?" for _ in kinds)
            rows = self._conn.execute(
                f"SELECT src, dst, weight FROM edges WHERE kind IN ({ph})",
                tuple(kinds)).fetchall()
            for src, dst, w in rows:
                if src in current and dst in current:
                    deg[src] += w
                    deg[dst] += w
            return deg
        except Exception:
            return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_graph.py -k degree_map -v`
Expected: the three degree_map tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tether/graph.py tests/test_graph.py
git commit -m "feat: Graph.degree_map - behavioral weighted degree over the usage graph"
```

---

### Task 3: hub-curated `boot_index()`

**Files:**
- Modify: `src/tether/store.py` (`__init__`, `boot_index`, new `_curated_index`)
- Test: `tests/test_store.py` (append)

**Interfaces:**
- Consumes: `Graph.degree_map()` (Task 2), `self._graph.enabled` (Tier A).
- Produces:
  - `Store.__init__(..., boot_index_cap=50, forget=False, forget_age_days=90, forget_interval=20, forget_max_per_sweep=10)` — **all** B1 constructor fields are added here (Tasks 4–5 use the forget fields); appended after Tier A's `assoc`/`recall_budget`.
  - `Store.boot_index() -> str` — today's unbounded newest-first when the graph is off or `count ≤ cap`; otherwise the two-slice curated index.
  - `Store._curated_index(rows, deg, cap) -> str` where `rows` is `[(id, type, title, updated_at)]` newest-first and `deg` is `{id: float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py (append)
import sqlite3
from tether.store import Store


def make_b1_store(assoc=True, **kw):
    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, assoc=assoc, **kw)
    s.migrate()
    return s


def _add_edge(s, a, b, kind="hebbian", w=1.0):
    lo, hi = (a, b) if a < b else (b, a)
    s._conn.execute("INSERT INTO edges(src, dst, kind, weight, updated_at) "
                    "VALUES (?,?,?,?,?)", (lo, hi, kind, w, "t"))
    s._conn.commit()


def test_boot_index_small_store_unchanged():
    s = make_b1_store(boot_index_cap=50)
    for i in range(3):
        s.remember("user", f"T{i}", "b")
    idx = s.boot_index()
    assert "# Load-bearing" not in idx
    assert len(idx.splitlines()) == 3


def test_boot_index_curates_above_cap_with_hubs():
    s = make_b1_store(boot_index_cap=4)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(8)]
    hub = ids[0]                                  # oldest -> not in the recent reserve
    for other in ids[1:4]:
        _add_edge(s, hub, other, "hebbian", 1.0)
    idx = s.boot_index()
    assert "# Load-bearing" in idx and "# Recent" in idx
    assert f"#{hub} " in idx.split("# Recent")[0]         # hub is in the load-bearing slice
    body = [ln for ln in idx.splitlines() if not ln.startswith("#")]
    assert len(body) <= 4                                 # capped


def test_boot_index_recent_only_when_no_behavioral_hubs():
    s = make_b1_store(boot_index_cap=4)
    for i in range(8):
        s.remember("user", f"T{i}", "b")          # no edges at all
    idx = s.boot_index()
    assert "# Load-bearing" not in idx            # no hubs -> recent-only, no headers
    assert len(idx.splitlines()) == 4             # bounded to cap


def test_boot_index_unbounded_when_graph_disabled():
    s = make_b1_store(assoc=False, boot_index_cap=4)   # graph OFF
    for i in range(8):
        s.remember("user", f"T{i}", "b")
    assert len(s.boot_index().splitlines()) == 8       # no curation without a graph
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -k boot_index -v`
Expected: FAIL — `TypeError: Store.__init__() got an unexpected keyword argument 'boot_index_cap'`.

- [ ] **Step 3: Add B1 constructor fields to `Store.__init__`**

Replace the (post-Tier-A) signature and add the field assignments at the end of the body:

```python
    def __init__(self, conn, device_id: str, sync_now, embedder=None,
                 author="", consolidate=False, dedup_threshold=0.92,
                 decay_half_life_days=None, assoc=False, recall_budget=24,
                 boot_index_cap=50, forget=False, forget_age_days=90,
                 forget_interval=20, forget_max_per_sweep=10):
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
        self._boot_index_cap = boot_index_cap
        self._forget = forget
        self._forget_age_days = forget_age_days
        self._forget_interval = forget_interval
        self._forget_max_per_sweep = forget_max_per_sweep
```

(This assumes Tier A's `__init__` already builds `self._graph` and holds `self._recall_budget`; B1 only appends the five new fields.)

- [ ] **Step 4: Replace `Store.boot_index` and add `_curated_index`**

```python
    def boot_index(self) -> str:
        rows = self._conn.execute(
            "SELECT id, type, title, updated_at FROM memories WHERE valid_to IS NULL "
            "ORDER BY updated_at DESC, id DESC").fetchall()
        if not rows:
            return "(no memories yet)"
        full = "\n".join(f"[{t}] #{i} {title}" for i, t, title, _ in rows)
        if not self._graph.enabled or len(rows) <= self._boot_index_cap:
            return full                                  # today's behavior
        try:
            return self._curated_index(rows, self._graph.degree_map(),
                                       self._boot_index_cap)
        except Exception:
            return full                                  # curation failure -> full list

    def _curated_index(self, rows, deg, cap) -> str:
        # rows: [(id, type, title, updated_at)] newest-first
        meta = {r[0]: (r[1], r[2]) for r in rows}        # id -> (type, title)
        upd = {r[0]: r[3] for r in rows}                 # id -> updated_at (tie-break)
        newest = [r[0] for r in rows]
        reserve = max(1, cap // 4)
        recent = newest[:reserve]
        recent_set = set(recent)
        hubs = sorted(
            (mid for mid in deg if deg[mid] > 0 and mid not in recent_set),
            key=lambda mid: (deg[mid], upd[mid], mid), reverse=True,
        )[: cap - reserve]                               # degree desc, then updated_at desc, then id desc
        chosen = set(hubs) | recent_set
        for mid in newest:                               # fill remaining budget with recency
            if len(chosen) >= cap:
                break
            if mid not in chosen:
                recent.append(mid)
                chosen.add(mid)

        def line(mid):
            t, title = meta[mid]
            return f"[{t}] #{mid} {title}"

        if not hubs:
            return "\n".join(line(mid) for mid in recent)
        parts = ["# Load-bearing"] + [line(mid) for mid in hubs]
        parts += ["# Recent"] + [line(mid) for mid in recent]
        return "\n".join(parts)
```

Note: `deg` (from `degree_map`) includes every current node keyed by id, and `upd`/`meta` cover exactly the current rows, so `upd[mid]` is always present for any `mid` in `deg`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -k boot_index -v`
Expected: the four boot-index tests pass.

- [ ] **Step 6: Run the whole store suite (no regressions)**

Run: `python -m pytest tests/test_store.py -q`
Expected: all pass (existing recall/boot tests unaffected — their stores are small or graph-off).

- [ ] **Step 7: Commit**

```bash
git add src/tether/store.py tests/test_store.py
git commit -m "feat: hub-curated boot-index (two labeled slices over behavioral degree)"
```

---

### Task 4: forgetting-by-disconnection — the sweep

**Files:**
- Modify: `src/tether/store.py` (add `_run_forgetting_sweep`)
- Test: `tests/test_store.py` (append)

**Interfaces:**
- Consumes: `Graph.degree_map()` (Task 2); the `self._forget*` / `self._boot_index_cap` fields (Task 3); module helpers `_now`, `_age_days` (existing in `store.py`).
- Produces: `Store._run_forgetting_sweep() -> int` — archives (sets `valid_to=now`, leaves `superseded_by` NULL) up to `forget_max_per_sweep` current memories that are **old** (`updated_at` age > `forget_age_days`) **and behaviorally isolated** (`degree_map` value 0). No-op (returns 0) unless enabled, a live behavioral graph exists, and the store is above the size floor. Never raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py (append)
_OLD = "2020-01-01T00:00:00+00:00"


def make_forget_store(**kw):
    kw.setdefault("boot_index_cap", 2)            # size floor = 2*2 = 4
    kw.setdefault("forget_max_per_sweep", 10)
    return make_b1_store(assoc=True, forget=True, **kw)


def _age(s, mid, iso=_OLD):
    s._conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (iso, mid))
    s._conn.commit()


def test_forgetting_archives_old_isolated():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")       # live behavioral graph elsewhere
    _age(s, ids[0])                                # old + isolated
    assert s._run_forgetting_sweep() == 1
    vt, sb = s._conn.execute(
        "SELECT valid_to, superseded_by FROM memories WHERE id=?", (ids[0],)).fetchone()
    assert vt is not None and sb is None           # archived, not superseded
    assert ids[0] not in [h["id"] for h in s.recall("T0")]


def test_forgetting_keeps_old_but_connected():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[0], ids[1], "explicit")       # behaviorally connected
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_keeps_isolated_but_recent():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    # ids[0] isolated but NOT aged -> kept
    assert s._run_forgetting_sweep() == 0


def test_forgetting_semantic_only_does_not_protect():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")        # live behavioral graph
    _add_edge(s, ids[0], ids[1], "semantic")       # ids[0] has ONLY a semantic edge
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 1          # semantic doesn't protect


def test_forgetting_noop_when_disabled():
    s = make_b1_store(assoc=True, forget=False, boot_index_cap=2)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_noop_without_behavioral_graph():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _age(s, ids[0])                                # old + isolated, but NO behavioral edges anywhere
    assert s._run_forgetting_sweep() == 0


def test_forgetting_respects_size_floor():
    s = make_forget_store()                        # cap=2 -> floor 4
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(3)]   # only 3 < 4
    _add_edge(s, ids[1], ids[2], "hebbian")
    _age(s, ids[0])
    assert s._run_forgetting_sweep() == 0


def test_forgetting_bounded_per_sweep():
    s = make_forget_store(forget_max_per_sweep=2)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(8)]
    _add_edge(s, ids[6], ids[7], "hebbian")        # keep two connected (live graph)
    for i in range(6):
        _age(s, ids[i])                            # 6 old + isolated
    assert s._run_forgetting_sweep() == 2          # capped


def test_forgetting_is_reversible():
    s = make_forget_store()
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])
    s._run_forgetting_sweep()
    s._conn.execute("UPDATE memories SET valid_to=NULL WHERE id=?", (ids[0],))
    s._conn.commit()
    assert ids[0] in [h["id"] for h in s.recall("T0")]           # un-forgotten
    assert s._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] >= 1   # edges retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -k forgetting -v`
Expected: FAIL — `AttributeError: 'Store' object has no attribute '_run_forgetting_sweep'`.

- [ ] **Step 3: Add `_run_forgetting_sweep` to `src/tether/store.py`**

```python
    def _run_forgetting_sweep(self) -> int:
        """Soft-archive old + behaviorally-isolated memories (opt-in, bounded,
        reversible). Returns the number archived. Never raises."""
        if not self._forget:
            return 0
        try:
            deg = self._graph.degree_map()          # behavioral, {} if unavailable
            if not any(v > 0 for v in deg.values()):
                return 0                            # no live behavioral graph -> refuse
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
            if count < 2 * self._boot_index_cap:
                return 0                            # store-size floor
            now = _now()
            rows = self._conn.execute(
                "SELECT id, updated_at FROM memories WHERE valid_to IS NULL "
                "ORDER BY updated_at ASC, id ASC").fetchall()   # oldest first
            archived = 0
            for mid, updated_at in rows:
                if archived >= self._forget_max_per_sweep:
                    break
                if _age_days(updated_at, now) <= self._forget_age_days:
                    break                           # rest are younger (ordered) -> done
                if deg.get(mid, 0.0) > 0:
                    continue                        # behaviorally connected -> keep
                self._conn.execute(
                    "UPDATE memories SET valid_to=? WHERE id=?", (now, mid))
                archived += 1
            if archived:
                self._conn.commit()
            return archived
        except Exception:
            return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -k forgetting -v`
Expected: all nine forgetting tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tether/store.py tests/test_store.py
git commit -m "feat: forgetting-by-disconnection sweep (old + behaviorally-isolated, reversible)"
```

---

### Task 5: the amortized trigger

**Files:**
- Modify: `src/tether/store.py` (`remember` calls `_maybe_forget`; add `_maybe_forget`)
- Test: `tests/test_store.py` (append)

**Interfaces:**
- Consumes: `_run_forgetting_sweep` (Task 4); `_meta_get`/`_meta_set` (existing); `self._forget`, `self._forget_interval` (Task 3).
- Produces: `Store._maybe_forget()` — increments the `meta` key `forget_counter` per call; when it reaches `forget_interval`, resets it and runs the sweep. When `self._forget` is False it returns immediately without touching `meta` (so `remember` is byte-identical to pre-B1). Called at the end of `remember`. Never raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py (append)

def test_forget_trigger_fires_at_interval():
    s = make_forget_store(forget_interval=3)
    ids = [s.remember("user", f"T{i}", "b")["id"] for i in range(6)]
    _add_edge(s, ids[4], ids[5], "hebbian")
    _age(s, ids[0])                                # now old + isolated
    before = s._conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (ids[0],)).fetchone()[0]
    for i in range(3):                             # 3 writes -> counter hits interval
        s.remember("user", f"X{i}", "b")
    after = s._conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (ids[0],)).fetchone()[0]
    assert before is None and after is not None    # the trigger archived it


def test_forget_trigger_disabled_never_fires():
    s = make_b1_store(assoc=True, forget=False, boot_index_cap=2)
    for i in range(6):
        s.remember("user", f"T{i}", "b")
    assert s._conn.execute(
        "SELECT value FROM meta WHERE key='forget_counter'").fetchone() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -k forget_trigger -v`
Expected: FAIL — `test_forget_trigger_fires_at_interval` fails (`after` is still None; nothing triggers the sweep).

- [ ] **Step 3: Wire the trigger into `remember` and add `_maybe_forget`**

In `remember`, add the call as the final statement before `return` (after `self._sync_now()`):

```python
        self._sync_now()
        self._maybe_forget()
        return {"id": mid, "action": action}
```

Add the method:

```python
    def _maybe_forget(self) -> None:
        """Amortized trigger: every forget_interval writes, run one bounded
        sweep. No-op (and no meta writes) when forgetting is disabled."""
        if not self._forget:
            return
        try:
            n = int(self._meta_get("forget_counter") or 0) + 1
            if n >= self._forget_interval:
                self._meta_set("forget_counter", 0)
                self._conn.commit()
                self._run_forgetting_sweep()
            else:
                self._meta_set("forget_counter", n)
                self._conn.commit()
        except Exception:
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -k forget_trigger -v`
Expected: both trigger tests pass.

- [ ] **Step 5: Run the whole store suite**

Run: `python -m pytest tests/test_store.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/tether/store.py tests/test_store.py
git commit -m "feat: amortized forgetting trigger (every N writes; no-op when disabled)"
```

---

### Task 6: server wiring + docs

**Files:**
- Modify: `src/tether/server.py` (`_get_store`)
- Modify: `tests/test_mcp.py` (append)
- Modify: `README.md`

**Interfaces:**
- Consumes: the five `config` resolvers (Task 1); the B1 `Store` constructor fields (Task 3).
- Produces: the lazily-built store carries the B1 config; the MCP surface is otherwise unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp.py (append)

def test_server_wires_forget_config(monkeypatch, tmp_path):
    from tether import server
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("TETHER_SEMANTIC", "0")     # no embedder needed
    monkeypatch.setenv("TETHER_FORGET", "1")
    monkeypatch.setenv("TETHER_BOOT_INDEX_CAP", "7")
    server._store = None
    try:
        s = server._get_store()
        assert s._forget is True
        assert s._boot_index_cap == 7
    finally:
        server._store = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp.py -k forget_config -v`
Expected: FAIL — `s._forget` is False (server doesn't pass B1 config).

- [ ] **Step 3: Wire the config in `src/tether/server.py`**

Extend the `Store(...)` construction in `_get_store` (add the B1 kwargs after the Tier A ones):

```python
        store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                      embedder=embedder, author=config.author(),
                      consolidate=config.consolidate_enabled(),
                      dedup_threshold=config.dedup_threshold(),
                      decay_half_life_days=config.decay_half_life_days(),
                      assoc=config.assoc_enabled(),
                      recall_budget=config.recall_budget(),
                      boot_index_cap=config.boot_index_cap(),
                      forget=config.forget_enabled(),
                      forget_age_days=config.forget_age_days(),
                      forget_interval=config.forget_interval(),
                      forget_max_per_sweep=config.forget_max_per_sweep())
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Update `README.md`**

Add after the "Associative recall (optional)" section:

````markdown
## Self-organizing store (optional)

As a store grows, tether keeps it legible using the same usage graph:

- **Hub-curated boot-index.** The auto-loaded memory index is capped once it
  passes `TETHER_BOOT_INDEX_CAP` (default 50) and a graph exists. Above the cap
  it shows two labeled slices — **load-bearing** memories (highest *behavioral*
  degree: `explicit` links + learned co-recall, never mere similarity) and the
  most **recent** ones — so the index stays small and shows what actually
  matters. Below the cap, or without a graph, it's the full newest-first list as
  before.
- **Forgetting-by-disconnection** (opt-in, `TETHER_FORGET`). A bounded sweep
  runs every `TETHER_FORGET_INTERVAL` writes and *soft-archives* memories that
  are both **old** (`TETHER_FORGET_AGE_DAYS`, default 90) and **behaviorally
  isolated** (no `explicit`/`hebbian` edge — semantic similarity doesn't count).
  Archived memories drop out of recall and the boot-index but are **retained and
  reversible** (it reuses the same mark-invalid machinery as consolidation;
  nothing is deleted). Safety rails: never runs without a live behavioral graph,
  below `2 × CAP` memories, or more than `TETHER_FORGET_MAX_PER_SWEEP` (default
  10) per sweep.

| var | default | effect |
|---|---|---|
| `TETHER_BOOT_INDEX_CAP` | `50` | curate the boot-index above this size |
| `TETHER_FORGET` | off | enable the forgetting sweep |
| `TETHER_FORGET_AGE_DAYS` | `90` | minimum age to be eligible to fade |
| `TETHER_FORGET_INTERVAL` | `20` | writes between sweeps |
| `TETHER_FORGET_MAX_PER_SWEEP` | `10` | max archived per sweep |

With `TETHER_FORGET` off (default) and a normal store size, recall and the
boot-index behave exactly as before.
````

- [ ] **Step 6: Commit**

```bash
git add src/tether/server.py tests/test_mcp.py README.md
git commit -m "feat: wire B1 self-organizing-store config into the server; docs"
```

---

## Self-Review

**1. Spec coverage:**
- Hub-curated boot-index (read-time, two slices, cap-gated, graph-gated) → Task 3. ✓
- Forgetting (two gates old ∧ behaviorally-isolated, soft-archive via `valid_to`, safety rails) → Task 4. ✓
- Behavioral degree excludes semantic; emits zeros → Task 2. ✓
- Amortized inline trigger → Task 5. ✓
- Config (`TETHER_BOOT_INDEX_CAP`, `TETHER_FORGET*`) → Task 1. ✓
- Degrade-never (graph off → unbounded; forget off → identical remember) → Tasks 3, 5, tested. ✓
- No new verb / no new module / no schema migration → nothing adds a tool, table, or column. ✓
- Reversibility + edges retained → Task 4 `test_forgetting_is_reversible`. ✓
- Fully hermetic tests (no embedder/numpy) → every test uses `assoc=True` + hand-inserted edges. ✓

**2. Placeholder scan:** No TBD/TODO. Every code step is complete. The one cross-task assumption (Tier A merged) is called out in a hard dependency gate, not left implicit.

**3. Type consistency:** `degree_map(kinds=...) -> dict[int,float]` produced in Task 2, consumed identically in Tasks 3 & 4. `_curated_index(rows, deg, cap)` signature matches its Task 3 caller; `rows` is the 4-tuple `(id, type, title, updated_at)` produced by `boot_index`'s query and read by `_curated_index`. `_run_forgetting_sweep() -> int` (Task 4) called by `_maybe_forget` (Task 5). All five constructor fields added once in Task 3 and read in Tasks 3–5. Config resolver names in Task 6's `Store(...)` match Task 1 exactly.

## Execution notes

- Tasks are dependency-ordered; each ends green and independently reviewable.
- **Do not start before Tier A is merged** (see the hard dependency gate) — the constructor edits, `self._graph`, and the `edges` table all come from Tier A.
- Existing tests stay green: `assoc` defaults False (boot-index unbounded, graph off) and `forget` defaults False (no trigger), so no pre-B1 behavior changes unless a test opts in.
