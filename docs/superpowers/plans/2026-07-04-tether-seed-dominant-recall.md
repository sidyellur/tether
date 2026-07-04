# Seed-Dominant Recall Ranking Implementation Plan

> **⚠️ Superseded mechanism:** this plan specifies a strict *two-tier* ranking.
> The bench harness rejected that (it flattened graph-only recall to the v0.2
> baseline) and a seed-floor variant (it regressed control). The shipped fix is
> **protect-head / re-rank-tail** — see the "Design evolution" section of the
> spec (`2026-07-04-tether-seed-dominant-recall-design.md`). The tasks below are
> kept as the historical plan; the goal, budget change, and test intent still
> hold, but the ranking code differs from Task 1's snippet.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a query's own direct matches structurally undisplaceable by spread-reached nodes, fixing the #25 degrade-never violation at the quality layer, while preserving the usage-graph upside.

**Architecture:** Split `recall`'s final ranking into two tiers — v0.2 seeds (in exact v0.2 order) on top, everything reached by priming/spread below, by activation. Drop the default recall budget 24 → 8 (now a pure upside/cost knob since seeds are protected at every budget). The change is confined to the order-assembly in `Store.recall`; `spread()`, edge weights, HOP_DECAY, and receipts are untouched.

**Tech Stack:** Python 3, sqlite3, numpy (optional, already a soft dep), Model2Vec (real-run only, local).

## Global Constraints

- Degrade-never holds unchanged: `assoc=False` recall stays byte-identical to v0.2 (verified by an existing test — do not break it).
- No new dependencies. Hermetic tests use the existing `FakeEmbedder`; the real model is only for the local acceptance run.
- The fix touches only `Store.recall`'s order assembly and two budget-default constants. Do NOT modify `graph.spread`, `KIND_W`, `HOP_DECAY`, `HEBBIAN_CAP`, or the `via` receipt shape.
- Spec: `docs/superpowers/specs/2026-07-04-tether-seed-dominant-recall-design.md`. Fixes issue #25.

## File Structure

- Modify: `src/tether/store.py` — `recall()` order assembly (Task 1).
- Modify: `src/tether/config.py:91` and `src/tether/store.py:138` — budget default (Task 2).
- Modify: `tests/test_store.py` — new ranking tests (Task 1).
- Modify: `tests/test_config.py:100,106,108` — default budget expectation (Task 2).
- Modify: `pyproject.toml`, `README.md` — version + docs note (Task 3).

---

### Task 1: Two-tier rank fusion in `recall()`

**Files:**
- Modify: `src/tether/store.py:468-499` (the `recall` method)
- Test: `tests/test_store.py` (append after `test_recall_budget_zero_is_passthrough`, ~line 562)

**Interfaces:**
- Consumes: `self._seed_scores(query, type) -> {id: score}` (v0.2 hybrid seeds); `self._graph.session_activation(sid) -> {id: activation}`; `self._graph.spread(seed_activation, budget, type) -> (activation: {id: float}, receipts: {id: dict})`; module constant `_PRIMING_WEIGHT = 0.25`.
- Produces: `recall(...) -> list[dict]` with the same shape as today (adds `via` per hit); ranking now two-tier.

- [ ] **Step 1: Write the failing test — a high-weight Hebbian neighbor cannot outrank the seed (the exact #25 failure)**

```python
def test_recall_seed_not_buried_by_high_weight_hebbian_neighbor():
    # #25: a within-task co-recalled neighbor (NOT a query match), reached over a
    # capped Hebbian edge (factor 5.0*0.4=2.0, amplifying), must not outrank the
    # query's own direct hit.
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]     # matches query
    b = s.remember("project", "Picnic", "quarterly pizza budget review")["id"]  # no match
    s._graph._upsert_edge(a, b, "hebbian", 5.0, "2026-01-01T00:00:00+00:00", mode="max")
    s._conn.commit()
    ids = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert a in ids and b in ids
    assert ids.index(a) < ids.index(b)          # seed dominates the spread-reached node
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_store.py::test_recall_seed_not_buried_by_high_weight_hebbian_neighbor -v`
Expected: FAIL — under the current single-scale sort, `b` (activation 0.032) outranks `a` (seed 0.016), so `ids.index(a) < ids.index(b)` is false.

- [ ] **Step 3: Rewrite `recall` with two-tier ranking**

Replace the method body from the associative-path comment onward. Full new method:

```python
    def recall(self, query, type=None, limit=20, budget=None, session=None) -> list:
        if not query or not query.strip():
            return []
        seeds = self._seed_scores(query, type)
        if not self._graph.enabled:
            if not seeds:
                return []
            order = [mid for mid, _ in sorted(
                seeds.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
            return self._hydrate(order)          # v0.2 shape, no `via`
        # associative path: seed -> prime -> spread -> two-tier rank -> learn -> receipts
        if budget is None:
            budget = self._recall_budget
        sid = self._graph.resolve_session(session, self._meta_get, self._meta_set)
        # `seeds` stays the immutable v0.2 result (the protected tier); prime a copy
        # so priming/spread never reorder the seed tier.
        activated = dict(seeds)
        for mid, a in self._graph.session_activation(sid).items():
            activated[mid] = activated.get(mid, 0.0) + _PRIMING_WEIGHT * a
        if not activated:
            return []
        activation, receipts = self._graph.spread(activated, budget, type)
        # two-tier ranking: v0.2 seeds keep their exact v0.2 order at the top;
        # everything reached by priming/spread fills in below, by activation. A
        # direct hit can never be outranked by a spread-only node (fixes #25).
        tier1 = [mid for mid, _ in sorted(
            seeds.items(), key=lambda kv: (-kv[1], kv[0]))]
        tier2 = [mid for mid, _ in sorted(
            ((m, a) for m, a in activation.items() if m not in seeds),
            key=lambda kv: (-kv[1], kv[0]))]
        order = (tier1 + tier2)[:limit]
        self._graph.touch_session(sid, order)
        self._conn.commit()
        hits = self._hydrate(order)
        for h in hits:
            r = receipts.get(h["id"])
            if r is not None and h["id"] not in seeds:
                h["via"] = {"path": [{"from": r["from"], "kind": r["kind"], "w": r["w"]}],
                            "hops": r["hops"]}
            else:
                h["via"] = {"seed": True}
        return hits
```

- [ ] **Step 4: Run the new test + the existing associative suite to verify pass + no regression**

Run: `pytest tests/test_store.py -k "recall or assoc or via or budget or seed" -v`
Expected: PASS — including `test_recall_disabled_matches_v2`, `test_recall_associative_finds_linked_neighbor`, `test_recall_via_receipts_present`, `test_recall_budget_zero_is_passthrough`, and the new seed-dominance test.

- [ ] **Step 5: Write the failing test — tier-1 keeps v0.2 order (a Hebbian edge on a lower seed must not lift it above a higher seed)**

```python
def test_recall_seed_tier_keeps_v2_order():
    # Two query matches with different relevance; a fat Hebbian edge into the
    # weaker match must not reorder the seed tier.
    pytest.importorskip("numpy")
    s = make_assoc_store()
    strong = s.remember("user", "A", "JWT JWT JWT tokens tokens auth")["id"]   # higher bm25
    weak = s.remember("project", "B", "a passing mention of JWT")["id"]        # lower bm25
    s._graph._upsert_edge(weak, weak + 1000, "hebbian", 5.0,
                          "2026-01-01T00:00:00+00:00", mode="max")  # dangling; no effect on order
    s._conn.commit()
    order = [h["id"] for h in s.recall("JWT tokens", budget=8)]
    assert order.index(strong) < order.index(weak)   # v0.2 relevance order preserved
```

- [ ] **Step 6: Run it to verify it passes (two-tier already guarantees this)**

Run: `pytest tests/test_store.py::test_recall_seed_tier_keeps_v2_order -v`
Expected: PASS — both are seeds (tier 1), ranked by v0.2 score; the dangling edge adds no activation to either. (If the two bodies tie on bm25, strengthen `strong`'s body with more term repetitions until the v0.2 order is deterministic.)

- [ ] **Step 7: Write the failing test — a primed non-seed ranks below the seeds**

```python
def test_recall_primed_nonseed_ranks_below_seeds():
    # priming is an associative boost, not a direct hit: a primed node that is
    # not a v0.2 match must land in the spread tier, below every seed.
    pytest.importorskip("numpy")
    s = make_assoc_store()
    a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]           # seed
    b = s.remember("project", "Picnic", "quarterly pizza budget review")["id"]  # not a match
    # prime b heavily in session 's1'
    s._graph._conn.execute(
        "INSERT INTO session_members(session_id, memory_id, activation, updated_at) "
        "VALUES ('s1', ?, 100.0, '2026-01-01T00:00:00+00:00')", (b,))
    s._conn.commit()
    order = [h["id"] for h in s.recall("JWT tokens", budget=8, session="s1")]
    assert order.index(a) < order.index(b)
```

- [ ] **Step 8: Run it to verify it passes**

Run: `pytest tests/test_store.py::test_recall_primed_nonseed_ranks_below_seeds -v`
Expected: PASS — `b` is not in `seeds`, so despite priming it sorts into tier 2 below `a`.

- [ ] **Step 9: Run the full suite to confirm no regressions**

Run: `pytest -q`
Expected: PASS (same skip count as before — the opt-in real-model test). If `test_recall_budget_zero_is_passthrough` or any assoc test fails, the two-tier order assembly diverged from spec — fix before committing.

- [ ] **Step 10: Commit**

```bash
git add src/tether/store.py tests/test_store.py
git commit -m "fix(recall): two-tier ranking so seeds are never buried by spread (#25)"
```

---

### Task 2: Drop the default recall budget 24 → 8

**Files:**
- Modify: `src/tether/config.py:91` (`_DEFAULT_RECALL_BUDGET`)
- Modify: `src/tether/store.py:138` (the `recall_budget=24` constructor default)
- Test: `tests/test_config.py:100,106,108`

**Interfaces:**
- Consumes: nothing new.
- Produces: `config.recall_budget()` returns 8 when unset/invalid; `Store(...)` defaults `recall_budget=8`.

- [ ] **Step 1: Update the failing test expectations**

In `tests/test_config.py::test_recall_budget_default_and_parsing`, change the three default/fallback assertions from `24` to `8`:

```python
    assert config.recall_budget() == 8      # line 100: unset -> new default
    # line 102 (env "8")  stays == 8
    # line 104 (env "0")  stays == 0
    assert config.recall_budget() == 8      # line 106: invalid string -> default
    assert config.recall_budget() == 8      # line 108: negative -> default
```

(Read the surrounding `monkeypatch.setenv`/`delenv` lines to keep each assertion paired with its case; only the numeric expectation changes.)

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_config.py::test_recall_budget_default_and_parsing -v`
Expected: FAIL — code still returns 24.

- [ ] **Step 3: Change the constant**

In `src/tether/config.py`:

```python
_DEFAULT_RECALL_BUDGET = 8
```

- [ ] **Step 4: Change the constructor default**

In `src/tether/store.py:138`, in the `__init__` signature:

```python
                 decay_half_life_days=None, assoc=False, recall_budget=8,
```

- [ ] **Step 5: Run the config test + full suite**

Run: `pytest tests/test_config.py -v && pytest -q`
Expected: PASS. `make_assoc_store` pins `recall_budget=16` explicitly, so assoc tests are unaffected; `test_server_recall_budget_session` in `test_mcp.py` sets the env explicitly — verify it still passes.

- [ ] **Step 6: Commit**

```bash
git add src/tether/config.py src/tether/store.py tests/test_config.py
git commit -m "fix(recall): drop default recall budget 24 -> 8 (upside peak; #25)"
```

---

### Task 3: Acceptance run, docs, and version bump

**Files:**
- Modify: `README.md` (recall section — one line on seed dominance)
- Modify: `pyproject.toml:7` (version)

**Interfaces:** none.

- [ ] **Step 1: Run the bench acceptance test locally with the real model**

Run: `HF_HUB_OFFLINE=1 python -m bench.run`
Expected: the report prints and the **no-regression guard reads `PASS`** with control distribution `{improved: 0, unchanged: 12, regressed: 0}` (warmed control nDCG == v2 control nDCG, ≈0.866). The learning delta (warmed − cold, graph_only) stays **positive**. Record the printed numbers for the commit message and the journal.

- [ ] **Step 2: If the guard still FAILs, stop and diagnose (do not paper over)**

If any control query regresses, the tier split is wrong — most likely `seeds` was mutated before the tier-1 sort (priming leaking in) or the tier-2 filter (`m not in seeds`) is comparing the wrong set. Re-read Task 1 Step 3 against `store.py` and fix; the guard is the acceptance test, not an optional nicety.

- [ ] **Step 3: Add a one-line note to the README recall section**

Add (adjacent to the existing associative-recall description) exact text:

```markdown
Associative recall is **seed-dominant**: a query's own direct matches always rank
above memories reached only by association, so turning association on never demotes
a hit that keyword/semantic search already found.
```

- [ ] **Step 4: Bump the version**

In `pyproject.toml`, change `version = "0.5.0"` to `version = "0.5.1"` (bug-fix release; no API change).

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: note seed-dominant recall; bump 0.5.1 (#25)"
```

- [ ] **Step 6: Push the branch and open a PR**

```bash
git push -u origin HEAD
gh pr create --base main --title "fix: seed-dominant recall ranking (#25)" --body "<summary + before/after bench numbers>"
```

Include the before/after `bench/` numbers (control 0.107 → ≈0.866, guard FAIL → PASS; graph-only learning delta) in the PR body. Note the guard is now green on `main` once merged.

---

## Self-Review

**Spec coverage:** two-tier ranking (Task 1), provable non-regression (Task 1 tests + Task 3 bench guard), upside preserved (existing `test_recall_associative_finds_linked_neighbor` stays green; graph-only delta positive in Task 3), priming handled (Task 1 Step 7), default budget 24→8 (Task 2), receipts unchanged (Task 1 keeps `via` logic, keyed on `seeds`), docs + version (Task 3). All spec sections map to a task.

**Placeholder scan:** the only free-form field is the PR body in Task 3 Step 6, which references concrete numbers produced in Step 1 — acceptable (values are runtime output, not a design gap).

**Type consistency:** `seeds`/`activated`/`activation` are `{int: float}`; `receipts` is `{int: dict}`; `_upsert_edge(a, b, kind, weight, now_iso, mode=)` matches `graph.py:59`; `session_members` columns match `graph.py:36-42`. The tier-boundary set is `seeds` in both the ranking and the `via` membership test — consistent.
