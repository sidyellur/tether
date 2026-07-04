# tether Tier B1 — Self-Organizing Store (design)

**Status:** approved design, ready for implementation plan
**Date:** 2026-07-04
**Depends on:** Tier A — Associative Core (`docs/superpowers/specs/2026-07-04-tether-associative-core-design.md`, issue #16). B1 operates over Tier A's `edges` graph and does nothing useful without it.
**Related:** roadmap memory (tether associative-memory ideas and roadmap); Tier A plan (`docs/superpowers/plans/2026-07-04-tether-associative-core.md`).

## Why

After Tier A, tether *has* a usage graph but still treats every memory as equally present forever. Two consequences get worse as a real store grows:

1. **The boot-index scaling wall.** `boot_index()` lists *every* current memory, newest-first, and it's auto-loaded into context each session. Fine at 50 memories; at 5,000 the index itself is context bloat, and the load-bearing memories are lost in the noise.
2. **Nothing ever fades.** A one-off note that was never connected to anything and never looked at again sits in the store forever, competing with everything else.

B1 makes the store *self-organizing*: the boot-index surfaces the load-bearing memories (by graph centrality), and memories that are old, isolated, and unrehearsed can fade (reversibly, opt-in). Both are deterministic, 100% local, and degrade to today's behavior when the graph is absent.

## Scope

**In scope (B1):**
- **Hub-curated boot-index** — rank the auto-loaded index by graph centrality at scale, so it stays bounded and load-bearing.
- **Forgetting-by-disconnection** — a bounded, opt-in maintenance sweep that soft-archives old + isolated + unrehearsed memories.

**Deferred to B2 (separate spec):**
- **Overnight crystallization** — the agent-in-the-loop reflection loop (detect a dense cluster → surface it → the calling agent writes a "principle" memory → tether auto-links it over the sources). Different interaction model (async agent collaboration, a new surface), and the most dependent on observing a real graph. Split out per the same staggered approach that worked for bets 1 and 2.

**Out of scope (this tier entirely):** PageRank centrality, relaxed low-degree forgetting, an explicit un-forget verb.

## Design decisions (the frame)

- **Two features, two execution models.** Hub-curation is *read-time* — `boot_index()` is already recomputed per load, so hub-ranking is just a different ORDER BY; nothing is stored or maintained. Forgetting is the only *mutator*, so it alone uses the amortized-inline trigger.
- **One graph primitive powers both.** `Graph.degree_map()` — weighted degree per current node — feeds boot-index ranking (hub = high degree) and forgetting (isolated = zero degree). Weighted degree over PageRank: O(edges), deterministic, no iteration/convergence knob, and it's the exact signal forgetting needs.
- **Gates and slices, never blended scores.** The bet-2 recency bug came from two different-scaled signals fighting at a tie-break. B1 refuses to blend: the boot-index uses two *labeled slices* (load-bearing ∪ recent), and forgetting uses three *conjunctive gates* (old ∧ isolated ∧ unrehearsed). Each piece is independently legible and independently testable.
- **Reuse, don't add.** No new verb, no new module, no schema migration. Hub-curation lives in `boot_index()`; forgetting archives by reusing the supersession columns (`valid_to`/`superseded_by`) and a `meta` counter.
- **Degrade-never.** No Tier A / no edges ⇒ boot-index is exactly today's unbounded newest-first and forgetting is a hard no-op. `TETHER_FORGET` off (default) + default CAP ⇒ behavior byte-identical to pre-B1.

## Architecture

Follows Tier A's seam: `graph.py` owns *how memories relate*; `store.py` owns *what memories are* and orchestrates.

- **`graph.py`** gains one method:
  - `degree_map(type=None) -> dict[int, float]` — weighted degree for every current node: for each edge between two current (`valid_to IS NULL`) memories, add its `weight` to both endpoints' totals. Type filter optional. Pure read over `edges`; deterministic. Failure → `{}`.
- **`store.py`** consumes it in two places:
  - `boot_index()` — hub-ranked when a graph exists and the store is large (below); unchanged otherwise.
  - `_run_forgetting_sweep()` — the bounded, opt-in archiving pass, invoked amortized from `remember`.

No other files change except `config.py` (new resolvers) and `server.py` is untouched (both features work through existing surfaces — the resource and the store).

## Feature 1 — Hub-curated boot-index

**Behavior by regime:**
- **No Tier A / empty `degree_map`** → today's behavior exactly: all current memories, newest-first, unbounded. Curation is a *benefit* of the graph; a non-graph user's index is never silently truncated.
- **Graph present, `count ≤ CAP`** → also today's behavior (small stores don't need curating).
- **Graph present, `count > CAP`** → two reserved, labeled slices, capped at `CAP` total:

```
# Load-bearing
[type] #id title      (top nodes by degree_map, degree desc; ties → updated_at desc, then id)
...
# Recent
[type] #id title      (most recent RECENT_RESERVE memories, updated_at desc)
...
```

- `RECENT_RESERVE ≈ CAP // 4`; the load-bearing slice fills the remaining budget. Union is deduped (a memory that is both a hub and recent appears once, in the load-bearing slice).
- **Why a recent reserve:** a brand-new memory has degree 0, so a pure hub ranking would bury the thing just saved — usually the most relevant item. The reserve guarantees fresh memories always appear.
- **Why two labeled slices, not one blended `degree + w·recency` score:** avoids scale-mixing (the bet-2 bug class), and the labels are *legibility* — the agent sees why each memory is in its starting context, consistent with Tier A's `via` receipts.
- Per-line format is byte-identical to today (`[type] #id title`); section headers appear only when curation is active.
- `CAP = TETHER_BOOT_INDEX_CAP` (default 50); a very large value disables curation.

## Feature 2 — Forgetting-by-disconnection

**Opt-in** (`TETHER_FORGET`, off by default), like consolidation/dedup/decay.

**Eligibility — two conjunctive gates (both must hold):**
1. **Old** — `age > FORGET_AGE_DAYS` (default 90), measured on `updated_at`. Nothing recently written/refined is eligible.
2. **Isolated / unrehearsed** — `degree_map` value `== 0` (`FORGET_DEGREE_MAX = 0`, fixed): no edge of any kind to a current memory. This single gate carries *both* "unconnected" and "unrehearsed": Tier A co-recall accretes Hebbian edges, so a memory that is actually reached alongside others gains degree and is spared. (Fable's #16 insight — the Hebbian layer doubles as a retention signal.)

**Known limitation (accepted):** a memory recalled only ever *in isolation* — never co-recalled with another — accretes no Hebbian edge, so it stays degree-0 and, if also old, can fade. This is acceptable because the archive is reversible (a later recall surfaces it, or `valid_to` is cleared) and we refuse to add a durable `last_recalled_at` column (would violate the no-migration constraint). Rehearsal-by-*writing* is fully captured (it refreshes `updated_at`, failing gate 1).

**Archive mechanics — reuse supersession, add nothing:**
- Set `valid_to = now`, leave `superseded_by = NULL`. `recall` and `boot_index` already exclude `valid_to IS NOT NULL`, so archived memories disappear from both for free.
- The pair `(valid_to set, superseded_by NULL)` uniquely means "forgotten" — supersession always fills `superseded_by`. No new column; no ambiguity.
- **Reversible** — clearing `valid_to` un-forgets. No verb built now, but the data supports it.
- **Edges left dormant** — `degree_map`/`_neighbors` already ignore non-current nodes, so an archived memory's edges are inert but retained (so an un-forget restores connectivity). Only a hard `forget` deletes edges (Tier A `on_forget`).

**Safety rails:**
- **Requires a graph** — empty `degree_map` (Tier A off / no edges) ⇒ hard no-op. An edgeless store makes everything look isolated; refuse rather than mass-archive.
- **Store-size floor** — never runs below `2 × CAP` current memories.
- **Bounded per sweep** — at most `FORGET_MAX_PER_SWEEP` (default 10) archives per run, so even a misconfiguration erodes gradually and visibly.
- **Explicit links protected** — a memory with any explicit-kind edge is never archived (gate 2 already excludes it, since a link gives degree > 0; restated as a hard rule).

**Trigger (amortized inline):** a `meta` counter (`forget_counter`) increments on each `remember`; when it reaches `FORGET_INTERVAL` (default 20) the bounded sweep runs and the counter resets. No daemon, no new tool.

**Relationship to decay (bet 2):** complementary and independent — decay is a soft *ranking* penalty on old memories; forgetting removes isolated-and-old memories from the candidate set entirely. Both opt-in.

## Configuration

| var | default | effect |
|---|---|---|
| `TETHER_BOOT_INDEX_CAP` | `50` | curate the boot-index above this size; very large ⇒ never curate |
| `TETHER_FORGET` | off (`0/false/no/off`) | master switch for the forgetting sweep |
| `TETHER_FORGET_AGE_DAYS` | `90` | gate 1 — minimum age (days) to be eligible |
| `TETHER_FORGET_INTERVAL` | `20` | writes between amortized sweeps |
| `TETHER_FORGET_MAX_PER_SWEEP` | `10` | hard cap on archives per sweep |

`FORGET_DEGREE_MAX` is fixed at `0` (isolated-only) and not exposed.

## Error handling (degrade-never)

- `degree_map()` wrapped → any failure returns `{}` ⇒ boot-index falls back to newest-first, forgetting no-ops.
- `boot_index()` curation wrapped → any failure returns today's full newest-first list.
- `_run_forgetting_sweep()` wrapped → any failure skips the sweep and never blocks the `remember` that triggered it; archiving commits in its own step so a mid-sweep error can't corrupt the write.
- **No schema migration** — reuses `valid_to`/`superseded_by` and `meta`; new meta keys written idempotently. A v0.4 DB upgrades with zero DDL.

## Testing

Full detail belongs in the plan's Test Strategy section; the shape:

- **Fully hermetic — no embedder, no numpy.** `degree_map` is pure SQL over `edges`; every B1 test hand-inserts edges. Nothing needs Model2Vec or vector math.
- **`degree_map`** — weighted degree correct; counts only current (`valid_to IS NULL`) nodes; empty on no edges.
- **Boot-index** — `count ≤ CAP` unchanged; `count > CAP` + graph ⇒ two labeled slices, recent memories always present, deterministic order, deduped; `count > CAP` + no graph ⇒ today's unbounded behavior.
- **Forgetting gates** — each independently: old-but-connected → kept; isolated-but-recent → kept; old + isolated → archived. Plus the known-limitation case: solo-recalled (degree-0) + old → archived, then reversible.
- **Forgetting safety** — disabled → no-op; no-graph → no-op; below size-floor → no-op; `FORGET_MAX_PER_SWEEP` respected; explicit-link protection.
- **Archive mechanics** — `valid_to` set + `superseded_by` NULL; excluded from `recall` and `boot_index`; edges retained; reversibility (clearing `valid_to` restores).
- **Trigger** — counter increments per write; sweep fires at `FORGET_INTERVAL`; counter resets.
- **Degrade guarantee** — `TETHER_FORGET` off + default CAP ⇒ boot-index and recall byte-identical to pre-B1.

## Coverage matrix (guarantee → test area)

| Guarantee | Test area |
|---|---|
| Weighted degree, current-only | `degree_map` |
| Small store unchanged | boot-index ≤ CAP |
| Large store bounded + load-bearing + fresh | boot-index > CAP + graph |
| Non-graph user never truncated | boot-index > CAP + no graph |
| Only old+isolated archived (isolation carries "unrehearsed") | forgetting gates |
| No accidental mass-archive | safety rails (no-graph, size-floor, per-sweep cap, links) |
| Archive is reversible & audit-preserving | archive mechanics |
| Amortized trigger fires correctly | trigger |
| Feature-off ⇒ identical to pre-B1 | degrade guarantee |
