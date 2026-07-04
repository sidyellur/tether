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
- **One graph primitive powers both.** `Graph.degree_map()` — weighted degree per current node — feeds boot-index ranking (hub = high degree) and forgetting (isolated = zero degree). Weighted degree over PageRank: O(memories + edges), deterministic, no iteration/convergence knob, and it's the exact signal forgetting needs.
- **Behavioral degree, not semantic — the load-bearing decision.** Degree is measured over **behavioral edges only (`explicit` + `hebbian`), never `semantic`.** Semantic edges say "these are *similar*" (automatic, content-derived, and — because Tier A's kNN has no similarity floor — every embedded memory carries ~8 of them). Explicit + Hebbian say "these are *actually used together*" — the retention/importance signal both features are reaching for. Including semantic degree would (1) make forgetting's `degree == 0` gate almost never fire in the default embedder-on config (the "one-off note connected to nothing" would carry 8 semantic edges), and (2) make hub-ranking reward the densest *semantic* neighborhoods — i.e. the most redundant/generic notes — instead of the load-bearing ones. So `degree_map` counts only behavioral kinds. This keeps the change B1-local (no reopening Tier A's edge creation) and makes "isolated = not behaviorally connected" true to intent.
- **Gates and slices, never blended scores.** The bet-2 recency bug came from two different-scaled signals fighting at a tie-break. B1 refuses to blend: the boot-index uses two *labeled slices* (load-bearing ∪ recent), and forgetting uses two *conjunctive gates* (old ∧ behaviorally-isolated). Each piece is independently legible and independently testable.
- **Reuse, don't add.** No new verb, no new module, no schema migration. Hub-curation lives in `boot_index()`; forgetting archives by reusing the supersession columns (`valid_to`/`superseded_by`) and a `meta` counter.
- **Degrade-never.** No Tier A / no behavioral edges ⇒ boot-index falls back to today's unbounded newest-first and forgetting is a hard no-op. Precise byte-identical guarantee: with `TETHER_FORGET` off (default), `recall` and forgetting are byte-identical to pre-B1; the boot-index is byte-identical when `count ≤ CAP` **or** no graph exists, and switches to the curated two-slice format only above `CAP` with a graph present (hub-curation is on by default, independent of `TETHER_FORGET`; set `CAP` very large to disable it).

## Architecture

Follows Tier A's seam: `graph.py` owns *how memories relate*; `store.py` owns *what memories are* and orchestrates.

- **`graph.py`** gains one method:
  - `degree_map(kinds=("explicit", "hebbian")) -> dict[int, float]` — **behavioral** weighted degree for **every** current node, including explicit zeros. Implemented as a LEFT JOIN of current (`valid_to IS NULL`) memories against `edges` filtered to `kinds`, summing incident edge `weight` per node (edges to non-current nodes ignored); a node with no matching edge appears with degree `0.0`. Emitting explicit zeros is deliberate — forgetting needs the degree-`0` *set*, which a pure edge-scan can't produce. `kinds` is a parameter (defaults to behavioral) so a future feature could request semantic-inclusive degree without changing callers. Deterministic; O(memories + edges). Failure → `{}`.
- **`store.py`** consumes it in two places:
  - `boot_index()` — hub-ranked when a graph exists and the store is large (below); unchanged otherwise.
  - `_run_forgetting_sweep()` — the bounded, opt-in archiving pass, invoked amortized from `remember`.

No other files change except `config.py` (new resolvers) and `server.py` is untouched (both features work through existing surfaces — the resource and the store).

## Feature 1 — Hub-curated boot-index

**Behavior by regime:**
- **`count ≤ CAP`** → today's behavior exactly: all current memories, newest-first, unbounded-but-small. (No curation needed.)
- **`count > CAP`, no behavioral hubs** (no `explicit`/`hebbian` edges yet — e.g. a fresh Tier A store with only semantic edges) → the load-bearing slice is empty, so the index is the recent slice alone (bounded newest-first). Honest: "no behavioral hubs yet." Hubs emerge as the store is used.
- **`count > CAP`, behavioral hubs present** → two reserved, labeled slices, capped at `CAP` total:

```
# Load-bearing
[type] #id title      (top nodes by behavioral degree_map, degree desc; ties → updated_at desc, then id; degree-0 nodes excluded)
...
# Recent
[type] #id title      (most recent RECENT_RESERVE memories, updated_at desc)
...
```

- `RECENT_RESERVE ≈ CAP // 4`; the load-bearing slice fills the remaining budget from nodes with **positive** behavioral degree. Union is deduped (a memory that is both a hub and recent appears once, in the load-bearing slice).
- **Why a recent reserve:** a brand-new memory has behavioral degree 0, so a pure hub ranking would bury the thing just saved — usually the most relevant item. The reserve guarantees fresh memories always appear.
- **Why two labeled slices, not one blended `degree + w·recency` score:** avoids scale-mixing (the bet-2 bug class), and the labels are *legibility* — the agent sees why each memory is in its starting context, consistent with Tier A's `via` receipts.
- Per-line format is byte-identical to today (`[type] #id title`); section headers appear only when curation is active.
- `CAP = TETHER_BOOT_INDEX_CAP` (default 50); a very large value disables curation.

## Feature 2 — Forgetting-by-disconnection

**Opt-in** (`TETHER_FORGET`, off by default), like consolidation/dedup/decay.

**Eligibility — two conjunctive gates (both must hold):**
1. **Old** — `age > FORGET_AGE_DAYS` (default 90), measured on `updated_at`. Nothing recently written/refined is eligible.
2. **Behaviorally isolated / unrehearsed** — behavioral `degree_map` value `== 0` (`FORGET_DEGREE_MAX = 0`, fixed): no `explicit` or `hebbian` edge to a current memory. Semantic similarity does **not** count — a memory being *like* others is not a reason to keep it. This single gate carries both "unconnected" and "unrehearsed": Tier A co-recall accretes Hebbian edges (a recall returns a ranked list and `touch_session` wires those co-returned hits together), so a memory actually reached in real recalls gains behavioral degree and is spared, and an explicitly `link()`-ed memory has degree > 0 and is protected intrinsically. (Fable's #16 insight — the Hebbian layer doubles as a retention signal.)

**Known limitation (accepted):** a memory recalled only ever *in isolation* — never co-recalled with another — accretes no Hebbian edge, so it stays degree-0 and, if also old, can fade. This is acceptable because the archive is reversible (a later recall surfaces it, or `valid_to` is cleared) and we refuse to add a durable `last_recalled_at` column (would violate the no-migration constraint). Rehearsal-by-*writing* is fully captured (it refreshes `updated_at`, failing gate 1).

**Archive mechanics — reuse supersession, add nothing:**
- Set `valid_to = now`, leave `superseded_by = NULL`. `recall` and `boot_index` already exclude `valid_to IS NOT NULL`, so archived memories disappear from both for free.
- The pair `(valid_to set, superseded_by NULL)` uniquely means "forgotten" — supersession always fills `superseded_by`. No new column; no ambiguity.
- **Reversible** — clearing `valid_to` un-forgets. No verb built now, but the data supports it.
- **Edges left dormant** — `degree_map`/`_neighbors` already ignore non-current nodes, so an archived memory's edges are inert but retained (so an un-forget restores connectivity). Only a hard `forget` deletes edges (Tier A `on_forget`).

**Safety rails:**
- **Requires a live behavioral graph** — if the store has **zero** `explicit`/`hebbian` edges (Tier A off, or on but never used/linked), the sweep is a hard no-op. Without any behavioral signal every memory looks isolated; refuse rather than mass-archive. (Once *some* behavioral edges exist, the gates + rails below bound the blast radius even while much of an early store is still behaviorally isolated: only *old* memories are eligible, capped per sweep.)
- **Store-size floor** — never runs below `2 × CAP` current memories.
- **Bounded per sweep** — at most `FORGET_MAX_PER_SWEEP` (default 10) archives per run, so even a misconfiguration erodes gradually and visibly.
- **Explicit links protected** — a memory with any `explicit`-kind edge is never archived. Now *intrinsic*: explicit edges are part of behavioral degree, so a linked memory has degree > 0 and fails gate 2 directly (no separate special-case needed).

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

- **Fully hermetic — no embedder, no numpy.** `degree_map` is pure SQL over `edges`; every B1 test hand-inserts edges of specific kinds. Nothing needs Model2Vec or vector math.
- **`degree_map`** — behavioral weighted degree correct; **`semantic` edges excluded** (a node with only semantic edges reports degree 0); **degree-0 current nodes are emitted** (LEFT JOIN, not edge-scan); counts only current (`valid_to IS NULL`) endpoints.
- **Boot-index** — `count ≤ CAP` unchanged; `count > CAP` + behavioral hubs ⇒ two labeled slices, recent memories always present, degree-0 nodes excluded from load-bearing, deterministic order, deduped; `count > CAP` + only-semantic-edges ⇒ recent slice only (empty load-bearing); `count > CAP` + no graph ⇒ today's unbounded behavior.
- **Forgetting gates** — each independently: old-but-behaviorally-connected → kept; behaviorally-isolated-but-recent → kept; **semantic-only-connected + old → archived** (semantic doesn't protect); old + behaviorally-isolated → archived, reversible.
- **Forgetting safety** — disabled → no-op; zero behavioral edges in store → no-op; below size-floor → no-op; `FORGET_MAX_PER_SWEEP` respected; explicit-link protection (intrinsic via degree).
- **Archive mechanics** — `valid_to` set + `superseded_by` NULL; excluded from `recall` and `boot_index`; edges retained; reversibility (clearing `valid_to` restores).
- **Trigger** — counter increments per write; sweep fires at `FORGET_INTERVAL`; counter resets.
- **Degrade guarantee** — `TETHER_FORGET` off ⇒ `recall` and forgetting byte-identical to pre-B1; boot-index byte-identical when `count ≤ CAP` or no graph (curated only above `CAP` with behavioral hubs).

## Coverage matrix (guarantee → test area)

| Guarantee | Test area |
|---|---|
| Behavioral weighted degree (semantic excluded), zeros emitted, current-only | `degree_map` |
| Small store unchanged | boot-index ≤ CAP |
| Large store bounded + load-bearing + fresh; hubs mean "used", not "similar" | boot-index > CAP + behavioral hubs |
| Only-semantic store ⇒ recent-only, not redundancy-filled | boot-index > CAP + semantic-only |
| Non-graph user never truncated | boot-index > CAP + no graph |
| Only old + behaviorally-isolated archived; semantic doesn't protect | forgetting gates |
| No accidental mass-archive | safety rails (zero-behavioral-edges, size-floor, per-sweep cap, links) |
| Archive is reversible & audit-preserving | archive mechanics |
| Amortized trigger fires correctly | trigger |
| Feature-off ⇒ identical to pre-B1 (scoped) | degrade guarantee |

## Known limitations & inherited concerns

- **Solo-recall never protects.** A memory only ever recalled in isolation (never co-returned with another) accretes no Hebbian edge, so it stays behaviorally degree-0 and can fade if old. Narrow in practice — real recalls return a ranked list whose hits get co-wired — and always reversible. We decline a durable `last_recalled_at` column (would break the no-migration constraint).
- **Un-decayed Hebbian skews centrality (inherited from Tier A).** Tier A has no edge decay in scope, so a constantly co-recalled pair saturates its Hebbian weight (capped at `HEBBIAN_CAP = 5.0`) and stays a permanent hub. Bounded by the cap and by relative ranking, but real; the proper fix is edge decay (Fable #6, a later tier). Deferring crystallization to B2 avoids compounding this.
- **`boot_index()` cost shifts from O(memories) to O(memories + edges)** per session load (it's auto-loaded). At ~5k memories × a handful of behavioral edges this is a sub-millisecond scan, but it is a real change from the current pure row-list; noted so the plan can add a cheap ceiling if a giant store ever needs it.
