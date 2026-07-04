# Tier B2 — Crystallization (design)

**Status:** design, pending review
**Depends on:** Tier A (Associative Core, `graph.py`, shipped) and Tier B1
(hub-curated boot-index + `degree_map`, shipped). Not blocked by the parked
usage-learning work — see Foundation.
**Related:** memory #6 (Fable idea #13), the B1 spec (which deferred this), the
bench held-out finding (PR #31) that shaped the Foundation choice.

## Goal

Let tether **reflect**: detect dense clusters of related memories, surface them
to the calling agent as "this cluster may want a name," let the agent write a
**principle** memory, and auto-link that principle over its sources. Math finds
the structure; the agent supplies the words. The result is an emergent,
agent-curated hierarchy of principles on top of raw memories — the
"synthesizing / reflective consolidation" capability, 100% local, deterministic,
no LLM inside tether.

## Non-goals (this tier)

- No LLM inside tether. Naming is done by the *calling* agent, not tether.
- No background daemon or scheduler. "Overnight" = the agent reads a resource
  during a reflection pass; detection is pull-only (see Interaction).
- No principles-of-principles engine, no automatic re-clustering of principles
  (the peak-seeding exclusion below deliberately prevents recursion this tier).
- No new recall *ranking* logic beyond adding one edge kind to spreading, and
  that step is gated by the no-regression bench guard.

## Foundation — cluster on **semantic + explicit**, boost with behavioral

A cluster is **defined** by semantic (topical) + explicit (`link()`) structure;
behavioral (hebbian) edges fold in as a **density booster** whose weight starts
at **0** and is turned up only once the eval validates the behavioral signal.

**Why semantic here, when B1 deliberately excluded it.** B1's "similarity ≠
importance" answered a *retention* question ("what is load-bearing / what may
fade?"), where similarity is genuinely the wrong signal. Crystallization answers
an *abstraction* question ("what conceptual generalization is worth naming?").
For abstraction, semantic similarity is the *correct* primitive — a principle is
by definition a generalization over similar things. The two tiers use opposite
edge philosophies **on purpose**, because they ask opposite questions. This is
deliberate, not drift.

**Semantic is membership; explicit + behavioral are the peaks.** Semantic
density is nearly uniform — every embedded memory has ~8 kNN neighbors, so on
semantic edges alone the whole store looks equally connected and "dense cluster"
is meaningless. Explicit links (a human's "these belong together" vote) and
behavioral co-recall are what rise *above* that uniform floor. So:

- **peaks** (what nucleates a cluster) = explicit + behavioral edges;
- **membership** (what fills it out) = the semantic neighborhood around a peak.

## Detection — seed-from-peak, expand-by-semantic (NOT Louvain)

1. **Score peaks.** Each `explicit`/`hebbian` edge gets a *boundness* score:
   `explicit → W_e`, `hebbian → w · W_b` with **`W_b = 0` at launch**. Semantic
   edges score 0 for peaks. `crystallized` edges are **excluded** from peak
   scoring (loop correctness — see Interaction).
2. **Seed.** Candidate nuclei = connected components of edges whose boundness
   clears a floor. Union-find with canonical ordering → deterministic.
3. **Expand.** Grow each nucleus along **semantic** edges, **bounded by a cosine
   threshold + a hop cap** — this boundary is where membership gets its
   precision; without it a peak expands straight across the uniformly-connected
   store.
4. **Emit** each cluster (size ≥ `MIN_CLUSTER`) as a candidate:
   `{cluster_id (ephemeral display handle only), member_ids, member_titles,
   why (the seed edges), descriptor}`.
5. **Dedup** at read time by **source-set overlap** against existing crystallized
   principles (suppress if overlap ≥ `DEDUP_OVERLAP`). Robust to membership
   drift; needs no stored cluster identity (see Interaction adjustment A).

**Why seed-from-peak and not global weighted community detection (Louvain):**
- We want **nuclei + a large unclustered remainder**, not an exhaustive
  partition that forces every memory into a bucket.
- Principles **overlap**; hard partitions give each node one community.
- The **uniform-semantic-floor breaks global modularity** — the sparse
  meaningful edges wash out; seed-from-peak starts *from* those sparse edges so
  they drive the result.
- **Determinism + incrementality.** Louvain is stochastic (random init,
  order-dependent, unstable community ids); seed-from-peak is deterministic and
  local. **Louvain is the crystallization analog of PageRank**, which B1 already
  rejected for hub-ranking to avoid a stochastic convergence knob — this applies
  the same committed principle.

### Cold-start fallback — relative semantic-density outlier (gated)

Pure explicit/behavioral seeding has a real blind spot: the highest-value
crystallization is *"five notes you never linked that cohere"* — the connection
the user didn't notice. Explicit-only seeding structurally cannot find it (no
links = no peak), and that non-obvious insight is what justifies the feature
over plain RAG. So we also seed from **unusually tight semantic neighborhoods**,
but **never via a magic number on a blended scale** (won't survive an embedder
change). A semantic neighborhood seeds a candidate only if its internal density
is an **outlier against the store's own baseline** (top percentile, or
`mean + k·σ` of pairwise cosine). Self-calibrating across stores and models.

**Because crystallization is a batch reflection pass, not a live interrupt, this
can lean permissive** — a dud in an overnight pass is cheap (the agent triages
it, naming rejects it); the agent-naming gate *is* the precision filter and costs
nothing offline. The fallback ships **off-or-conservative at launch**, and the
eval moves the bar.

## Interaction — the async agent-collaboration loop

1. **Surface — a pull-only MCP resource** `tether://crystallization`. Candidates
   are computed **on read** (no daemon; "overnight" = the agent opens it during
   reflection). **It must NOT ride the auto-loaded boot-index path** — the
   "no-daemon is free" argument holds only if detection is rare (adjustment B).
   The read-time compute is **process-memoized, invalidated by a write
   dirty-flag**, so repeated reads in one pass don't recompute (adjustment C).
   Detection is a **derived view exactly like B1 hub-curation** (computes, does
   not mutate) → read-time by B1's own precedent, not a maintained table.
2. **Name — the calling agent** reads the candidates, discards topics, and for
   each real principle writes it back.
3. **Write-back — extend `remember(crystallizes=[source_ids])`.** With derived
   dedup (adjustment A) there is **no crystallization-specific state to write** —
   this collapses to "create memory + add typed links," which is inside
   remember's existing meaning (`crystallizes` is `links` wearing a kind +
   provenance). No 5th verb; no feature-specific side effect on the general verb.
4. **Link kind — a new `crystallized` edge** (principle → each source).

### Adjustment A — dedup identity

Do **not** anchor identity/dedup to a member-set hash: membership drifts (add a
nearby memory → semantic expansion pulls it in → hash changes → the principle
re-surfaces). Derive "already covered" at read time from **crystallized-edge
source-set overlap**. `cluster_id` is an ephemeral display handle only.

### The `crystallized` edge-kind role matrix

| role | `crystallized` | why |
|---|---|---|
| B2 peak-seeding | **excluded** | else a principle re-seeds the cluster it just named — principles-of-principles runaway |
| B1 `degree_map` / hubs | included | a named principle becomes a boot-index hub |
| B1 forgetting | included (via degree) | a principle protects its sources from disconnection-forgetting |
| Tier A recall spreading (`KIND_W`) | **included, bench-guarded** | recalling a source surfaces its principle — the retrieval payoff — but must not regress control recall |

Reusing `explicit` could not separate peak-seeding-out from hub-degree-in
(`explicit` seeds peaks), so the distinct kind is a **loop-correctness**
requirement, not a nicety. The kind also carries **provenance** (agent-named vs
human-linked vs machine-kNN) for the resource's `why` receipts and for eval.

### The third importance axis (honest note)

Making principles hubs via manufactured edges introduces a **third** importance
source — **named** — alongside **used** (hebbian) and **linked** (explicit). It
is defensible: an agent judging something principle-worthy is a strong
importance signal. But it is no longer strictly "importance = use," and we say so
plainly rather than let it read as drift from B1's thesis.

## Degrade-never / opt-in

- Off by default: `TETHER_CRYSTALLIZE` (like `TETHER_CONSOLIDATE` /
  `TETHER_FORGET`). When off, the resource returns empty and `crystallizes=` on
  remember is ignored (falls back to plain links or no-op).
- No embedder → no semantic membership/fallback; detection degrades to
  peak-edge components only. No graph → empty. Never raises.
- Adding `crystallized` to recall spreading is the only recall-touching change;
  it ships behind the bench no-regression guard, and with `TETHER_ASSOC=0` /
  `budget=0` recall stays byte-identical to v0.2 as before.

## Config knobs (all eval-tuned; defaults conservative)

| var | meaning | launch default |
|---|---|---|
| `TETHER_CRYSTALLIZE` | master opt-in | off |
| `W_e` / `W_b` | explicit / behavioral peak weight | `W_e > 0`, **`W_b = 0`** |
| expansion cosine floor + hop cap | membership boundary | conservative (tight) |
| `MIN_CLUSTER` | min members to emit | 3 (tunable) |
| `DEDUP_OVERLAP` | source-overlap fraction to suppress | conservative |
| semantic-density fallback | percentile / `mean+k·σ`; on/off | **off-or-conservative** |

## Eval plan (bench-driven, the project's discipline)

The design is deliberately measurable:
- **Semantic-only-seeded candidate acceptance rate.** When the fallback surfaces
  a semantic-only cluster, does the agent-naming step accept it (principle → real
  value → loosen the bar) or reject it (topic bucket → noise → keep it tight)?
  Low reject → lower the bar; high reject → keep it tight. Measured, not guessed.
- **`W_b` unlock.** Behavioral booster stays at 0 until the eval shows
  behavioral-seeded clusters raise acceptance without adding noise.
- **Recall no-regression.** Adding `crystallized` to spreading must keep the
  control class green (the same guard that gates all recall changes).

## Test Strategy

- **Hermetic detection tests** (no embedder, no numpy): hand-insert `edges` of
  each kind; assert seed-from-peak finds the peak component, semantic-expand
  respects the cosine/hop bound, `crystallized` is excluded from peak-seeding but
  present in `degree_map`, dedup suppresses on overlap, `MIN_CLUSTER` filters.
- **Resource tests:** pull-only (never computed on boot-index load), process-memo
  hit/invalidate on write, empty when `TETHER_CRYSTALLIZE` off / no graph.
- **`remember(crystallizes=…)`:** creates the memory + `crystallized` edges to
  each source; off-flag ignores it; degrade-never on bad ids.
- **Recall spreading:** `crystallized` edge lets a source-query reach its
  principle; the bench no-regression guard stays PASS with crystallized edges
  present (`FakeEmbedder` e2e).
- **Config resolvers:** defaults + parsing for every knob above.

## Open items for the plan (not open decisions)

Exact values for the expansion cosine floor, hop cap, `DEDUP_OVERLAP`,
`MIN_CLUSTER`, and the fallback percentile are eval-tuned constants the plan
pins with bench evidence; they are knobs, not design forks.
