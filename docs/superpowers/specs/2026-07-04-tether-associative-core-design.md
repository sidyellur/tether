# tether — Associative Core (Tier A) design

> **A usage graph over memories, and spreading-activation recall.** `recall` stops
> returning isolated fragments and starts returning a *connected neighborhood*
> that sharpens with use — while the reasoning stays in the calling agent and
> tether stays 100% local, deterministic, and degrade-never.

## Why

Even with v0.2's hybrid keyword+semantic recall, `recall("X")` returns **fragments
that match your words.** Two things — the whole value of memory — are missing:

1. **Connections.** The most useful memory is often the one that doesn't match the
   query but is *connected* to something that does. Recalling "sqlite-vec" should
   also surface the degrade-never contract that drove the decision and the
   same-class FTS5 bug — memories phrased differently that flat search can't reach.
2. **Learning from use.** Every recall is stateless. The agent works out how a set
   of memories connect, acts, and that connective work is thrown away. The store
   never notices that two memories keep being used together, or that six notes
   circle one principle. It's a filing cabinet, not a mind.

Tier A fixes both **without an LLM inside tether**: a graph whose edges come from
geometry and behavior (not parsed meaning), traversed by spreading activation,
learning from co-recall. The connective payoff of a knowledge graph, minus the
LLM/entity-extraction cost, plus the thing knowledge graphs lack — a graph that
reorganizes itself from how it's used.

## Positioning: a *usage graph*, not a knowledge graph

Knowledge graphs (Zep/Graphiti, Cognee, the MCP KG server) extract entities and
relations from text with an LLM at write time — expensive, drift-prone, and it
violates tether's local/degrade-never ethos. Their edges are *extracted from
content* and static. tether's edges come from three cheap, local sources —
embedding geometry, explicit links, and learned co-recall — and are strengthened
by usage. Same traversal payoff; deterministic; and it learns, which KGs don't.

## Scope decisions (settled during brainstorming)

| Question | Decision |
|---|---|
| Primary axis | Novel **capability**, not raw performance |
| Capabilities in scope | Self-organizing/associative recall, synthesizing recall (via better bundles), reflective structure — **not** proactive/ambient nudging |
| Where intelligence lives | **In the calling agent.** tether supplies structure, signals, and the right bundle; the agent reasons. No LLM in tether, ever. |
| This build's scope | **Tier A — the Associative Core** only. Tiers B (self-organizing store) and C (dreaming, cortex map) are roadmap, not this spec. |
| Tool surface | Enrich `recall`; **no new verbs**. Two optional `recall` params + `via` receipts. |
| Foundation | Build on v0.2 (embeddings, links, hybrid recall). Additive; degrade-never. |

## Architecture

One structural change: a new module, **`graph.py`** (the association layer), so
"how memories relate" is a concern separate from "what memories are."

- **`store.py`** — memories CRUD + hybrid (FTS5 + vector) **seed** retrieval. Unchanged in spirit.
- **`graph.py`** (new) — owns the `edges` + `session_members` tables and their SQL,
  edge maintenance, the session working set, and `spread(seeds, budget, type)`.
- **`recall` orchestrates:** hybrid seeds (existing `store` internals) → `graph.spread`
  → attach receipts → return. `remember`/`link`/`forget`/`migrate` delegate edge
  upkeep to `graph`. A reviewer can understand `graph.py` without reading `store.py`'s
  SQL, and vice versa.

**Degrade-never & cold-start (the spine).** Spreading activation is purely
*additive*. On a fresh store (no edges) or without the `[semantic]` extra,
`graph.spread` contributes nothing and `recall` returns exactly today's hybrid
results. It is never worse than v0.2 — it gets better as edges accrue. Every
mechanism has an explicit passthrough path.

**Determinism.** Everything in Tier A is deterministic given the DB state: the walk
is a fixed traversal (tie-break by activation desc, then id), semantic edges are
precomputed kNN, Hebbian edges are integer counting. (The seeded-RNG "dreaming" is
Tier C, out of scope here.)

## Data model

### `edges` — the usage graph

```sql
CREATE TABLE IF NOT EXISTS edges (
    src        INTEGER NOT NULL,      -- canonical: src < dst (undirected)
    dst        INTEGER NOT NULL,
    kind       TEXT NOT NULL,         -- 'semantic' | 'explicit' | 'hebbian'
    weight     REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
```

Three edge **kinds**:
- **`semantic`** — precomputed kNN from embeddings; `weight` = cosine similarity.
- **`explicit`** — from `link(a,b)`; fixed high weight (`1.0`).
- **`hebbian`** — co-presence within a session; `weight` accrues (capped) with use.

At traversal, a neighbor's transmit weight is a config-weighted blend across the
kinds connecting the pair: `w = α·semantic + β·explicit + γ·hebbian`. Edges touching
superseded (`valid_to` set) or deleted memories are ignored at traversal;
`forget(id)` deletes that id's edge rows (we do **not** rely on SQLite FK cascade,
which is off by default).

### `session_members` — the ephemeral working set

Powers **both** priming and Hebbian learning from one structure.

```sql
CREATE TABLE IF NOT EXISTS session_members (
    session_id  TEXT NOT NULL,
    memory_id   INTEGER NOT NULL,
    activation  REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, memory_id)
);
```

- **Session id**: an optional `session` param on `recall`/`remember`; if absent,
  **time-bucketed** — a gap since the last call beyond `SESSION_GAP` (default 30 min)
  starts a new session. `last_activity` + `current_session` live in the existing
  `meta` table.
- On each call: decay this session's activations (×0.5), bump the touched memories,
  then (a) feed current activations into recall as a **priming** seed, (b) increment
  **Hebbian** edges among the top-M co-active members (small, capped increments).
  Sessions older than `SESSION_TTL` are pruned lazily.

### Reused as-is
`memories.embedding` (semantic edges), `link()`/`links` (explicit edges), and the
entire v0.2 hybrid recall (it produces the seeds the walk starts from).

## Recall flow

Key simplification: **one currency — activation.** Seed scores *are* initial
activation; spreading *adds* activation; final rank is total activation per memory.
No cross-scale blending (the lesson from bet 2's recency-vs-RRF tie-break: different
scales fighting is where bugs live).

```
recall(query, type=None, limit=20, budget=<default>, session=None):
  if not query.strip(): return []
  sid = resolve_session(session)                 # param, else time-bucket

  # 1. SEED — v0.2 hybrid (FTS5 + vector via RRF); each seed's score = initial activation
  activation = hybrid_seed_scores(query, type)   # {memory_id: score}

  # 2. PRIME — add this session's residual activation to the seeds
  for mid, act in session_activation(sid):
      activation[mid] = activation.get(mid, 0.0) + PRIMING_WEIGHT * act
  if not activation: return []

  # 3. SPREAD — bounded activation walk over the graph
  activation, receipts = graph.spread(activation, budget=budget, type=type)

  # 4. RANK — by total activation
  order = top_n(activation, limit)

  # 5. LEARN — decay + bump the session working set; lay Hebbian edges among co-active
  graph.touch_session(sid, order)

  # 6. RETURN — hydrated memories, each with its `via` receipt
  return [ {**hydrate(mid), "via": receipts.get(mid)} for mid in order ]
```

### `graph.spread(seed_activation, budget, type)`
- Frontier = seeds. A node "fires" by transmitting `activation × blended_edge_weight ×
  HOP_DECAY` to each neighbor; propagate only if the transmit exceeds threshold `ε`.
- **`budget`** = max node-expansions the walk may make (Fable's activation-budget dial,
  made predictable). `budget=0` → no spreading → recall == v0.2. Default ~24 → a gentle
  neighborhood on by default; larger → panorama; cost is bounded and predictable.
- **`HOP_DECAY < 1`** keeps seeds dominant — spreading is a *boost*, never an override
  (same discipline as v0.2's gentle recency weight).
- Only current memories (`valid_to IS NULL`) and, if a `type` filter is set, matching
  type are reachable.
- Processing order: (activation desc, id) — deterministic.
- Records, per surfaced node, the strongest incoming path (predecessor, edge kind,
  weight) → **receipts**.
- Returns `(activation_totals, receipts)`.

### Receipts (`via`)
Every returned hit says why it surfaced:
- direct: `{"via": {"seed": "vector", "score": 0.81}}`
- associative: `{"via": {"path": [{"from": 42, "kind": "hebbian", "w": 0.7}], "hops": 1}}`

So the agent can reason about provenance and push back on stale associations —
explain-always alongside degrade-never.

## Edge maintenance

All reuse existing write hooks; no new write path.
- **semantic** — on `remember` (and during the existing `backfill_embeddings` pass):
  compute the memory's top-k (`KNN_K`≈8) nearest current memories by cosine (the
  `_vector_ids` numpy scan we already have) and upsert canonical `semantic` edges,
  keeping the max weight if the pair already exists. No embedder → skipped.
- **explicit** — `link(a,b)` keeps its signature but also upserts an `explicit` edge;
  `migrate()` backfills existing `links` JSON → explicit edges once.
- **hebbian** — in `graph.touch_session`: co-active members get small, capped weight
  increments, bounded to the top-M active, so a big session can't mint a heavy clique.
- **`forget(id)`** — also deletes that id's `edges` and `session_members` rows.

### Known approximation (called out, not hidden)
Semantic kNN is computed at write time against the store as it exists then; a much
later memory that would have been a closer neighbor does not retroactively rewire
old nodes. Acceptable for Tier A. A periodic full rebuild is a later-tier maintenance
pass, not built here.

## Tool surface & config

Minimal, in tether's spirit.
- **`recall`** gains two optional params: **`budget: int`** (breadth dial; `0` = off)
  and **`session: str`** (opt-in id; time-bucketed if omitted). Each returned hit gains
  a **`via`** field. No new verbs.
- **`link` / `remember` / `forget`** — unchanged signatures; edge upkeep is internal.
- **Config** (small; all degrade): **`TETHER_ASSOC`** (master on/off, default on) and
  **`TETHER_RECALL_BUDGET`** (default ~24). Blend weights, `HOP_DECAY`, `KNN_K`,
  `SESSION_GAP`, `SESSION_TTL`, `ε` stay as sensible module constants — resisting a
  knob explosion.

## Error handling — degrade, never throw

Consistent with the existing contract; the worst case is always "exactly v0.2 recall."
- `TETHER_ASSOC=0`, `budget=0`, or an empty graph → spreading is a no-op → recall is
  byte-identical to v0.2.
- No embedder → no semantic edges; Hebbian/explicit still spread if present.
- Session table missing or a session op raising → priming (step 2) and learning
  (step 5) are wrapped in try/except and skipped silently → plain recall.
- A malformed/oversized `budget` is clamped to a safe range.
- Edge maintenance failures during a write never fail the write (embedding-style
  `_or_none` wrapping).

## Testing

The implementation plan will carry a full **Test Strategy** section (standing rule).
Shape:
- **Hermetic + deterministic** with the `FakeEmbedder` (no network, no model download).
- **Unit**: edge maintenance (remember writes kNN edges; link writes an explicit edge;
  forget deletes edges + session rows); spreading over a hand-built tiny graph (a 2-hop
  associate surfaces at `budget≥2`, vanishes at `budget=0`; `HOP_DECAY` keeps a strong
  seed above a weak associate; receipts trace the path); session priming (second recall
  in a session is biased by the first), Hebbian edge creation among co-active members,
  and time-bucket derivation.
- **Degrade paths as first-class tests**: `TETHER_ASSOC=0` → recall byte-identical to
  v0.2; no embedder → no semantic edges but recall works; empty graph → passthrough.
- **Determinism**: fixed `FakeEmbedder` + fixed ordering → stable results.
- **Integration**: `recall` over MCP exposes `budget`/`session` and returns `via`;
  default behavior (no params) is unchanged for existing callers.

## Out of scope (deferred, not dropped)

Later tiers and ideas the brainstorm surfaced, explicitly not built here:
- **Tier B — self-organizing store**: overnight crystallization (cluster → agent-named
  principle hierarchy), hub-based boot-index curation, forgetting-by-disconnection.
- **Tier C — dreaming & cortex map**: replay-based offline consolidation; a visual
  topology resource; multi-agent stigmergy (per-author edge trails).
- **Edge decay / refractory protection**, **periodic semantic-kNN rebuild**,
  **causal/anti/tag-resonance edges**, **path recall**, **lens filters**,
  **negative-space report**, **contradiction detection**, **counterfactual (time-travel)
  recall**, **homeostatic self-tuning**. All layer on the Tier A `edges` + `spread`
  foundation without reworking it.
