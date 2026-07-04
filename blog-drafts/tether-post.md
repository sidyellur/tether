# tether v0.2: finding the durable position, and the bugs the plan couldn't see

*A shared, local-first memory layer for personal agents — and what the second release taught me about strategy, degrade-never engineering, and why every plan needs a test strategy.*

## What tether is

tether is an [MCP](https://modelcontextprotocol.io) server backed by a local SQLite file. Any MCP-compatible agent can `remember`, `recall`, `link`, and `forget` durable notes — facts about you, your projects, your preferences — so context follows you instead of dying with each session.

The bet behind it: the near future is personal agents living across many devices — laptop, desktop, phone. For that to feel like *one* assistant rather than several amnesiac ones, memory has to be a substrate that follows you, readable and writable from every device and any agent, not siloed inside a single tool. tether is deliberately a *convenience layer*: it makes an agent more useful when present, and it never breaks the agent's work when degraded.

v0.1 established the shape. Four verbs and nothing more (I resisted list/get/raw-SQL — `remember`/`recall`/`link`/`forget` is the minimum usable surface). Upsert-on-write, so the store dedups as you go rather than rotting into near-duplicates: same `type` + normalized `title` refines in place, and agents never have to decide "new or existing?" before writing. A local SQLite file as the zero-config default, with opt-in libSQL/Turso sync sharing the exact same code path. And an auto-loaded boot index — a compact one-line-per-memory list surfaced each session, a kind of silent fifth verb — so memory helps even when the agent doesn't think to search.

v0.1 also taught me that "it builds and tests pass" ≠ "it ships." Two sharp edges surfaced in one afternoon: PyPI blocks distribution names that collide with well-known brands (Tether/USDT), so the distribution became `tether-memory` while import/CLI/repo stayed `tether`; and a GitHub Actions release workflow that set `permissions: id-token: write` and nothing else silently dropped `contents: read`, because specifying *any* permission flips the rest of the block to `none`. Neither showed up in local testing. Both were found only by reading the actual CI logs.

That's the backdrop. This post is about v0.2 — where the interesting decisions were.

## The pincer: why I stopped thinking of tether as a first mover

Before writing any v0.2 code, I ran deep product research — three parallel agents on competitive landscape, technical direction, and go-to-market. It reframed the strategy hard.

tether v0.1 is **not** sitting in an empty category. `ai-memory-mcp` already ships a near-identical pitch: SQLite FTS5, zero cloud dependencies, cross-MCP-client, sync. That's the clone risk from below. And from above, the platforms are commoditizing *personal* memory — Anthropic shipped standardized memory import under a "context should belong to the user" banner. So the position isn't "open field"; it's a **pincer**: OSS clones below, platform features above.

What survives that squeeze is the part neither side can own: **user- and team-owned, neutral, local-first memory.** The moat isn't the primitive (everyone has FTS5 over SQLite) — it's trust, data locality, and team-sharing. The business analogues here are Tailscale and Obsidian: individuals are the funnel, teams are the revenue, and sync is the one thing free local users actually pay for.

That reshaped the roadmap into prioritized bets rather than a feature list. Bet 1: semantic search. Bet 2: consolidation. Both are non-negotiable substrate whether the future is solo or team. Then, later, team shared-memory as the actual business. And an explicit *non*-goal: no full knowledge-graph engine — the field has converged on hybrid retrieval, and even Mem0 is retreating from external graphs. Naming what I'm *not* building was as important as naming what I am.

## Bet 1 — semantic search without the C extension

The obvious way to add vector search to SQLite is the `sqlite-vec` C extension. I didn't use it, and the reason is the whole ballgame for a tool whose contract is *degrade-never*.

**SQLite extension loading is disabled in many system Python builds** — notably macOS system Python. If tether's semantic search depended on loading a native extension, it would simply fail to import on a large fraction of the machines it's supposed to run on. That directly violates the promise that memory helps when present and never breaks the agent when degraded.

So instead of a native index, semantic search is an `embedding BLOB` column plus **in-process numpy brute-force cosine**. This is the same brute force the research endorsed — under 10ms at anywhere from hundreds to ten thousand vectors, which is the entire realistic range for a personal memory store — but with none of the deployment fragility. It also keeps the test suite hermetic (no native build step) and it matches v0.1's own schema note, which had already reserved space for "an embedding BLOB column that backfills existing rows." The v0.1 schema was designed so this slotted in without a data migration, and it did.

The embeddings themselves come from a **local static model** (Model2Vec). Static is the key word: no neural forward pass, no network call on the hot path — embedding a query is a lookup-and-average, not an inference. FTS5 keyword search stays the always-available floor; semantic is purely additive, and the two are fused with **Reciprocal Rank Fusion** so a query finds relevant memories even when the exact words differ ("automobile" recalls a note about your "car"). Without the optional `[semantic]` extra, or with `TETHER_SEMANTIC=0`, tether is keyword-only FTS5 and nothing hangs.

Bet 1 shipped as PR #13 (merged `5950eaf`), essentially verbatim from the plan: `embed.py` with a Model2Vec `Embedder` producing unit-normalized vectors, the `embedding` BLOB column plus a `meta` table, `backfill_embeddings()` with a model-change reset, hybrid `recall` via RRF, and degrade-never behavior on missing numpy, a broken embedder, or no embedder at all. Vectors are stored as little-endian float32 via `struct`, and because they're unit-normalized, similarity is just a numpy dot product (which equals cosine).

One detail I'm keeping as a lesson: the plan carried complete code, and it flagged exactly one unverified external API — `self._model.encode([text])[0]`, the Model2Vec call it couldn't test without the dependency present. It shipped exactly like that, no correction needed. A detailed plan turns the remote build into a near-transcription, and the *pre-identified* risk cost nothing precisely because it was named up front. Hold that thought — bet 2 is the counterexample.

## Staggering the two bets (they collide in store.py)

Both bets ran as separate remote sessions, but they couldn't run fully in parallel. They both touch `store.py`'s `migrate`, `remember`, and `recall` — the busiest functions in the codebase. Run them concurrently and you get two divergent rewrites of the same three functions to reconcile by hand.

So I staggered them: bet 1 lands on `main` first, then bet 2 branches off the *updated* `main`, and bet 2's plan opens with a hard "Depends on Bet 1" gate. Consolidation actually *wants* bet 1's embeddings anyway (near-duplicate detection reuses them), so the dependency was real, not just a merge-conflict dodge.

## Dogfooding, and the remote-session memory gap

Somewhere in here I noticed that the thing I was doing to *write this post* was tether's exact use case. The blog journal — decisions, bugs, the first-publish gotchas, the v0.2 brainstorming — is durable, cross-session project knowledge. So it shouldn't live in a repo-committed `NOTES-FOR-BLOG.md`. I moved it into tether itself: one upserted memory per project, recalled-appended-remembered as new work lands (`44e41de` retired the file). The blog is now a *byproduct* of good memory hygiene, not a separate system. And the boot-index noise that accumulates as the journal grows is exactly what bet 2's recency-weighting and consolidation are for — a satisfying closed loop.

But dogfooding immediately exposed a real hole: **remote/cloud sessions can't journal into tether.** Two reasons. First, the "journal into tether" instruction lives in my user-level `~/.claude/CLAUDE.md`, which doesn't travel to a remote environment. Second, the tether MCP and its local SQLite DB are laptop-local, and sync isn't configured — so a remote session has nothing to read or write. And I'd just retired `NOTES-FOR-BLOG.md`, which had been the *one* channel a remote session could use.

The stopgap is a **local-harvest** pattern: remote sessions write rich PR descriptions and commit messages, and a local session — where tether is reachable — harvests merged PRs into the journal. (The bet-1 journal entry was itself backfilled from PR #13 this way.) The *proper* fix is Turso sync plus tether registered in the remote environment, which would make remote sessions first-class memory participants. In other words, the gap is itself a concrete argument for prioritizing the sync feature that the GTM research already said is the thing people pay for.

## Bet 2 — consolidate by marking invalid, never overwriting

The design principle for consolidation is: **never destroy history.** Supersession is temporal — `valid_from` / `valid_to` / `superseded_by` — and superseded rows are *retained* as an audit trail, excluded from recall and the boot index. Only `forget` actually deletes. On top of that: gentle recency-weighting on recall (deliberately small — enough to break ties, not enough to override a strong semantic match), opt-in near-duplicate consolidation using bet 1's embeddings, and opt-in time-decay. Safe defaults, powerful opt-ins; everything off by default.

I also added an `author` column *now*, before there was any multi-writer feature to use it. That's deliberate: when the team tier arrives, I want it to be an ACL wrapper over existing data, not a schema migration on everybody's store. Build the groundwork while the table is small.

Bet 2 shipped as PR #14 (merged `dee864b`): 59 tests passing, 2 intentionally skipped (the opt-in real-model tests). The recency-vs-RRF composition I'd fretted about in the plan landed exactly as written (`_rrf_scores` plus a `_RECENCY_WEIGHT * s` term).

And then TDD found two bugs the plan's literal code did not predict — both living in seams a plan structurally cannot see.

### Bug 1: FTS5 shadow-index corruption from migration ordering

The new `valid_from` healing `UPDATE` in `migrate()` fired *before* the FTS5 rebuild, on a table that had just been created over pre-existing rows. That `UPDATE` touched `memories` while its FTS5 shadow index was in an inconsistent state, and corrupted it.

The fix was to reorder `migrate()` so the FTS rebuild always runs before any `UPDATE` touches `memories`. The lesson generalizes: in a migration that mixes DDL, healing `UPDATE`s, and FTS triggers, order-of-operations is load-bearing. (It's the same class of bug as a v0.1 lesson about FTS internals — the shadow index does not tolerate being written through in the wrong sequence.)

### Bug 2: the recency tie-break that got swamped by SQLite's scan order

I *had* flagged recency as delicate around fusion — I worried it wouldn't compose cleanly with RRF. My instinct pointed at the right area, but the actual bug lived a layer below where I'd put it.

`_fts_ids` ordered results purely by bm25 `rank`. When two rows are *identically* relevant, bm25 gives them the same rank, and SQLite then breaks the tie by its own arbitrary scan order. That arbitrary order translated into a full RRF rank-step — roughly 1/61 — separating the two rows. And that step *swamped* the intended 0.25 gentle-recency weight. The recency weight was never the problem; the tie order was decided at the SQL layer, before recency could ever apply.

The fix was one line in the right place: add `updated_at DESC` as a secondary `ORDER BY` in `_fts_ids`, so genuine bm25 ties resolve by recency at the SQL layer instead of by scan order. (I also caught a stale `remember` docstring that still claimed `action` was only `created`/`updated`, never `consolidated`.)

## The takeaway: a plan buys transcription; TDD buys the seams

Put bet 1 and bet 2 side by side and you get a clean statement of what detailed planning does and doesn't do.

A detailed plan with complete code buys you a *faithful transcription* — bet 1 shipped verbatim, and even its single flagged external-API risk cost nothing because it was named in advance. But the bugs that actually matter live in the **seams** a plan can't see: migration order-of-operations, an SQL-layer tie-break interacting with a scoring weight two layers up. No amount of per-task detail predicts those, because they only exist at the integration boundaries *between* the tasks. TDD is what surfaces them.

That's why the standing rule out of this cycle is: **every implementation plan must include an explicit Test Strategy section** — levels, hermeticity controls, a coverage matrix, and deliberate non-goals. Per-task TDD hides cross-cutting gaps (integration seams, real-data migration, partial-degrade coverage). Bet 2 is the concrete vindication: the plan was right about the code and wrong about the seams, and the test strategy is what caught the difference.

## Where this leaves tether

v0.2 gave tether hybrid keyword-plus-semantic recall that degrades to keyword-only on any machine, temporal consolidation that never loses history, and the schema groundwork (`author`) for the team tier that the product research says is the actual business. The remaining gap — remote sessions can't reach local memory — isn't a loose end so much as a signpost: it points straight at sync, which is both the missing dogfooding channel and the one feature free local users are willing to pay for.

The primitive was never the moat. User- and team-owned, local-first, degrade-never memory is.
