# tether v0.1 — Blog Journal

## The core insight: memory as a shared substrate

The problem tether solves: personal agents will live across devices (laptop, desktop, phone). For that to feel like *one* assistant, not several amnesiac ones, memory has to be a substrate that follows you — readable and writable from every device and any agent, not siloed in a single tool. This isn't about perfect sync or guarantees; it's about being a *convenience layer* that helps when present and never breaks the agent when degraded.

## Design choice: local-first + opt-in sync

We chose to ship a local SQLite file as the default — zero config, no accounts, no network. The alternative (Turso/libSQL primary with local replicas) only works if sync is reliable. By making it opt-in, we sidestep the vendor risk and deployment complexity. Agents that don't need cross-device memory use it as-is. Agents that do can point at a Turso primary and get an embedded replica. The same code path handles both.

## Real-world spike: .sync() doesn't fail fast

During Task 7, we discovered libsql-experimental's `.sync()` doesn't fail fast on unreachable backends — it retries the handshake internally (observed every ~2-3s) without returning control. The plan assumed we could call `conn.sync()` inline as an initial connectivity probe. We corrected this by bounding the initial sync in a background thread + timeout, the same pattern we already use for later syncs. This change is subtle but critical: it keeps server startup from hanging indefinitely on a dead network.

## The upsert-on-write design

The store deduplicates on write, not on read. A memory with the same `type` + normalized `title` updates in place. This was a deliberate choice: agents shouldn't need to know "is this fact new or existing?" before calling remember. Re-remembering a fact refines it. The cost is a dedup index on (type, title_norm), but the benefit is that the store doesn't rot into a heap of near-duplicates.

## Four verbs, nothing more

We resisted the urge to add `list`, `get`, or raw SQL. The four verbs (remember, recall, link, forget) are the minimum surface needed for an agent to use memory. A fifth verb (the boot-index resource) is auto-loaded each session, so agents don't need to think about it. This minimalism makes the tool usable across contexts and easy to reason about.

## Deferred, not dropped

The spec laid out three things we didn't build: semantic search (embeddings), entity/edge graph model, and automatic corrupt-DB recovery. We designed the schema so all three slot in without data migration. The important part: we called out what's deferred in the docs, not silently dropped it. v0.1 works; v0.2's roadmap is visible.
