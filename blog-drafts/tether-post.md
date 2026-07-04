# Tether v0.1: A Shared Memory Layer for Personal Agents

We shipped [tether](https://github.com/sidyellur/tether) yesterday — an MCP server that turns a local SQLite file into a durable memory substrate for personal agents across devices. If you're building agents that span a laptop, desktop, and phone, this is the post for you.

## The problem is amnesia at scale

Picture this: you're using Claude on your laptop, and it learns something important about your project — a decision you made, a constraint you mentioned, a preference you established. You close the tab. Three hours later, on your phone, you ask the same agent a question. It has no idea what you told it before. So you explain again. And again. Each device is a fresh start, and you're burning tokens re-establishing context that already exists.

This matters because personal agents are *mobile* by definition. If an agent is useful, you'll want to talk to it from wherever you are. But context doesn't follow you unless memory does — and not the kind locked in vector stores or fine-tuning checkpoints. You need a *durable, writable, cross-device memory substrate* that any agent can read and update from any device, any time.

That's what tether solves.

## The core insight: memory as shared ground

The key design decision isn't technical — it's architectural. Instead of building a memory *system* (embeddings, entity graphs, reasoning layers), tether is a *substrate*. Think of it like a filesystem: not everyone needs to understand ext4 to use a laptop, but a common storage layer makes it possible for different tools to share files.

In our case, different agents on different devices share the same SQLite database. Your Claude session on the laptop can call `remember()` to save a fact. Your Claude session on the phone calls `recall()` to find it. A third-party agent in the middle can link memories together or refine them. None of them need to coordinate — they're all reading and writing the same shared ground.

This isn't about perfect consistency or strong guarantees. It's about being a *convenience layer* that helps when present and doesn't break the agent when degraded. An agent using tether should never feel worse off for having it than without it.

## Design choice: local-first, sync optional

Here's where we got the most pushback during design: why local-first at all? Why not just build against a cloud primary from day one?

The answer is deployment friction. A truly cross-device sync system requires a reliable backend, proper authentication, and careful handling of offline states. That's a lot of operational complexity for a v0.1. More importantly, many agents don't *need* cross-device sync — they're happy running on a single machine. Forcing them through that complexity would be a mistake.

So we inverted it: local SQLite is the default. Zero config. No accounts. No network. You install tether, point Claude Code at it, and memory just works. An agent that wants to stay local never thinks about sync.

For agents that do need to follow you across devices, you point tether at a Turso/libSQL primary and it becomes an embedded replica. The exact same code path handles both cases — the difference is a single environment variable. Write a memory on your laptop, and it propagates to your phone within seconds. Go offline, and the agent keeps working against the replica; changes converge when you reconnect.

This design sidesteps vendor risk (you're not forced into a SaaS dependency) and keeps the surface simple. It also means we can iterate on sync without breaking single-machine deployments.

## The hidden cost of `.sync()`: a real-world lesson

During implementation, we hit a snag that wouldn't have shown up in unit tests or design docs.

The plan was straightforward: when tether starts up and a backend is configured, call `conn.sync()` on libsql-experimental to probe connectivity. If the handshake fails, log it and move on. This was meant to be cheap and fast — a way to know whether the server is online before we commit to any behavior.

What we actually found: `conn.sync()` doesn't fail fast. When the backend is unreachable, libSQL retries the handshake internally and retries again every ~2-3 seconds. This is good for resilience in the common case, but it means calling `sync()` inline will hang your startup if the backend is down.

So we redesigned: the initial sync now runs in a background thread with a timeout. It's the same pattern we already use for periodic syncs later in the server lifecycle, just moved to startup. The change is subtle — three lines of code — but critical. Without it, tether's startup blocks indefinitely on a dead network, which defeats the purpose of having local-first defaults.

This is the kind of thing that unit tests don't catch because the tests run on localhost. It's worth noting because it illustrates a principle: *be careful about libraries that abstract away I/O and timeouts*. They're usually trying to be helpful, but they can surprise you when your assumptions about control flow don't match the implementation.

## Four verbs: deliberate minimalism

The tether API is four functions.

- `remember(type, title, body, tags?, links?)` — save a memory, upserting on type+title
- `recall(query, type?, limit?)` — keyword search
- `link(id_a, id_b)` — bidirectional link
- `forget(id)` — delete a memory

That's it. No `get()`, no `list()`, no raw SQL. No builder patterns or state machines. Just four verbs that do one thing each.

This restraint is intentional. We resisted adding a read-first `get()` method because agents shouldn't need to know "is this fact new or existing?" before calling remember. In tether, re-remembering a fact refines it in place. We deduplicate on write, not on read — the store is indexed on (type, title_norm), and a new fact with the same normalized title updates the existing row.

The upsert-on-write model is more powerful than it looks. It means agents can be optimistic: just remember the fact, and the system handles deduplication. Over time, a given memory might be updated a hundred times as an agent learns more about it, but the store stays clean instead of rotting into a heap of near-duplicates.

We also ship an auto-loaded resource, `tether://memory-index`, that surfaces a compact one-line-per-memory index at the start of each session. Agents don't need to think about searching — the index is already there, ready to jog their memory or seed a conversation.

## What we deferred (and why we did it right)

The design spec called out three things we didn't build for v0.1:

- Semantic search (embeddings)
- Entity/edge graph model (for richer relationships)
- Automatic corrupt-database recovery

All three are useful, and all three would have added weeks to the timeline. So we made a deliberate choice: design the schema so they slot in without data migration. The `memories` table has room for embedding vectors. The schema supports rich relationships even if we're only using simple bidirectional links right now. The durability story is clear about what we don't handle.

The important part: we called this out explicitly in the docs, not silently dropped it. v0.1 works and is useful today. v0.2's roadmap is visible. Anyone deploying tether knows what they're getting and what's coming.

## Why this matters for multi-agent systems

If you're building a system where multiple agents (or multiple instantiations of the same agent) need to collaborate on behalf of the same person, tether gives you a common language. One agent can `remember()` a decision. Another can `recall()` it and build on it. A third can `link()` it to a related fact. None of them need to implement their own memory system — they just read and write to shared ground.

This becomes crucial as AI tooling matures. Right now, most agents are single-threaded: you ask, it thinks, you get an answer. But future agents will be long-lived, collaborative, and distributed across devices. They'll need to accumulate knowledge over time, share context across sessions, and let you bounce between tools without losing continuity. That's only possible if memory is a substrate, not a silo.

tether is a small piece of that puzzle. It's deliberately boring — four verbs, SQLite, no magic. That boringness is the point. In a few years, when agents are everywhere, boring shared infrastructure will be more valuable than novel but isolated systems.

## Try it

Install with Claude Code:

```sh
claude mcp add tether -- uvx tether
```

Local memory is free. If you want cross-device sync, set `TETHER_SYNC_URL` and `TETHER_SYNC_TOKEN` pointing to a Turso database and you're done.

Feedback and issues are welcome. We're building this in public, and v0.2 will be shaped by what agents actually need.
