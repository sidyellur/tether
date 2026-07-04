# tether

**A shared memory layer for personal agents, across devices.** `tether` is an
[MCP](https://modelcontextprotocol.io) server backed by a local SQLite file. Any
MCP-compatible agent can `remember`, `recall`, `link`, and `forget` durable notes
— facts about you, your projects, your preferences — so context follows you
instead of dying with each session.

It runs **local-only with zero configuration**. Point it at a hosted
[libSQL/Turso](https://turso.tech) primary and the same file becomes an embedded
replica that syncs your memory across every device in near-real-time.

## Why

The near future is personal agents living across many devices — laptop, desktop,
phone. For that to feel like *one* assistant rather than several amnesiac ones,
memory has to be a substrate that follows you: readable and writable from every
device and from any agent, not siloed inside a single tool.

`tether` is that substrate. It is deliberately a *convenience layer* — it makes an
agent more useful when present, and never breaks the agent's work when degraded.

## Status

Early. The design is written up in
[`docs/superpowers/specs/2026-07-03-tether-design.md`](docs/superpowers/specs/2026-07-03-tether-design.md);
implementation has not started yet.

## Design at a glance

- **Four verbs**, nothing more: `remember` · `recall` · `link` · `forget`.
- **Upsert on write** so the store doesn't rot into near-duplicates.
- **Rich recall** (id, type, title, body, tags, `updated_at`) so an agent can
  judge staleness and cite what it updates.
- **An auto-loaded boot index** — a compact one-line-per-memory list surfaced to
  the agent each session, so memory helps even when the agent doesn't think to
  search.
- **Local-first, sync optional** — the local path is untouched when no backend is
  configured; degradation never throws.
- **Keyword search now, embeddings later** — the SQLite schema is built so
  semantic search and a full entity/edge graph slot in without migrating data.

## License

MIT
