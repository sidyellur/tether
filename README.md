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

v0.1 is implemented. Design and rationale: [`docs/superpowers/specs/2026-07-03-tether-design.md`](docs/superpowers/specs/2026-07-03-tether-design.md). Implementation plan: [`docs/superpowers/plans/2026-07-03-tether-v0.1-implementation.md`](docs/superpowers/plans/2026-07-03-tether-v0.1-implementation.md).

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

## Install

Requires Python ≥3.10 on a POSIX system (Linux/macOS).

Register it with Claude Code — with [uv](https://docs.astral.sh/uv/):

```sh
claude mcp add tether -- uvx tether
```

…or install it first:

```sh
pip install tether
claude mcp add tether -- tether
```

By default memory lives in a local SQLite file at
`~/.local/share/tether/memory.db` (override with `TETHER_DB`). No accounts, no
network — this is the whole tool for a single machine.

## Sync across devices (optional)

Point tether at a [Turso](https://turso.tech) / libSQL database and the local
file becomes an embedded replica — local-speed reads, writes that propagate to
your other devices. Install the extra and set two env vars:

```sh
pip install 'tether[sync]'
export TETHER_SYNC_URL='libsql://<your-db>.turso.io'
export TETHER_SYNC_TOKEN='<your-auth-token>'
```

If the backend is unreachable, tether logs `sync offline` and keeps working
against the local file; writes converge when it comes back.

## Tools

| Tool | What it does |
|---|---|
| `remember(type, title, body, tags?, links?)` | Save a memory; upserts on `type`+`title` so facts refine rather than duplicate |
| `recall(query, type?, limit?)` | Keyword search; returns id/type/title/body/tags/updated_at |
| `link(id_a, id_b)` | Bidirectional link between two memories |
| `forget(id)` | Delete a memory |

Plus an auto-loaded resource `tether://memory-index` — a compact one-line-per-memory index surfaced each session.

## License

MIT
