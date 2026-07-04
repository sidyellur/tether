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
claude mcp add tether -- uvx tether-memory
```

…or install it first:

```sh
pip install tether-memory
claude mcp add tether -- tether-memory
```

(The package is named `tether-memory` on PyPI — `tether` was already reserved
as a common brand name. `tether` in `claude mcp add tether -- ...` is just the
label Claude Code uses to refer to this server; it doesn't need to match the
installed command.)

By default memory lives in a local SQLite file at
`~/.local/share/tether/memory.db` (override with `TETHER_DB`). No accounts, no
network — this is the whole tool for a single machine.

## Sync across devices (optional)

Point tether at a [Turso](https://turso.tech) / libSQL database and the local
file becomes an embedded replica — local-speed reads, writes that propagate to
your other devices. Install the extra and set two env vars:

```sh
pip install 'tether-memory[sync]'
export TETHER_SYNC_URL='libsql://<your-db>.turso.io'
export TETHER_SYNC_TOKEN='<your-auth-token>'
```

If the backend is unreachable, tether logs `sync offline` and keeps working
against the local file; writes converge when it comes back.

## Semantic search (optional)

By default `recall` is **hybrid**: keyword (FTS5) results are fused with
semantic (vector) results, so a query finds relevant memories even when the
exact words differ ("automobile" recalls a note about your "car"). Semantic
recall runs a small **static** embedding model locally — no network, no API
key, nothing to hang on. Install the extra:

```sh
pip install 'tether-memory[semantic]'
```

Without the extra (or with `TETHER_SEMANTIC=0`), tether runs keyword-only
FTS5 — semantic is a pure add-on and never a requirement. The first run embeds
existing memories once (a one-time backfill); after that it is incremental.

Environment:

| Var | Default | Effect |
|---|---|---|
| `TETHER_SEMANTIC` | on | set `0`/`false`/`off` to force keyword-only recall |
| `TETHER_EMBEDDING_MODEL` | `minishlab/potion-base-8M` | override the local static model |

## Consolidation (optional)

tether keeps a superseded fact rather than overwriting it: when a memory is
replaced, the old one is marked no longer current (retained for history) and
excluded from `recall` and the boot index. Recall also gently favors more
recent facts. Two opt-in behaviors go further:

| Var | Default | Effect |
|---|---|---|
| `TETHER_CONSOLIDATE` | off | on (`1`/`true`) merges a near-duplicate on write — supersedes the old fact instead of fragmenting the store (needs the `[semantic]` extra) |
| `TETHER_DEDUP_THRESHOLD` | `0.92` | cosine similarity required to treat two facts as duplicates |
| `TETHER_DECAY_HALF_LIFE_DAYS` | off | set a positive number to exponentially down-rank older facts in recall |
| `TETHER_AUTHOR` | device id | attribution recorded on each memory |

Consolidation never deletes — only `forget` does. All of this degrades to
plain keyword recall when the semantic extra is absent.

## Associative recall (optional)

`recall` doesn't just return keyword/semantic matches — it follows a **usage
graph** to related memories, so asking about one thing surfaces its connected
context. The graph's edges come from three local, deterministic sources — no
LLM, no network:

- **semantic** — nearest neighbours by embedding (needs the `[semantic]` extra),
- **explicit** — the `link()` verb,
- **hebbian** — memories you recall *together* get wired together over time.

Every hit carries a `via` receipt saying why it surfaced (a direct match, or the
edge it came through), and two optional `recall` args tune it:

| Arg / var | Default | Effect |
|---|---|---|
| `budget` (per call) | `TETHER_RECALL_BUDGET` | how far to follow associations; `0` = direct matches only |
| `session` (per call) | time-bucketed | group related recalls so they prime each other |
| `TETHER_ASSOC` | on | set `0`/`false`/`off` for plain keyword+semantic recall |
| `TETHER_RECALL_BUDGET` | `24` | default association breadth |

With `TETHER_ASSOC=0` (or `budget=0`, or an empty graph), `recall` behaves exactly
as before — associative recall is purely additive and never breaks a lookup.

## Tools

| Tool | What it does |
|---|---|
| `remember(type, title, body, tags?, links?)` | Save a memory; upserts on `type`+`title` so facts refine rather than duplicate |
| `recall(query, type?, limit?, budget?, session?)` | Hybrid keyword + semantic search, then follows the usage graph to related memories; returns id/type/title/body/tags/updated_at + a `via` receipt |
| `link(id_a, id_b)` | Bidirectional link between two memories |
| `forget(id)` | Delete a memory |

Plus an auto-loaded resource `tether://memory-index` — a compact one-line-per-memory index surfaced each session.

## License

MIT
