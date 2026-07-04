# tether — design

> **A shared memory layer for personal agents, across devices.** `tether` is an
> [MCP](https://modelcontextprotocol.io) server backed by a local SQLite file.
> Any MCP-compatible agent can `remember`, `recall`, `link`, and `forget` durable
> notes. It runs local-only with zero configuration; point it at a hosted
> [libSQL/Turso](https://turso.tech) primary and the same file becomes an embedded
> replica that syncs your memory across every device in near-real-time.

## Why

The near future is personal agents living across many devices — laptop, desktop,
phone. For that to feel like *one* assistant rather than several amnesiac ones,
memory has to be a substrate that follows you, readable and writable from every
device and from any agent, not siloed inside a single tool.

`tether` is that substrate. It is deliberately a *convenience layer*: it makes an
agent more useful when present, and must **never** break the agent's actual work
when absent or degraded.

## Scope decisions (settled during brainstorming)

| Question | Decision |
|---|---|
| Core purpose | Multi-agent **shared** memory for one person, across devices |
| Consumers | **Any** MCP-compatible agent (built as an MCP server, like `cleat`) |
| Data model | **Hybrid**: typed linked notes now; entity/edge graph layered on later |
| Storage | Local-first SQLite; **near-real-time** cross-device sync as a layer |
| Sync backend | libSQL/Turso **embedded replicas** (local-speed reads, no server to run) |
| Retrieval | **Hybrid**: FTS5 keyword/tag search now; embeddings later, no data migration |
| Distribution | Publishable like `cleat` (`uvx tether`); local-only default, sync opt-in |
| Tool surface | **Four verbs** only: `remember`, `recall`, `link`, `forget` |

## Architecture

A single installable package (same shape as `cleat`: `uvx tether`, no install
step required) that runs as an MCP server over a local SQLite file.

- **Local-only by default.** No accounts, no network. The DB lives at a standard
  path (e.g. `~/.local/share/tether/memory.db`). Every tool call reads/writes it
  directly. This alone is a complete, useful single-machine memory server.
- **Sync is opt-in.** If a backend URL + auth token are configured, the same
  local file is opened as a libSQL **embedded replica**: reads are served locally
  (no per-read network hop), writes round-trip to the hosted primary and then
  propagate back down to the other devices. No config → the sync layer is never
  engaged and the local-only path is byte-for-byte unchanged.

Reads staying local-speed is the point: during a live agent session `recall` is
called often; `remember` is comparatively rare. Cross-device freshness comes from
a short, bounded `sync_now()` before reads plus a background sync tick — never
blocking a read on the network.

### Components

1. **`store`** — schema, CRUD, and the FTS5 keyword index. Plain versioned `.sql`
   migrations, no ORM (the schema is small). Owns all SQL; nothing else speaks it.
2. **`sync`** — isolated optional wrapper. With config present, opens the
   embedded-replica connection and exposes `sync_now()`; absent/invalid config
   means this module is never engaged. Backend failure degrades to the local file.
3. **`server`** — the thin MCP tool + resource layer. Validates input and calls
   into `store`; no logic of its own. Mirrors how `cleat`'s `server.py` sits over
   `engine.py`.
4. **bootstrap/CLI** — the `uvx tether` entry point.

## Data model (v1)

One table, `memories`:

| Column | Notes |
|---|---|
| `id` | primary key |
| `type` | `user` \| `feedback` \| `project` \| `reference` (reused from the proven memory taxonomy) |
| `title` | short human-readable label; used by the upsert dedup probe |
| `body` | the fact itself; per-type structure (e.g. `Why:` / `How to apply:` for feedback/project) encouraged by convention |
| `tags` | comma-separated text for now (promote to a join table only if querying demands it) |
| `links` | JSON array of referenced ids (mirrors `[[name]]` cross-links) |
| `created_at` | set on insert |
| `updated_at` | bumped on every upsert; the staleness signal |
| `device_id` | which device wrote it — for sync debugging, not agent-facing |

An FTS5 shadow table indexes `title` / `body` / `tags`, kept in sync in the same
transaction as writes.

**Forward-compatible:** the "embeddings later" plan adds an `embedding BLOB`
column plus a `sqlite-vec` index and backfills existing rows — no migration of
current data. The "graph later" plan adds `entities` / `edges` tables alongside
`memories` without touching it.

## Tool surface (four verbs)

- **`remember(type, title, body, tags?, links?) -> {id, created|updated}`**
  **Upsert semantics** to fight duplication: a normalized, case-insensitive title
  match within the same `type` updates that row in place and bumps `updated_at`;
  otherwise inserts a new row. The store stays clean without relying on the agent
  to recall-before-write every time. Returns which happened so the agent knows.
- **`recall(query, type?, limit?) -> [{id, type, title, body, tags, updated_at}]`**
  FTS5 MATCH ranked by relevance, optionally filtered by `type`. Returns rich
  metadata — not just body text — so the agent can judge staleness (`updated_at`),
  cite ids when updating, and weigh a fresh project fact differently from an old
  one.
- **`link(id_a, id_b)`** — bidirectional link between two memories.
- **`forget(id)`** — hard delete (no soft-delete/audit trail for a personal tool).

The `memories` table is **not** exposed to the agent — no raw SQL verb. Narrow
verbs keep a confused/adversarial agent from dropping the table, mass-exfiltrating,
or corrupting the index; they also let the schema evolve (embeddings, graph,
storage swap) without breaking any agent. The human retains full raw access — it
is just a SQLite file on disk, openable and editable by hand.

## The boot resource (highest-leverage feature)

Pull-based `recall` only fires when the agent *thinks* to call it; mid-task it
often won't. So the single most useful surface is an **auto-included MCP resource**
— not a fifth verb — that the client loads into context each session, exactly the
way a `MEMORY.md` index already proves out.

- **Contents:** a **compact index of all memories** — one line per memory
  (`[type] #id title`), `SELECT id, type, title FROM memories ORDER BY updated_at
  DESC`. The agent sees everything that exists at a glance and pulls full bodies
  via `recall` using the id. Scales to hundreds of memories while staying small.
- Regenerated on each read, so it is always current.

## Error handling — the load-bearing rule

Memory is a convenience layer; it must **never** break the agent's real work.
Every failure degrades, never throws upward:

- **Sync backend unreachable / bad auth / timeout** → fall back to the local file
  silently; reads and writes still work locally and converge later. Surface a
  one-line `sync: offline` note, not an error.
- **Corrupt/missing DB file** → recreate from migrations; on real SQLite
  corruption, move the bad file aside and start fresh rather than hard-failing
  every call.
- **Malformed tool input** → structured error for that one call; never crash the
  server.
- **FTS query with special characters** → sanitize/escape rather than letting a
  MATCH syntax error bubble up.

(This mirrors the lesson from `cleat`'s parser: a layer fed untrusted or
unreliable input must treat every field as hostile and degrade, never raise on
the response path.)

## Testing

- **Store unit tests** — insert/upsert (the dedup probe: same title updates,
  different inserts), FTS ranking, `type`/tag filter, `forget`, migrate-from-empty.
  TDD, following `cleat`'s `tests/` layout.
- **Sync tests** — config-absent path is byte-identical to local-only;
  config-present opens a replica; backend-down degrades to local **without
  raising** (monkeypatch the connection to throw; assert reads/writes still
  succeed) — the analog of `cleat`'s "probe must never raise" test.
- **Server/MCP tests** — each verb round-trips; the boot resource renders the
  current index; malformed input returns structured errors, not crashes.
- **Manual `__main__` self-test** — remember → recall → upsert-dedup → forget
  against a temp DB, asserting the boot index reflects each change. Same
  convention as `python -m cleat.engine`.

## Out of scope (v1 — deliberately not built here)

Embedding/semantic search (v2, schema is ready for it), the entity/edge graph
(later, tables slot alongside `memories`), a `list`/`get`/browse verb (kept to
four verbs), multi-user/multi-tenant sharing, secret redaction, conflict-resolution
policy beyond libSQL's own last-writer-wins, and any hosted service operated *by
us* for other people (users bring their own backend).
