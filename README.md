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

**v0.5.1.** The core (four memory verbs + boot index + FTS5) shipped in v0.1;
since then recall has grown a semantic arm, consolidation, an associative usage
graph, a self-organizing store, and opt-in crystallization — each additive and
each degrading cleanly to plain keyword recall. Every feature below is
implemented.

Design and rationale start at
[`docs/superpowers/specs/2026-07-03-tether-design.md`](docs/superpowers/specs/2026-07-03-tether-design.md);
the associative core, seed-dominant recall, self-organizing store (Tier B1), and
crystallization (Tier B2) each have their own design doc under
[`docs/superpowers/specs/`](docs/superpowers/specs/), with matching plans in
[`docs/superpowers/plans/`](docs/superpowers/plans/).

## Design at a glance

- **Four memory verbs**: `remember` · `recall` · `link` · `forget`. (Enabling
  crystallization adds one reflection-control tool, `dismiss_cluster` — not a
  memory operation.)
- **Upsert on write** so the store doesn't rot into near-duplicates.
- **Rich recall** (id, type, title, body, tags, `updated_at`, plus a `via`
  receipt saying why each hit surfaced) so an agent can judge staleness and
  cite what it updates. `body` is a query-centered excerpt, not the whole
  memory — see [Excerpts](#excerpts).
- **An auto-loaded boot index** — a compact one-line-per-memory list surfaced to
  the agent each session, so memory helps even when the agent doesn't think to
  search.
- **Local-first, sync optional** — the local path is untouched when no backend is
  configured; degradation never throws.
- **Hybrid search, associative on top** — FTS5 keyword hits and local static
  embeddings are fused, then a usage graph (explicit links, learned co-recall,
  semantic neighbours) pulls in connected memories. Every layer is additive
  and degrades to plain keyword recall.
- **Safe under parallel tool calls** — an agent that fires several
  `remember`/`recall` calls at once gets atomic, correctly-reported results;
  see [Performance and durability](#performance-and-durability).

## Install

Requires Python ≥3.10 on Linux, macOS, or Windows.

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
`~/.local/share/tether/memory.db` — on Windows,
`%LOCALAPPDATA%\tether\memory.db` (override either with `TETHER_DB`, or set
`XDG_DATA_HOME`, which is honored on every platform). No accounts, no
network — this is the whole tool for a single machine.

## Project awareness

tether knows which project it is serving: Claude Code sets
`CLAUDE_PROJECT_DIR` in every MCP server's environment, and tether takes the
directory's name as the project (override or disable with `TETHER_PROJECT`).
That drives three things, none of which need configuration:

- **The boot index leads with this project.** The auto-loaded index opens
  with a `# This project (<name>)` section, then `# Everything else`, so the
  agent starts a session already looking at the decisions and gotchas for the
  repo it is in rather than whatever you touched last, anywhere.
- **Work memories are tagged automatically.** `project`, `feedback` and
  `reference` memories get a `proj:<name>` tag unless the agent passes a
  `proj:` tag itself. `user` memories are about you, not the work, and stay
  global. The tag is an ordinary tag: `recall(tags="proj:<name>")` lists a
  project's memories deterministically.
- **Recall prefers this project on a near-tie.** A hit tagged with the
  current project ranks a few places ahead of an equally-good hit from
  another project; it never outranks a clearly better match, and untagged
  memories are neither boosted nor penalized.

| Var | Default | Effect |
|---|---|---|
| `TETHER_PROJECT` | basename of `CLAUDE_PROJECT_DIR` | name the project explicitly; `off` disables project awareness |

Nothing falls back to the working directory: outside Claude Code (or with
`TETHER_PROJECT=off`) tether behaves exactly as before.

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

Writes push immediately. Reads also pull, debounced to at most once every
`TETHER_SYNC_READ_INTERVAL` seconds (default 30) — so a device that only
*asks* things still sees what your other devices wrote, instead of staying
frozen at its own startup state until it happens to write something. The
read-path pull is bounded much more tightly than the write-path one: if the
backend is slow, the recall returns local data and the pull lands for the
next read rather than making you wait.

| Var | Default | Effect |
|---|---|---|
| `TETHER_SYNC_READ_INTERVAL` | `30` | seconds between read-path pulls; `0` = only sync on writes |
| `TETHER_DEVICE_ID` | hostname | the device id recorded on each memory (and the default `TETHER_AUTHOR`) |

One thing to know about replicas: libSQL forwards **every write** to the
hosted primary, and with the associative graph on (the default) `recall`
writes too — it records what was recalled together so memories can wire up
over time. On a replica that makes each recall a few network round-trips on
top of the local search. If that matters more to you than learned
associations, `TETHER_ASSOC=0` makes recall read-only again.

## Keyword search

The keyword arm is SQLite FTS5 over title, body and tags, ranked by bm25. Ask
in plain language: a memory that contains *some* of the query's words is a
hit, and one that contains more of them ranks higher, so "how do we run the
integration tests?" finds the note that says "pytest runs the tests" (common
function words are ignored). The index stems English words, so `tests`
matches `test` and `deciding` matches `decided`. Stemming is English-only;
turn it off for a store in another language and tether rebuilds the index
on the next start.

| Var | Default | Effect |
|---|---|---|
| `TETHER_FTS_STEMMING` | on | set `0`/`false`/`off` to index words exactly as written |

Measured on the LoCoMo long-conversation benchmark (one memory per dialogue
turn, 1,536 questions, "did recall return the turns that answer it" in the
top 10), the keyword arm alone finds the evidence for 62% of questions,
against 54% for a textbook BM25 over the same text.

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

Consolidation never deletes — `forget` soft-deletes (see [Tools](#tools)), and
only the admin CLI's `purge` is permanent. All of this degrades to plain
keyword recall when the semantic extra is absent.

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
| `TETHER_RECALL_BUDGET` | `8` | default association breadth |
| `TETHER_PROTECT_HEAD` | `8` | how many top direct hits are locked above associations |
| `TETHER_SEED_FLOOR` | `0.35` | minimum cosine similarity a semantic hit needs to seed an associative walk; below it a memory is only reachable through an edge. `0` disables the floor |

Associative recall is **seed-dominant**: the top direct matches are locked in
place, and associations only fill the slots below them — so turning association
on never demotes a hit that keyword/semantic search already ranked highly.

With `TETHER_ASSOC=0` (or `budget=0`, or an empty graph), `recall` behaves exactly
as before — associative recall is purely additive and never breaks a lookup.

## Self-organizing store (optional)

As a store grows, tether keeps it legible using the same usage graph:

- **Hub-curated boot-index.** The auto-loaded memory index is capped once it
  passes `TETHER_BOOT_INDEX_CAP` (default 50) — the cap always applies, so a
  large store never gets an unbounded index. With a graph, above the cap it
  shows two labeled slices — **load-bearing** memories (highest *behavioral*
  degree: `explicit` links + learned co-recall, never mere similarity) and the
  most **recent** ones — so the index stays small and shows what actually
  matters. Without a graph (`TETHER_ASSOC=0`), it falls back to a plain
  most-recent-N list instead of the hub/recency split. Below the cap it's the
  full newest-first list either way.
- **Forgetting-by-disconnection** (opt-in, `TETHER_FORGET`). A bounded sweep
  runs every `TETHER_FORGET_INTERVAL` writes and *soft-archives* memories that
  are both **old** (`TETHER_FORGET_AGE_DAYS`, default 90) and **behaviorally
  isolated** (no `explicit`/`hebbian` edge — semantic similarity doesn't count).
  Archived memories drop out of recall and the boot-index but are **retained and
  reversible** (it reuses the same mark-invalid machinery as consolidation;
  nothing is deleted). Safety rails: never runs without a live behavioral graph,
  below `2 × CAP` memories, or more than `TETHER_FORGET_MAX_PER_SWEEP` (default
  10) per sweep.

| var | default | effect |
|---|---|---|
| `TETHER_BOOT_INDEX_CAP` | `50` | curate the boot-index above this size |
| `TETHER_FORGET` | off | enable the forgetting sweep |
| `TETHER_FORGET_AGE_DAYS` | `90` | minimum age to be eligible to fade |
| `TETHER_FORGET_INTERVAL` | `20` | writes between sweeps |
| `TETHER_FORGET_MAX_PER_SWEEP` | `10` | max archived per sweep |

With `TETHER_FORGET` off (default) and a normal store size, recall and the
boot-index behave exactly as before.

## Crystallization (optional, off by default)

With `TETHER_CRYSTALLIZE=1`, tether reflects: it detects dense clusters of
related memories and offers them for naming. Read `tether://crystallization`
during a reflection pass (it is pull-only, never auto-loaded) to get candidate
clusters; name a real principle with `remember(..., crystallizes=[source_ids])`
— which writes the principle and links it over its sources — or drop a candidate
with `dismiss_cluster(id_a, id_b)`. Clusters are seeded by explicit links +
usage (semantic similarity fills out membership), so this finds *"these belong
together"* structure, not mere topical similarity. tether finds the structure;
your agent supplies the words.

A crystallized principle becomes a boot-index hub and is reachable from its
sources in recall. Note: this makes "named" a third importance signal alongside
"used" and "linked" — deliberate, since an agent judging something
principle-worthy is a strong signal.

## Tools

| Tool | What it does |
|---|---|
| `remember(type, title, body, tags?, links?, crystallizes?)` | Save a memory; upserts on `type`+`title` so facts refine rather than duplicate. `crystallizes=[ids]` writes it as a principle over those sources (needs `TETHER_CRYSTALLIZE`) |
| `recall(query?, type?, limit?, budget?, session?, tags?, id?, full?)` | Hybrid keyword + semantic search, then follows the usage graph to related memories; returns id/type/title/body/tags/updated_at + a `via` receipt. `body` is a **query-centered excerpt** — see [Excerpts](#excerpts) — with `id=N` fetching one memory whole. `tags` is an exact-match filter (a memory must carry every listed tag); combine it with `query`, or omit `query` for a guaranteed-complete tag lookup |
| `link(id_a, id_b)` | Bidirectional link between two memories |
| `forget(id)` | Soft-delete a memory: marks it no longer current (excluded from recall/the boot index) via the same reversible `valid_to` machinery as consolidation, rather than deleting the row. See [Export and permanent deletion](#export-and-permanent-deletion) for a real, permanent delete |
| `dismiss_cluster(id_a, id_b)` | Reflection control (crystallization): drop the candidate cluster nucleated by peak edge `(id_a, id_b)` so it isn't re-surfaced. Not a memory operation; only relevant with `TETHER_CRYSTALLIZE` |

Plus three resources: the auto-loaded `tether://memory-index` (a compact
one-line-per-memory index surfaced each session), the pull-only
`tether://status` (runtime config: semantic/sync state, memory and edge
counts, DB path — for debugging what's actually active), and, with
`TETHER_CRYSTALLIZE`, the pull-only `tether://crystallization` (candidate
clusters for a reflection pass).

## Excerpts

`recall` returns a **relevance-centered excerpt** of each memory's body, not
the whole thing — the window is centered on the first query term that appears,
so you see *why* the memory matched rather than just its opening lines. A hit
that was cut also carries `truncated: true` and `body_chars` (the real length),
and you fetch the one memory you actually want in full with `recall(id=N)`.

This is the search-engine shape: the result list is an index of pointers with
enough text to judge relevance, not a payload of documents. It matters because
memories can be large — a single 44KB journal memory made unrelated queries
cost ~57–67KB per call while the retrieval itself took under a millisecond.
The response was fat, not the engine:

| query | full bodies | excerpts |
|---|---|---|
| `seed dominance` | 57.0KB | **1.4KB** |
| `hebbian edges` | 66.9KB | **2.0KB** |
| `cold start latency` | 66.9KB | **2.0KB** |

A memory shorter than the excerpt width is returned whole and unmarked, exactly
as before.

| Var / arg | Default | Effect |
|---|---|---|
| `TETHER_EXCERPT_CHARS` | `500` | excerpt width; `0` returns full bodies |
| `id` (per call) | — | fetch just this memory, whole |
| `full` (per call) | `false` | full bodies for every hit — costs the whole payload; prefer `id=` |

## Performance and durability

tether is meant to be invisible in an agent's loop, so the hot paths are
measured and kept flat as the store grows. Numbers below are from a local
SQLite store with the semantic and associative layers on, single process:

| memories | `remember` | `recall` (rare term) | `recall` (term in most memories) |
|---|---|---|---|
| 500 | 1.6 ms | 1.8 ms | 3.7 ms |
| 2,000 | 1.8 ms | 2.5 ms | 7.7 ms |
| 8,000 | 3.4 ms | 1.0 ms | 19 ms |

A few things that make this hold:

- **Writes don't scale with the store.** The embedding matrix used for
  semantic search and neighbour wiring is kept in memory and patched row by
  row on every write, rather than re-read from SQLite. It is rebuilt only
  when vectors change wholesale (a model change, a backfill) or when another
  process has written to the file (a CLI `purge`, a second server, a sync
  pull) — SQLite's `data_version` counter catches that.
- **Parallel tool calls are serialized.** MCP runs each tool call on its own
  thread, and agents issue calls in parallel. All Store operations take one
  lock, so a `recall` and a `remember` arriving together are each atomic:
  no interleaved transactions, no half-committed writes, and `action` is
  always right.
- **Commits don't fsync.** Local connections run WAL with
  `synchronous=NORMAL`: still safe against corruption, but the last few
  transactions can be lost if the *machine* loses power before a checkpoint
  (an application crash loses nothing). Every `remember` and, with the graph
  on, every `recall` commits, so this is one disk sync saved per call.
- **Search stays cheap.** Vector search is a single numpy matmul over the
  in-memory matrix — well under a millisecond at thousands of memories,
  which is why there is no vector-index extension to install. Keyword cost
  is FTS5's: proportional to how many memories match the query.

Costs to expect once: the first boot after installing the `[semantic]` extra
(or changing the model) embeds every existing memory and wires its
neighbours, which takes a second or two per few thousand memories. The boot
index and tag-only lookups scan the store on each call; both are fast at
typical sizes (under 10 ms at 2,000 memories) and are the next things on the
list to cache.

## Export and permanent deletion

`forget` never deletes data — it soft-deletes, like consolidation. Two admin
operations, deliberately kept off the MCP tool surface so an agent can't
trigger them, live in a small CLI instead:

```sh
tether export                    # dump all current memories to JSON (stdout)
tether export -o backup.json     # ...or to a file
tether import backup.json        # merge an export back into the store
tether restore <id>              # un-forget a soft-deleted memory
tether purge <id> --yes          # permanently delete a memory (bypasses forget)
```

`import` replays records through the normal write path, so it upserts on
`type`+`title` like `remember` does — importing into a non-empty store merges
rather than duplicating. Ids are not preserved (an id in the file may map to a
different one here); links are remapped accordingly, and a link pointing
outside the file is dropped rather than pointed at the wrong memory. The
report tells you what happened: `{"created", "updated", "skipped", "linked",
"dropped_links"}`.

`restore` clears `valid_to`, reversing a `forget` (or a consolidation, or a
forgetting sweep). It refuses if a newer memory has since claimed the same
`type`+`title`, naming the blocker rather than failing opaquely.

`purge` refuses to run without `--yes`. All commands honor the same
`TETHER_DB`/`TETHER_SYNC_*` env vars as the server.

## License

MIT
