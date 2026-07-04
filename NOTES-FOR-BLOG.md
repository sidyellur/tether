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

## Publishing: two sharp edges in one afternoon

Shipping to PyPI turned up two problems neither the design nor the plan anticipated.

First, the package name. `tether` had shown as available weeks earlier (a plain `curl` to the JSON API returned 404), but PyPI's "add a pending publisher" flow rejected it outright with "This project name isn't allowed." The JSON API's 404 doesn't distinguish "never registered" from "registered but blocked/reserved" — we had to check the PEP 503 Simple Index API instead (the endpoint `pip` itself actually queries) to confirm the name had never been registered at all. That left one explanation: PyPI's admins almost certainly block names colliding with well-known brands — in our case, Tether/USDT, one of the largest cryptocurrencies — to head off dependency-confusion attacks. We renamed the PyPI distribution to `tether-memory`, while keeping the Python import, CLI entry point, and repo name as `tether`. Distribution name and import name never have to match; treating them as independent knobs cost us nothing.

Second, a GitHub Actions permissions footgun. Our release workflow requested `permissions: id-token: write` for PyPI's OIDC trusted publishing — and nothing else. It turns out that specifying *any* permission switches the whole block from "inherit repo defaults" to "everything else is `none`." `contents: read`, which `actions/checkout` needs, silently vanished. The failure mode was maximally confusing: GitHub Actions reported "repository not found" on our own private repo, rather than anything resembling a permissions error — a private repo returns "not found" instead of "forbidden" to an insufficiently-scoped token, as a security measure against leaking repo existence. The fix was one line (`contents: read`), but finding it meant reading the actual checkout logs rather than trusting the summary. Lesson: once you touch the `permissions:` block at all, audit every step's real requirements — there's no partial inheritance.

Neither issue was visible in local testing (build succeeded fine, tests passed fine) — both only surfaced once we tried to actually publish. A reminder that "it builds and tests pass" and "it ships" are different bars.
