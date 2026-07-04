# tether — task tracker

Checklist view of the v0.1 build. Full step-by-step detail (code, tests,
commands) lives in
[`docs/superpowers/plans/2026-07-03-tether-v0.1-implementation.md`](docs/superpowers/plans/2026-07-03-tether-v0.1-implementation.md).
Design rationale is in
[`docs/superpowers/specs/2026-07-03-tether-design.md`](docs/superpowers/specs/2026-07-03-tether-design.md).

## v0.1 — local-first memory MCP server + opt-in sync

- [ ] **Task 1 — Scaffolding.** `pyproject.toml`, `src/tether/__init__.py`, `tests/__init__.py`. Editable install + pytest run green.
- [ ] **Task 2 — `config.py`.** DB path, sync credentials, device id resolved from env. Zero-config default = local file, no sync.
- [ ] **Task 3 — `store.py` schema.** `memories` table + external-content FTS5 index + sync triggers; idempotent `migrate()`.
- [ ] **Task 4 — `remember()`.** Upsert on `type` + normalized `title` so facts refine instead of duplicating.
- [ ] **Task 5 — `recall()`.** FTS5 keyword search, type filter, rich returns (id/type/title/body/tags/updated_at); punctuation-safe.
- [ ] **Task 6 — `link()` / `forget()` / `boot_index()`.** Bidirectional links, hard delete, compact newest-first index.
- [ ] **Task 7 — `sync.py`.** Connection factory: local `sqlite3` or libSQL embedded replica; degrade-to-local on any failure. **Spike the `libsql-experimental` client first.**
- [ ] **Task 8 — `server.py`.** `FastMCP` — four verbs + `tether://memory-index` resource + `main()`. Tools return `{"error": ...}` rather than crash.
- [ ] **Task 9 — Docs + self-test.** README install/sync/tools; `scripts/selftest.py` end-to-end smoke.

## Deferred (not in v0.1 — designed for, not built)

- [ ] Semantic/embedding search (`embedding BLOB` + `sqlite-vec`, backfill; no data migration).
- [ ] Entity/edge graph model layered alongside `memories`.
- [ ] Automatic corrupt-DB move-aside-and-recreate recovery (v0.1 degrades to a per-call error instead).
- [ ] True bounded/backgrounded sync tick beyond the best-effort `sync_now`.
