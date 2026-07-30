# tether — where the work is tracked

This file used to be the v0.1 build checklist. Every box in it stayed
unchecked while the project shipped v0.1 through v0.5.1, so it described a
state of the world that hadn't been true for months. Rather than maintain a
second, slower copy of the truth, it now just points at the real sources.

**Current status: v0.5.1.** The four memory verbs, boot index, FTS5, semantic
recall, consolidation, the associative usage graph, the self-organizing store,
and opt-in crystallization are all implemented. See the README for what each
one does and how to turn it on.

## Where things live

| What | Where |
|---|---|
| **Open work** | [GitHub Issues](https://github.com/sidyellur/tether/issues) |
| **Design rationale** | [`docs/superpowers/specs/`](docs/superpowers/specs/) — one spec per feature, starting with [the original design](docs/superpowers/specs/2026-07-03-tether-design.md) |
| **Implementation plans** | [`docs/superpowers/plans/`](docs/superpowers/plans/) — the step-by-step build record matching each spec |
| **Shipped behavior** | [README](README.md) |

## Historical note

The v0.1 plan's "deferred, designed-for-but-not-built" list has largely been
built: semantic/embedding search shipped in v0.2, and the entity/edge graph
became `graph.py` in the associative core. What remains from that list —
`sqlite-vec`-backed vector search, automatic corrupt-DB recovery, and a true
backgrounded sync tick — is tracked in Issues, where it can carry the
discussion and evidence a checkbox can't.
