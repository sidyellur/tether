# B1 usage-learning spike — diagnosis + prototype (2026-07-04)

**Branch:** `feat/b1-usage-learning-spike` (spike for human review — not merged).
**Question:** oracle proves ~0.133 nDCG of frozen graph_only headroom is
reachable via direct behavioral edges (cold 0.154 → oracle 0.287), but
usage-learning captured none of it (warmed ≈ cold, held-out delta −0.016).
Why, and can a deterministic local change to the learning rule claim it?

**Answer:** yes — root cause was the Hebbian learning *input*, not the Hebbian
rule, the traversal, or the eval seeds. Two defects compounded. After the fix,
the held-out frozen learning delta is **+0.118** (cold 0.154 → warmed
**0.271**, oracle 0.287): **~89% of the headroom is now claimable from usage**,
with all honesty guards green and the control no-regression guard PASS.

## Diagnosis (instrumented, frozen warmed graph, `main` @ pre-spike)

### 1. The within-task edges mostly never form

Oracle wires all 36 within-task pairs at cap 5.0. Under warming (`main`):

| edge population (kind=hebbian) | count |
|---|---|
| within-task pairs formed | **9 / 36** (only tasks 1–2: auth, search) |
| within-task pairs missing | 27 |
| spurious cross-task edges | **37**, *all at cap 5.0* |

Of the 11 (target → gold) pairs the graph_only eval queries need, **8 were
missing** (all except the three involving `dave_orm`/`priya_oncall`, i.e.
tasks 1–2 again).

### 2. Root cause A — uniform bump: the "usage" signal is the returned list

`store.recall` passes the **entire returned order** (up to `limit=20` of a
33-memory corpus, ~60% of the store) to `touch_session`, which bumps every id
by a **uniform +1.0**. The recall's actual subject (rank 1) carries exactly the
same weight as rank-20 padding. Session activation therefore carries almost no
information about what was used — it converges to "appeared in a top-20
recently", which nearly everything does.

### 3. Root cause B — mass ties are broken by `memory_id`: low-id squatting

With uniform bumps, activations are massively tied, and the Hebbian top-M cut
(`ORDER BY activation DESC, memory_id LIMIT 8`) decides **entirely on the id
tie-break**. Trace of one full warm session (`warm:data`, members
etl_nightly / timezone_bug / dashboard_owner / lena_utc):

```
recall('nightly aggregation')   bumped 20 ids, top-8 after:
  [priya_oncall 1.0, lena_utc 1.0, nadia_trust 1.0, auth_bug 1.0,
   pool_fix 1.0, rollback_policy 1.0, search_slow 1.0, redis_cache 1.0]
... identical top-8 after all four recalls (activations 1.875 each)
```

The queried member `etl_nightly` — **rank 1 of its own recall** — never enters
the top-8: the corpus authors the six people first (lowest ids), so the 8
lowest-id memories present in any returned list permanently squat the top-M.
Tasks 3–6 (higher ids) can never wire their own members; instead the low-id
clique (people + auth/search members, ids 1–11) wires pairwise at cap. That is
exactly the 9-formed/37-spurious pattern above, and the spurious cap-weight
clique's fan-out actively *hurt*: frozen warmed (0.138) lost three queries cold
(0.154) won (idempotency_bug→flag_rollout 7→None, timezone_bug→dashboard_owner
8→None, search_slow→index_decision 5→None).

### 4. What the gap is NOT

- **Not an eval-seed problem:** every graph_only target ranks **1** in v2 for
  its eval query (all 11), so spreading always starts from the right seed.
- **Not the Hebbian increment/cap/decay constants:** the 9 edges that did form
  hit cap 5.0 in 3 repeats — formation, once triggered, is plenty strong.
- **Not the traversal (for THIS gap):** under oracle edges the golds land at
  rank 7–9 — the protect-head keeps ~6–8 direct-hit slots above the tail, so
  rank ≈ 8 *is* the mechanism's ceiling (nDCG ≈ 0.287, not 1.0). That residual
  is a traversal/protect-head property, out of scope for B1; the B1 target was
  reaching 0.287 from usage, and the blocker was purely which edges exist.

Attribution of the 0.133: **~100% learning-rule (edge-formation input), 0%
eval-seed, 0% within-scope traversal.** The leading hypothesis from the brief
(chicken-and-egg: distant B never enters A's working set) is **half right** —
distant members DO enter the working set (each is queried by title in the same
session), but the top-M cut never selects them because uniform bumps + id
tie-break let unrelated low-id memories squat the working set. It's not that
the signal is absent; it's that the rule drowns it.

## The fix (deterministic, local, no LLM, no network)

Two changes, both to the learning input; traversal, ranking, and the #25
protect-head are untouched.

**1. Rank-weighted session bump** (`graph.py::touch_session`): the id at rank
r of the learned list is bumped by `HEBBIAN_BUMP_DECAY**r` (0.5^r: 1.0, 0.5,
0.25, …) instead of a uniform +1.0, and ranks whose bump falls below
`SESSION_TTL_ACTIVATION` are skipped. A recall's working-set contribution now
reflects what the recall was *about*. Activations become graded → the id
tie-break becomes irrelevant → no more squatting. A new
`HEBBIAN_WIRE_FLOOR=0.25` gates pair-wiring to members that are genuinely
active (recently a top-2 subject), so stragglers that merely survive the TTL
are not wired. **The old rule is exactly the knob setting
`HEBBIAN_BUMP_DECAY=1.0, HEBBIAN_WIRE_FLOOR=0.0`** (test-pinned).

**2. Learn from the head, not the full returned list**
(`store.py::recall`): `touch_session` now receives the protected head (the
direct seed hits) instead of the whole `order`. Rank-weighting alone was not
enough — measured intermediate: all 36 within-task pairs formed, but frozen
warmed *dropped* to 0.035, because session **priming** re-surfaces any member
that once entered the session into the next result list's tail, where it gets
re-bumped and re-wired: a feedback loop that wired **80** spurious cross-task
edges, many at cap. Spread- and priming-surfaced members now *consume* session
activation (priming still works unchanged) but never *produce* it. Co-recall
wiring then means "recalled as subjects in the same session" — precisely the
distant-pair usage signal B1 needed.

### Warmed edge set after the fix

| | main (before) | spike (after) |
|---|---|---|
| within-task pairs formed | 9 / 36 | **36 / 36** (w = 2.0–5.0) |
| needed (target→gold) pairs | 3 / 11 | **11 / 11** |
| spurious cross-task edges | 37, all at cap 5.0 | 23, mostly w ≤ 3.0 |

The remaining spurious edges come from genuine seed overlap between title
queries (e.g. shared vocabulary), are weight-limited, and did not stop the
result below.

## Frozen graph_only nDCG@10 (held-out, contamination-free)

| condition | before (main) | after (spike) |
|---|---|---|
| v2 (keyhole) | 0.000 | 0.000 |
| cold | 0.154 | 0.154 |
| warmed | 0.138 | **0.271** |
| oracle | 0.287 | 0.287 |
| **held-out learning delta** | **−0.016** | **+0.118** dist {improved 7, unchanged 2, regressed 2} |
| headroom captured (of 0.133) | ≈0% | **≈89%** |

Per-query frozen gold ranks (cold → warmed, after): 7 of 11 queries improve
(golds enter top-10 at ranks 4–9 that were absent before); 2 unchanged; 2
regress — `auth_bug→dave_orm` (its edge formed but at the low end, w≈2, and
auth-task siblings crowd the tail) and `etl_nightly→lena_utc` (cold rank 6 via
a semantic edge; warmed crowds it out with stronger hebbian siblings). Both are
fixable-looking with knob nudges, but on N=11 that would be eval-set fitting —
deliberately not done.

## Guards

- `assert_golds_far`, `assert_warmup_disjoint`, `assert_principles_far`,
  `assert_targets_found`: **green, unmodified** (bench runs them before any
  number is produced; no assertion was touched).
- Control no-regression guard: **PASS**, dist {improved 0, unchanged 12,
  regressed 0}; control nDCG 1.000 in all conditions.
- Full suite: **150 passed, 1 skipped** (4 new tests: rank-weighted bump,
  no-ties/no-squatting, wire floor, old-rule-as-knob-setting; 2 tests updated
  to the new expected activations; 1 bench test widened to check session state
  as well as edges, since a single-seed eval query no longer necessarily adds
  an edge — intent preserved, the freeze-restore test is unchanged).
- Degrade-never: `TETHER_ASSOC=0` / `budget=0` / empty graph paths are
  untouched (learning only happens inside the assoc branch); protect-head
  ordering logic untouched, so #25 holds by construction and by the guard.

## Honest caveats

- Small N (11 graph_only queries, 1 hand-authored corpus): this is an
  existence proof that the learning rule was the blocker, not a
  generalization claim. Corpora B/C remain the real test.
- Warm-up replays title-queries; real sessions query by content. The signal
  the fix learns from is temporally-adjacent direct recalls within roughly
  the 2-recall decay window (decay 0.5, floor 0.25) — not semantic
  relatedness. In a long, topic-interleaved organic session, this will wire
  unrelated but time-adjacent subjects together. The mitigation is that such
  edges are weak (w≈0.5 vs cap 5.0), weight-scaled in traversal, and bounded
  by protect-head, so this is graceful degradation rather than avoidance —
  a documented risk, not a solved one. An organic-usage corpus with
  interleaved topics and content (non-title) queries is exactly what
  corpora B/C must stress to test this premise.
- 2 of 11 queries regressed vs cold; the mean gain is not uniform.
- The remaining 0.016 to oracle is mostly the two regressed queries plus
  edge-weight differences (2–4.5 vs cap 5.0); the rank-7–9 oracle ceiling
  itself is a protect-head traversal property, a separate (B-later) question.

## Verdict: **ship** (after B-tier review)

The warmed ≈ cold mystery is closed with a mechanical root cause and a
two-line-of-concept fix that is deterministic, local, additive, knob-reversible
and captures ~89% of the proven headroom on the frozen bench with zero control
regressions. Recommend: review the two per-query regressions, then promote to
main; next measurement should be an organic-usage corpus (B/C) rather than
more knob tuning on meridian.
