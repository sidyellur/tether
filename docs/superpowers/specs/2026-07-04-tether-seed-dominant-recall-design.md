# Seed-Dominant Recall Ranking — Design

**Date:** 2026-07-04
**Status:** approved (brainstorming complete)
**Fixes:** #25 (Tier A: associative spreading buries direct hits — degrade-never violated at the quality layer)
**Acceptance test:** `bench/` no-regression guard flips FAIL → PASS (run locally with the real model)

## Problem

Enabling associative recall craters direct-hit quality. On the `bench/` control class
(queries v0.2 already ranks well; gold = the direct target):

| condition | control nDCG | graph-only nDCG |
|---|---|---|
| v2 (assoc off) | **0.866** | 0.167 |
| warmed (assoc on, default budget 24) | **0.107** | 0.228 |

11 of 12 control queries regress. There is **no budget > 0** where control is preserved
(sweep in #25). This is a degrade-never violation at the *quality* layer — the functional
guarantee (byte-identical when `assoc=False`) still holds and every functional test passes;
only the quality benchmark surfaces it.

## Root cause

`recall` (store.py:486-488) ranks seeds and spread-reached nodes together on **one additive
activation scale**:

```python
activation, receipts = self._graph.spread(scores, budget, type)
order = sorted(activation.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
```

A seed's activation *is* its RRF score — tiny (top seed ≈ 1/61 ≈ 0.016). A hop transmits
`a * w * HOP_DECAY` (graph.py:210). The intent was "HOP_DECAY 0.4 keeps a hop below the
seed," and that holds for semantic (`w`≈1.0 → factor 0.4) and explicit (1.2 → 0.48) edges —
both < 1, attenuating. **But Hebbian weight caps at 5.0**, so a warmed/oracle Hebbian hop has
factor `5.0 × 0.4 = 2.0` — it *amplifies*. A within-task co-recalled neighbor (not the query's
target) receives `0.016 × 2.0 = 0.032`, double the direct hit's own seed activation. Multi-seed
accumulation stacks further. Nothing structurally enforces "spreading is a boost, never an
override" — HOP_DECAY was assumed sufficient and is not.

The upside is real and must be preserved: the usage graph does surface connected-but-non-matching
memories (graph-only 0.167 → 0.228, → 0.347 at budget 8; 6/11 improved, 0 regressed). This is a
fix to the ranking, not a retreat from the bet.

## Design: protect-head / re-rank-tail

**Design evolution (what the bench rejected).** The original design here was a strict *two-tier*
split (v0.2 seeds locked on top, spread-reached nodes below). The bench harness killed two
successive designs before the third held — this is exactly what the harness is for:

1. **Strict two-tier** (seeds above all non-seeds): guard PASSED (control 0.866) but graph-only
   went *dead flat* at the v2 baseline (0.167 across cold/warmed/oracle) — the upside vanished.
   Root cause, found by instrumenting one query: **semantic recall has no similarity floor
   (issue #15), so `_seed_scores` returns the *entire store* as near-tied seeds.** Every memory is
   a "seed," so "seeds above non-seeds" degenerates to "rank everything by v0.2 score," discarding
   spread entirely.
2. **Seed-activation floor / weighted-RRF fusion**: any weight on spread that helped graph-only
   *immediately* regressed control (even w=0.1: control 0.866 → 0.824). Because control golds are
   *direct-but-not-graph-central* and graph-only golds are *graph-central-but-not-direct*, a single
   fused score that rewards connectivity inherently demotes direct hits.

The instrumentation revealed the real structure: control golds sit in the **top few** v0.2 ranks;
graph-only golds are buried deep in the flat v0.2 tail (rank 18–30) yet carry huge spread
activation (~0.97 vs a seed RRF of ~0.016 — a 60× scale gap). So the winning design protects the
head and re-ranks the tail:

- Compute `seed_order` = the v0.2 seeds sorted by v0.2 score (`(-score, id)`).
- **Head:** `seed_order[:PROTECT_HEAD]` — locked in exact v0.2 order. A direct hit here cannot be
  demoted by spreading.
- **Tail:** every other activated memory, re-ranked by **spread activation** descending. A
  connected-but-weakly-matched memory (deep in the v0.2 tail, high activation) climbs into the
  top-10 slots the direct hits didn't claim.
- `order = (head + tail)[:limit]`.

`PROTECT_HEAD` is the one knob (default **8**, `TETHER_PROTECT_HEAD`). It trades protection for
upside: larger protects more direct hits but leaves fewer top-10 slots for the tail re-rank.

### Why it works (empirical, not provable)

Unlike strict two-tier, this is *not* a by-construction guarantee — it is tuned against the bench.
At `PROTECT_HEAD=8` on the SCENARIO corpus:

- **control nDCG = 0.866 == v2** (guard PASS, 0/12 regressed) — control golds live in the top ~3
  v0.2 ranks, well inside the protected head, so they are never touched.
- **cold graph-only = 0.167 == v2** — with only semantic/explicit edges (no learned Hebbian), the
  tail re-rank is neutral: assoc-on-before-use doesn't hurt.
- **warmed graph-only = 0.221 > 0.167** (learning delta **+0.054**, 2 improved / 0 regressed;
  R@k 0.455 → 0.636) — learned co-recall lifts the connected golds out of the tail.

Below ~8 the protected head is too small and *cold* graph-only dips under v2 (semantic-edge spread
reshuffles the near-tied tail unhelpfully before any learning); above ~10 the head eats all the
top-10 slots and the upside disappears. 8 is the validated middle.

### Priming and receipts

Priming still adds `_PRIMING_WEIGHT * session_activation` into a *copy* of the seeds (`activated`)
before spread, so it can lift a memory within the tail re-rank but never reorders the protected
head (which is keyed off the pre-priming `seeds` scores). The `via` receipt logic is unchanged —
it keys off `h["id"] not in seeds`; since semantic recall makes nearly everything a seed, most
hits carry `{"seed": True}`, which is honest (they *were* v0.2 candidates) even when their rank
came from the tail re-rank. Receipt fidelity for tail-promoted memories is a known limitation
(below).

## Secondary fix: default recall budget 24 → 8

Independently, drop the default `TETHER_RECALL_BUDGET` 24 → 8 (both `config._DEFAULT_RECALL_BUDGET`
and the `Store(..., recall_budget=8)` constructor default). The #25 sweep peaked associative upside
near budget 8 and degraded toward 24 as spreading floods a small store.

## Known limitations

- **Tuned, not proven.** `PROTECT_HEAD=8` is fit to one 34-memory corpus; it is a sensible default,
  not a scale-invariant guarantee. The honest framing (already in `bench/run.py`'s `_HONESTY`
  banner) applies: existence proof, not generalization. A future improvement is a *relative* head
  (protect v0.2 hits whose score clearly stands above the flat tail) rather than a fixed count.
- **Root dependency on issue #15.** The whole difficulty stems from semantic recall returning the
  entire store with no relevance floor. A min-score / gap cutoff on `_seed_scores` (issue #15) would
  make "seed" meaningful again and likely simplify this ranking. #15 should be revisited alongside.
- **`via` receipts for tail-promoted hits** report `{"seed": True}` rather than a spread path,
  understating the graph's role in their rank. Cosmetic; deferred.

## Scope / non-goals

- **In scope:** the head/tail ranking in `recall`; the `protect_head` knob (Store + config + server);
  the budget default; unit tests; a `bench/` acceptance run; README note; version bump.
- **Not in scope:** changing `spread()` internals, edge weights, HOP_DECAY, Hebbian cap, or priming
  weight; fixing issue #15's semantic floor (noted as the deeper fix). B2 (crystallization) stays
  deferred behind this fix, per #25.

## Test strategy

**Hermetic unit tests** (FakeEmbedder, hand-inserted edges — no model):
1. A high-weight Hebbian neighbor cannot outrank the query's own direct hit (the #25 failure;
   `budget=1` isolates the single amplifying hop).
2. Degrade-never: `assoc=False` recall is byte-identical to v0.2 (regression guard on the refactor).
3. Default budget resolves to 8; default `protect_head` resolves to 8 (config + Store constructor).

**Acceptance test** (local, real `potion-base-8M`, `HF_HUB_OFFLINE=1`): re-run
`python -m bench.run` on `SCENARIO`. Expected: `no_regression` guard **PASS** with control
distribution `{regressed: 0}` and warmed control nDCG == v2 control nDCG; graph-only learning
delta stays positive. This run is laptop-bound (org egress blocks the model remotely — same reason
Task 8 was local), so the whole fix is implemented and validated locally.
