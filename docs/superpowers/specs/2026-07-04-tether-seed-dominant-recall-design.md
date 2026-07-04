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

## Design: two-tier rank fusion

Split the final ranking into two tiers separated by the **v0.2 seed set**, and make seeds
structurally undisplaceable by spread-reached nodes.

The seed set is exactly what `recall` already computes as `scores = self._seed_scores(query, type)`
— the pure v0.2 hybrid result (FTS+semantic RRF, gentle recency, optional decay), captured
**before** priming or spread mutate it.

- **Tier 1 (protected):** the v0.2 seeds, ranked in **exact v0.2 score order** (`(-score, id)`).
  They occupy the top slots verbatim.
- **Tier 2 (associative):** every other node that received activation — via priming *or* spread
  — ranked by total activation descending. Filled in **below** all seeds.
- `order = (tier1 + tier2)[:limit]`.

### Why it is provably non-regressing

Control-class golds *are* v0.2 seeds sitting near the top. Two-tier keeps every seed at its exact
v0.2 rank and only appends non-seeds below it — no gold can be displaced from its v0.2 position.
Therefore **warmed control nDCG === v2 control nDCG, at any budget, with 0 regressions, by
construction.** This is stronger than the guard's "≥ v2 − ε": it is exact equality. Degrade-never
at the quality layer becomes a structural property, the counterpart to the functional
byte-identical-when-off guarantee.

Formally: for any k, v0.2's top-k over the seeds is a prefix of two-tier's ranking (identical
order); appended tier-2 nodes are all non-seeds, so no seed's rank ≤ its v0.2 rank changes, and
no gold (a seed) is displaced. nDCG@k over any gold set drawn from the seeds is unchanged.

### Why upside survives

Graph-only golds are *semantically far* from their query (the anti-rigging design — asserted
cos < 0.35 by `selfcheck.assert_golds_far`), so they are never v0.2 seeds. They land in tier 2
and are still surfaced by spread, ranked among the associative nodes by activation. The learning
delta (warmed − cold on graph-only) stays positive.

### Priming

Priming (session-recency boost, `_PRIMING_WEIGHT * a`) currently adds into `scores` before spread.
Under two-tier, a primed-but-not-v0.2-matched node is **not** a seed, so it falls into tier 2 —
correct: priming is an associative boost and must not override a direct match either. Primed nodes
that *are* v0.2 seeds stay in tier 1 at their v0.2 rank (priming does not reorder tier 1). The
headline bench warmed run uses a fresh session, so priming contributes nothing there regardless.

### Receipts (`via`) — unchanged

The `via` receipt logic keys off `h["id"] not in scores` (the seed set): seeds get
`{"seed": True}`, tier-2 nodes keep their spread path receipt. Because the seed set is the same
object used for the tier boundary, this stays correct with no change.

## Secondary fix: default recall budget 24 → 8

With two-tier ranking, budget can no longer hurt the control class (seeds are protected at every
budget), so it becomes a pure upside/cost knob. The #25 sweep peaked graph-only nDCG at **budget 8**
(0.347) and degraded toward 24 as the spread tier floods with noise. Drop the default from 24 to 8
in both places it is defined:

- `config._DEFAULT_RECALL_BUDGET` (server path)
- the `Store(..., recall_budget=24)` constructor default (library path)

This is a product-default change, not a safety mechanism — safety comes from the two-tier split.

## Scope / non-goals

- **In scope:** the final-ranking split in `recall`; the two budget-default constants; unit tests;
  a `bench/` acceptance run; README/docs note; version bump.
- **Not in scope:** changing `spread()` internals, edge weights, HOP_DECAY, Hebbian cap, or priming
  weight — the fix is entirely in how the final order is assembled, leaving the activation dynamics
  (and their receipts/telemetry value) intact. Edge decay (the un-decayed-Hebbian concern) remains a
  later tier. B2 (crystallization) stays deferred behind this fix, per #25.

## Test strategy

**Hermetic unit tests** (FakeEmbedder, hand-inserted edges — no model, no numpy dependency beyond
existing):
1. A high-weight Hebbian neighbor of a seed cannot outrank that seed (the exact #25 failure).
2. Tier-1 order equals the pure-v0.2 order for the same query (seeds not reordered by activation).
3. A primed non-seed node ranks below all seeds.
4. Spread-only (tier-2) nodes still appear in the result and below the seeds (upside preserved).
5. Degrade-never: `assoc=False` recall is byte-identical to v0.2 (unchanged guarantee — regression
   guard on the refactor).
6. Default budget resolves to 8 (config and Store constructor).

**Acceptance test** (local, real `potion-base-8M`, `HF_HUB_OFFLINE=1`): re-run
`python -m bench.run` on `SCENARIO`. Expected: `no_regression` guard **PASS** with control
distribution `{regressed: 0}` and warmed control nDCG == v2 control nDCG; graph-only learning
delta stays positive. This run is laptop-bound (org egress blocks the model remotely — same reason
Task 8 was local), so the whole fix is implemented and validated locally.
