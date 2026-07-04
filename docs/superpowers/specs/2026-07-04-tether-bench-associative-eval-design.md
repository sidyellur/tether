# tether — Associative Recall Evaluation Harness (`bench/`) design

**Status:** approved design, ready for implementation plan
**Date:** 2026-07-04
**Depends on:** Tier A — Associative Core (shipped, v0.4.0, PR #18). The harness measures the recall path that Tier A introduced (`Store.recall` with `assoc`/`budget`/`session`, `graph.py`'s edges + `spread` + `touch_session`).
**Related:** roadmap memory (tether associative-memory ideas and roadmap); Tier A design/plan; the blog journal (the "central bet" this harness exists to test).

## Why

Everything so far has verified that the Associative Core **runs** — degrade-never holds, the suite is green, semantic recall surfaces a non-matching note. That is *functional* verification. It has never been **measured** that associative recall is actually *better* than v0.2 hybrid recall, let alone that it *improves with use*.

Improving-with-use is Fable's central bet and the north star of the whole associative direction:

> *A graph grown purely from usage will, within weeks of real work, out-retrieve an LLM-extracted knowledge graph — because what an agent needs next is best predicted by what it reached for together before, not by what entities co-occur in text.*

That is an empirical claim with a clever implementation under it. This harness turns it into a number. It is the instrument that lets us *see* the bet succeed or fail — and, once it exists, it becomes the regression guard that answers "does B1 forgetting hurt recall?" and "does B2 crystallization help?" for every later tier.

**Two things must be measured, not one.** The *upside* (does the graph surface connected-but-non-matching memories?) AND the *downside* (does turning the graph on ever *hurt* the queries v0.2 already handles well?). tether's core covenant is **degrade-never**, and today that is enforced only *functionally* (byte-identical when off) — never at the *quality* layer (assoc-on ≥ v0.2 on queries v0.2 already handles). A harness that reports only lift on the graph's home turf is asymmetric for a degrade-never product. This design measures both.

## Scope

**In scope:**
- A committed, documented, hand-authored **scenario corpus** with two labeled query classes:
  - **graph-only queries** — golds reachable *only* through behavioral edges (measure upside/lift).
  - **control (direct) queries** — golds v0.2 already surfaces well (measure *no-regression*).
- A **runner** that measures recall quality across four conditions (v0.2 hybrid / assoc-cold / assoc-warmed / assoc-oracle) over both query classes.
- A **real-recall-path warm-up** that accretes Hebbian edges through the actual mechanism, and a **hand-wired oracle** upper bound with a precisely-defined "ideal" graph.
- **Metrics** (Recall@k, MRR, nDCG) with derived headline numbers: learning delta, headroom, and a **no-regression guard**.
- **Ablations** toggling each mechanism (semantic / explicit / hebbian / priming / spread-budget).
- A **pre-flight self-check** with two symmetric assertions: golds are *far* (anti-rigging) and targets are *found* (seed-findability).

**Out of scope (deferred, not dropped):**
- Realism/scale corpora (a real-derived corpus B; a programmatically generated corpus C for parameter sweeps). This spec is the *existence proof*; generalization is a later credibility upgrade.
- B1/B2-specific measures (boot-index relevance, forgetting precision, redundancy reduction). The harness is structured so these slot in later without rework.
- Latency/throughput profiling. This harness measures *quality*, not speed.
- Any comparison against external systems (Mem0, Zep, LoCoMo/LongMemEval) — explicitly rejected in the product research as contested/saturated/non-reproducible.

## Design decisions (the frame)

- **Existence proof, stated honestly.** A small hand-authored corpus with the real embedder proves the *learning delta is real and measurable* on a controlled case. It is NOT a generalization claim. The report header says so; N and the corpus's provenance are printed. No overclaiming.
- **The corpus must be un-riggable by construction, and prove it.** The only interesting lift is on golds that are *relevant but neither lexically nor semantically close* to the query — reachable only through behavioral edges. The harness **asserts** every graph-only gold is below a semantic-similarity threshold to its query before measuring; if a gold is too semantically close, the run fails loudly. Any lift over v0.2 is then genuinely attributable to the graph, not to smuggled semantic overlap.
- **Measure the downside too — degrade-never at the quality layer.** A second, control query class has golds v0.2 *already* surfaces well. The harness asserts **assoc-warmed does not regress** these (nDCG ≥ v0.2 within a small tolerance ε). Spreading + (later) B1 forgetting can pull noise that outranks a good direct hit; this guard catches it. It is the quality-layer counterpart to the functional byte-identical-when-off guarantee, and the exact regression test B1/B2 will need.
- **Warm through the real recall path, not by hand.** The claim is "use improves recall," so warm-up must produce Hebbian edges via the actual `recall`→`touch_session` loop, modeling realistic task-sessions. A hand-wired oracle exists only as an upper-bound comparison, never as the headline number.
- **The headline number measures durable edges, not warm sessions.** The warmed condition's eval runs in a **fresh, neutral session** distinct from every task session, so residual priming (an ephemeral session effect) does not contaminate the learning delta. Priming is measured separately, as an ablation.
- **Measurement, not a test.** The harness is a script that prints a report, not a pass/fail pytest. It lives outside the suite so it never gates CI on a quality number. One tiny smoke test guarantees it doesn't silently rot.
- **Deterministic + reproducible.** Static embeddings (Model2Vec) have no forward pass and no randomness; the warm-up replay is a fixed sequence. Same corpus + same code ⇒ same numbers. Real model opt-in via `importorskip("model2vec")`, matching the existing real-model test convention.

## Architecture

A new top-level `bench/` package (precedent: `scripts/selftest.py`). It imports `tether` directly and drives `Store`/`Graph` — it does not go through the MCP server.

- **`bench/corpus.py`** — the scenario data: a list of memories (type, title, body), a list of tasks (each a set of memory ids used together), an *optional* list of author-declared explicit links, and two lists of eval queries (graph-only and control), each: query text, target memory id, gold memory ids. Pure data + light helpers. Committed and frozen.
- **`bench/warmup.py`** — `warm(store, corpus)`: replays each task as one `session`, issuing a `recall()` per member memory with a matching query so the task's memories co-occur in the session working set and get Hebbian-wired by `touch_session`. Deterministic ordering. Uses task-specific session ids, never the neutral eval session.
- **`bench/conditions.py`** — builds a fresh `Store` for each of the four conditions from the same corpus (loads memories, applies the condition's graph setup), returning a store ready to evaluate. The oracle condition hand-inserts the ideal Hebbian edges directly via `Graph._upsert_edge` (definition below).
- **`bench/metrics.py`** — pure functions: `recall_at_k`, `mrr`, `ndcg` over a ranked id list vs a gold set. No dependencies beyond the ranking.
- **`bench/selfcheck.py`** — two pre-flight assertions (below): `assert_golds_far` (anti-rigging) and `assert_targets_found` (seed-findability).
- **`bench/run.py`** — entry point. Runs the self-check, builds each condition, evaluates both query classes, aggregates metrics, prints the report (tables + derived numbers + honesty header). `python -m bench.run`.

Nothing in `src/tether/` changes. The harness is strictly additive and read-only against the library's public surface (plus `Graph._upsert_edge` for the oracle, which is the same seam the tests already use).

## The four conditions

All share one corpus and both query classes. Each gets a fresh store.

| Condition | Graph setup | Question it answers |
|---|---|---|
| **v0.2 hybrid** | `assoc=False` | The baseline: keyword+semantic RRF, no graph. |
| **assoc-cold** | `assoc=True`; semantic + explicit edges built at load; **no warm-up** | Does spreading over content/link edges alone help, before any usage? |
| **assoc-warmed** | `assoc=True`; then `warmup.warm()` via the real recall path; eval in a **fresh session** | **The headline: does real usage improve recall (upside), without regressing direct hits (downside)?** |
| **assoc-oracle** | `assoc=True`; hand-wired **ideal** Hebbian edges (below) | Ceiling. Is the bottleneck *spreading* (mechanism) or *learning* (usage not producing the right edges)? |

**Oracle "ideal" graph, precisely defined:** for every task, wire a **full clique** among its member memories with `hebbian` edges at the maximum weight (`HEBBIAN_CAP = 5.0`). This is the strongest Hebbian state the learning loop could ever converge to (every within-task pair maximally co-recalled), so `oracle − warmed` isolates *how far real usage falls short of the perfect graph* — the mechanism-vs-learning diagnosis. No cross-task edges are added (those would be noise the real loop shouldn't produce either).

## Corpus (`bench/corpus.py`)

- ~30–60 memories, real prose, a coherent fictional software team (decisions, bugs, preferences, people, infra). Real prose so the real embedder produces honest geometry.
- **Tasks** group memories that are *used together*. Example task "auth outage": {the bug note, "Dave distrusts ORMs", the rollback decision, the postmortem}. A memory may belong to more than one task.
- **Graph-only eval queries** each target one memory by a directly-matching query; the **golds** are the *other* members of that memory's task(s), authored to share **no salient lexical overlap and low semantic similarity** with the query. These are the "connected but non-matching" memories only the usage graph can reach — the upside case.
- **Control (direct) eval queries** each target a memory whose gold *is* a memory v0.2 surfaces well (lexically/semantically close). These exist to prove assoc-on **does not regress** normal recall — the downside guard. They are the queries where the graph must, at worst, do no harm.
- **Explicit links** are an *optional, separate* corpus field — a small set of author-declared `link(a,b)` relationships, distinct from tasks (which drive Hebbian edges via warm-up). They feed the `explicit` edges present in `assoc-cold`. If the corpus declares none, `assoc-cold` is effectively semantic-only — itself a fair baseline; explicit links are included only where a human would genuinely have linked two memories.
- Frozen and documented in-file (a comment block explains each task's intent and why its graph-only golds are non-matching). **Not tuned to the result** — authored from the scenario, then measured.

## Warm-up (`bench/warmup.py`)

For each task: open a task-specific session id, and for each member memory issue a `recall(query=<memory's matching query>, session=<task session>)`. Because all members are touched within one session, `touch_session` wires the top-M co-active among them (Hebbian). Repeat each task a small fixed number of times (e.g., 3) to let weights accrete past the epsilon/one-off floor. The eval queries are **distinct** from the warm-up queries, and the warmed condition is evaluated in a **fresh, neutral session id** used by no task — so the headline measures durable Hebbian edges, not residual warm-session priming, and tests generalization to a new cue rather than memorization of the warm-up cue.

## Metrics (`bench/metrics.py`)

Per condition and **per query class**, averaged over the class's queries:
- **Recall@k** (k = 5, 10) — is the gold in the top-k of the returned ranking?
- **MRR** — reciprocal rank of the first gold.
- **nDCG@k** — rank-discounted gain over the gold set.

**Distribution, not just the mean.** With small N (~10–20 queries per class) a mean hides everything: a "+0.15 learning delta" might be one query going 0→1 with the rest flat. So every condition-vs-baseline comparison also reports the **per-query breakdown** and an **{improved / unchanged / regressed} count** (a query counts as improved/regressed when its per-query nDCG moves beyond ±ε vs. the baseline condition). This is the honest complement to the "small N" caveat, and it is what makes the no-regression guard credible: the bar on the control class is **zero regressed**, not merely a mean that clears ε.

Derived headline numbers:
- **Learning delta = warmed − cold** (graph-only class) — the value that usage adds. *This is the number the whole project is about.* Because graph-only golds are asserted semantically far and carry no explicit link, **assoc-cold ≈ v0.2 on this class by construction** — so cold is a valid floor and the delta cleanly equals *what usage added*. (The anti-rigging self-check does double duty here.)
- **Headroom = oracle − warmed** (graph-only class) — how much the real learning loop leaves on the table vs. the ideal graph. Large headroom ⇒ the learning mechanism (not spreading) is the bottleneck.
- **No-regression guard (control class):** assert **warmed nDCG@k ≥ v0.2 nDCG@k − ε** (small ε, e.g. 0.02) **and zero queries regressed** in the distribution breakdown. A failure means turning the graph on degraded recall the baseline already handled — a degrade-never violation at the quality layer. Reported prominently; it is the guard B1/B2 inherit.

## Ablations

Re-run the **warmed** condition (graph-only class) with one mechanism disabled at a time — semantic edges, explicit edges, Hebbian edges, priming, and spread budget (budget=0) — and report each metric delta. This attributes the warmed number to its parts and can reveal a mechanism that contributes nothing (a candidate for simplification or a bug). Priming appears here, and *only* here, so it never contaminates the headline.

## Pre-flight self-check (`bench/selfcheck.py`)

Two symmetric assertions run before any measurement; either fails loudly, naming the offending pair:

- **`assert_golds_far` (anti-rigging).** For every *graph-only* (query, gold) pair, cosine similarity with the real embedder must be **below a threshold** (e.g., 0.35 — well under the range where v0.2 would surface it directly). Guarantees the graph-only corpus cannot be won by plain semantic search, so any measured lift is genuinely the graph's.
- **`assert_targets_found` (seed-findability).** For every eval query (both classes), the query's **target** memory must appear in v0.2's top-k. The graph spreads *from* seeds; if the seed itself isn't retrieved, a condition underperforms for reasons unrelated to the graph and the learning delta silently measures "seed missed." This asserts the spread has a valid starting point in every case.

## Honesty framing (report header)

`run.py`'s output opens with: corpus name + memory/query counts (per class), the real model id, both self-check thresholds that passed, and explicit caveats kept **load-bearing**:
- *"Existence proof on a controlled corpus; small N; not a generalization claim — see corpora B/C (out of scope here)."*
- *"The self-check guards against semantic smuggling, not against the corpus author shaping task structure toward what Hebbian captures well. A single green number here is evidence, not proof of the universal claim."*

The numbers are presented as evidence, framed so they cannot be honestly quoted as more than they are.

## Testing

The harness is a measurement, not a pass/fail gate, but it must not rot:
- **One smoke test** (`tests/test_bench.py`): runs `run.py`'s pipeline on a **mini-corpus** (a few memories, 1 task, 1 graph-only + 1 control query) with the `FakeEmbedder`, asserting it completes, produces all four conditions across both query classes, and returns well-formed metrics. Hermetic (no model download), fast.
- **Metric unit tests**: `recall_at_k`/`mrr`/`ndcg` on hand-built rankings with known answers (including empty-gold and gold-absent edge cases).
- **Self-check unit tests**: a deliberately riggable mini-corpus (a graph-only gold that IS semantically close) makes `assert_golds_far` raise; a query whose target is *not* in v0.2 top-k makes `assert_targets_found` raise; a clean corpus passes both.
- **No-regression assertion unit test**: a synthetic metrics pair where warmed < v0.2 − ε on the control class trips the guard.
- The full real-corpus run is **opt-in** (`importorskip("model2vec")`), never part of the default suite — run by hand / on demand, its numbers recorded in the journal, not asserted in CI.

## Out of scope (deferred, not dropped)

- Corpus B (real-derived) and corpus C (generated, scale/parameter sweeps).
- B1/B2 quality axes (boot-index relevance, forgetting precision, redundancy) — same harness shape, added when those tiers land.
- Latency/throughput profiling.
- External-system or public-benchmark comparisons.
