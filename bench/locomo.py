"""Retrieval-only evaluation of tether on LoCoMo (snap-research/locomo).

LoCoMo is ten long multi-session conversations with ~2,000 questions, each
labelled with the dialogue turns ("evidence") that answer it. We ingest
every turn as a tether memory, ask `recall(question)`, and measure whether
the evidence turns come back. No LLM anywhere: this is "did the memory layer
hand the agent the right facts", the one thing a memory layer controls, and
a number whose methodology no vendor can dispute. It is NOT the end-to-end
LLM-judged accuracy that Mem0 / Zep / Letta report; those need an answering
model and a judge, and the two are not comparable.

    python -m bench.locomo                 # real embedder if installed
    python -m bench.locomo --convs 2       # quick run
    python -m bench.locomo --conditions keyword,bm25

The dataset (~1.5 MB JSON) is downloaded on first use from the LoCoMo GitHub
repository into ~/.cache/tether/ (or pass --path / TETHER_LOCOMO_PATH).

Conditions:
  keyword   tether, no embedder (FTS5 only), graph off
  hybrid    tether, FTS5 + embedder fused by RRF, graph off (budget=0)
  full      tether, hybrid + associative graph at the default budget, cold
  bm25      rank_bm25 over the same texts - a textbook baseline (optional dep)

`hybrid` and `full` are skipped, with a note, when no embedder is available.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict

from bench import metrics
from tether.store import Store

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
KS = (5, 10, 20)
ALL_CONDITIONS = ("keyword", "hybrid", "full", "bm25")


# --- data --------------------------------------------------------------------

def default_path():
    return os.environ.get("TETHER_LOCOMO_PATH") or os.path.join(
        os.path.expanduser("~"), ".cache", "tether", "locomo10.json")


def load_locomo(path=None):
    """The LoCoMo conversations list, downloading the file on first use."""
    path = path or default_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sys.stderr.write(f"downloading LoCoMo to {path} ...\n")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def turns_of(conv):
    """[(dia_id, speaker, session_date, text)] for every turn, in order.
    An image-only turn contributes its caption so it stays findable."""
    out = []
    c = conv["conversation"]
    i = 1
    while f"session_{i}" in c:
        date = c.get(f"session_{i}_date_time", "")
        for t in c[f"session_{i}"]:
            text = t.get("text", "") or ""
            if t.get("blip_caption"):
                text += f" [shared an image: {t['blip_caption']}]"
            out.append((t["dia_id"], t["speaker"], date, text))
        i += 1
    return out


def questions_of(conv):
    """Answerable questions with evidence; the adversarial class (5) has no
    answer in the conversation and is excluded, as the LoCoMo authors do."""
    return [q for q in conv.get("qa", [])
            if q.get("category") != 5 and q.get("evidence") and q.get("question")]


def turn_text(speaker, date, text):
    return f"{speaker} ({date}): {text}"


# --- conditions --------------------------------------------------------------

def build_store(turns, embedder, assoc):
    """A fresh in-memory store with one memory per turn; returns it and the
    memory-id -> dia_id map."""
    conn = sqlite3.connect(":memory:")
    st = Store(conn, "bench", lambda *a, **k: None, embedder=embedder,
               assoc=assoc, sync_read_interval=0, excerpt_chars=0)
    st.migrate()
    id2dia = {}
    for dia, spk, date, text in turns:
        r = st.remember("reference", f"{dia} {spk}", turn_text(spk, date, text))
        id2dia[r["id"]] = dia
    return st, id2dia


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class _Bm25:
    def __init__(self, turns):
        from rank_bm25 import BM25Okapi  # optional dependency
        self.dias = [d for d, *_ in turns]
        self.index = BM25Okapi([_tok(turn_text(s, d, t)) for _, s, d, t in turns])

    def rank(self, question, k):
        scores = self.index.get_scores(_tok(question))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [self.dias[i] for i in order]


def _make_runner(cond, turns, embedder):
    """A callable question -> ranked [dia_id] for one condition, or None if
    the condition cannot run here (no embedder / no rank_bm25)."""
    if cond == "bm25":
        try:
            b = _Bm25(turns)
        except ImportError:
            return None
        return lambda q, k: b.rank(q, k)
    if cond in ("hybrid", "full") and embedder is None:
        return None
    st, id2dia = build_store(
        turns, embedder if cond != "keyword" else None, assoc=(cond == "full"))
    kw = {} if cond == "full" else {"budget": 0}
    counter = [0]

    def run(q, k):
        counter[0] += 1
        hits = st.recall(q, limit=k, session=f"locomo-{cond}-{counter[0]}", **kw)
        return [id2dia[h["id"]] for h in hits]
    return run


# --- evaluation --------------------------------------------------------------

def evidence_recall_at_k(ranked, evidence, k):
    """Fraction of the evidence turns in the top k. Stricter than
    bench.metrics.recall_at_k (which is hit-or-miss on ANY gold): a multi-hop
    question with two evidence turns scores 0.5 if only one comes back."""
    if not evidence:
        return 0.0
    top = ranked[:k]
    return sum(1 for e in evidence if e in top) / len(evidence)


def _score(ranked, evidence):
    out = {f"R@{k}": evidence_recall_at_k(ranked, evidence, k) for k in KS}
    out["all@10"] = 1.0 if all(e in ranked[:10] for e in evidence) else 0.0
    out["MRR"] = metrics.mrr(ranked, evidence)
    return out


def evaluate(data, embedder=None, conditions=ALL_CONDITIONS, convs=None):
    """Run the conditions over the conversations and return a report:

    {"n_questions": N, "skipped": [cond, ...],
     "conditions": {cond: {"R@5":..,"R@10":..,"R@20":..,"all@10":..,"MRR":..,
                           "ms": mean latency,
                           "by_category": {cat: {"R@10":.., "n":..}}}}}
    """
    data = data[:convs] if convs else data
    sums = defaultdict(lambda: defaultdict(float))
    cat_sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    latency = defaultdict(float)
    skipped = set()
    n = 0
    for conv in data:
        turns = turns_of(conv)
        qs = questions_of(conv)
        if not turns or not qs:
            continue
        runners = {}
        for cond in conditions:
            r = _make_runner(cond, turns, embedder)
            if r is None:
                skipped.add(cond)
            else:
                runners[cond] = r
        for q in qs:
            n += 1
            evidence = set(q["evidence"])
            cat = CATEGORIES.get(q.get("category"), "other")
            for cond, run in runners.items():
                t0 = time.perf_counter()
                ranked = run(q["question"], max(KS))
                latency[cond] += time.perf_counter() - t0
                for name, v in _score(ranked, evidence).items():
                    sums[cond][name] += v
                    if name == "R@10":
                        cat_sums[cond][cat][0] += v
                        cat_sums[cond][cat][1] += 1
    report = {"n_questions": n, "skipped": sorted(skipped), "conditions": {}}
    for cond in sums:
        row = {name: (v / n if n else 0.0) for name, v in sums[cond].items()}
        row["ms"] = (latency[cond] / n * 1000) if n else 0.0
        row["by_category"] = {
            cat: {"R@10": (s / c if c else 0.0), "n": c}
            for cat, (s, c) in cat_sums[cond].items()}
        report["conditions"][cond] = row
    return report


def format_report(report):
    cols = ["R@5", "R@10", "R@20", "all@10", "MRR"]
    lines = [f"LoCoMo retrieval-only: {report['n_questions']} questions "
             "(adversarial excluded), evidence recall"]
    if report["skipped"]:
        lines.append(f"skipped (no embedder / no rank_bm25): {', '.join(report['skipped'])}")
    lines.append(f"{'condition':<10}" + "".join(f"{c:>9}" for c in cols) + "   ms/query")
    for cond, row in report["conditions"].items():
        lines.append(f"{cond:<10}" + "".join(f"{row[c]:9.3f}" for c in cols)
                     + f"   {row['ms']:7.2f}")
    cats = [c for c in ("single-hop", "multi-hop", "temporal", "open-domain")
            if any(c in r["by_category"] for r in report["conditions"].values())]
    if cats:
        lines.append("")
        lines.append("R@10 by category:")
        lines.append(f"{'condition':<10}" + "".join(f"{c:>13}" for c in cats))
        for cond, row in report["conditions"].items():
            lines.append(f"{cond:<10}" + "".join(
                f"{row['by_category'].get(c, {}).get('R@10', 0.0):13.3f}" for c in cats))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--path", help="locomo10.json (default: download to ~/.cache/tether)")
    p.add_argument("--convs", type=int, help="only the first N conversations")
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                   help=f"comma-separated subset of {','.join(ALL_CONDITIONS)}")
    p.add_argument("--no-embedder", action="store_true",
                   help="skip loading the semantic model (keyword/bm25 only)")
    args = p.parse_args(argv)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}")
    embedder = None
    if not args.no_embedder:
        from tether.embed import get_embedder
        embedder = get_embedder()
        if embedder is None:
            sys.stderr.write("no embedder available (pip install 'tether-memory[semantic]'); "
                             "hybrid/full will be skipped\n")
    data = load_locomo(args.path)
    report = evaluate(data, embedder, conditions, convs=args.convs)
    print(format_report(report))
    return report


if __name__ == "__main__":
    main()
