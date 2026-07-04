"""Pure ranking metrics. No tether/embedding imports — plain arithmetic over
a ranked list of ids and a gold set."""
import math


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    return 1.0 if any(mid in gold for mid in ranked[:k]) else 0.0


def mrr(ranked, gold):
    if not gold:
        return 0.0
    for i, mid in enumerate(ranked):
        if mid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    dcg = 0.0
    for i, mid in enumerate(ranked[:k]):
        if mid in gold:
            dcg += 1.0 / math.log2(i + 2)  # rank i (0-based) -> discount log2(i+2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
