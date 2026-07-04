import math
from bench import metrics


def test_recall_at_k_hit_and_miss():
    assert metrics.recall_at_k([9, 3, 7], {7}, k=3) == 1.0
    assert metrics.recall_at_k([9, 3, 7], {7}, k=2) == 0.0
    assert metrics.recall_at_k([], {7}, k=5) == 0.0
    assert metrics.recall_at_k([1, 2], set(), k=5) == 0.0  # empty gold -> 0


def test_mrr_first_gold_rank():
    assert metrics.mrr([5, 8, 2], {2}) == 1.0 / 3
    assert metrics.mrr([2, 8, 5], {2, 8}) == 1.0        # first position
    assert metrics.mrr([1, 2, 3], {9}) == 0.0           # absent


def test_ndcg_at_k_perfect_and_worse():
    # single gold at rank 1 -> 1.0
    assert metrics.ndcg_at_k([7, 1, 2], {7}, k=3) == 1.0
    # single gold at rank 2 -> 1/log2(3) normalized by ideal (1.0)
    got = metrics.ndcg_at_k([1, 7, 2], {7}, k=3)
    assert math.isclose(got, (1 / math.log2(3)), rel_tol=1e-9)
    assert metrics.ndcg_at_k([1, 2, 3], {9}, k=3) == 0.0
    assert metrics.ndcg_at_k([1, 2], set(), k=2) == 0.0
