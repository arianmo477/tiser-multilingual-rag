"""Unit tests for utils.sampling.balance_by_dataset_name.

Run from the repo root:  python -m pytest tests/
"""

from collections import Counter

from utils.sampling import balance_by_dataset_name


def mk(name, n):
    return [{"dataset_name": name, "question_id": f"{name}-{i}"} for i in range(n)]


def ids(rows):
    return [r["question_id"] for r in rows]


def test_no_duplicates_with_small_buckets():
    # 'a'/'b' are smaller than the per-category quota (base = 500//5 = 100),
    # which is exactly the case the old implementation duplicated.
    data = mk("a", 30) + mk("b", 40) + mk("c", 500) + mk("d", 500) + mk("e", 500)
    out = balance_by_dataset_name(data, category="train", max_samples=500)
    assert len(out) == 500
    assert len(ids(out)) == len(set(ids(out)))  # no duplicates


def test_under_full_pool_returns_all_unique():
    data = mk("a", 10) + mk("b", 20)  # only 30 available, ask for 500
    out = balance_by_dataset_name(data, category="train", max_samples=500)
    assert len(out) == 30
    assert len(set(ids(out))) == 30


def test_balanced_when_all_buckets_large():
    data = sum((mk(n, 100) for n in "abcde"), [])
    out = balance_by_dataset_name(data, category="train", max_samples=500)
    counts = Counter(r["dataset_name"] for r in out)
    assert len(out) == 500
    assert all(v == 100 for v in counts.values())


def test_deterministic_for_fixed_seed():
    data = mk("a", 300) + mk("b", 300)
    out1 = balance_by_dataset_name(data, category="train", max_samples=200)
    out2 = balance_by_dataset_name(data, category="train", max_samples=200)
    assert ids(out1) == ids(out2)  # same seed -> identical selection AND order


def test_test_category_excludes_tot_semantic():
    data = mk("tot_semantic_test", 50) + mk("a", 100)
    out = balance_by_dataset_name(data, category="test", max_samples=100)
    assert all(r["dataset_name"] != "tot_semantic_test" for r in out)
