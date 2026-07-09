"""
utils/sampling.py

Class-balanced subsampling for TISER datasets, shared by training
(train_qlora.py) and evaluation (inference.py).

Kept in its own module (rather than io_gpu.py) so the sampling logic has no
heavy dependencies and can be unit-tested in isolation — see tests/test_sampling.py.
"""

import random
from collections import defaultdict


def _balance_by_key(data, key_fn, max_samples, seed=42):
    """Return up to `max_samples` items, balanced across buckets of key_fn(x).

    Each bucket contributes `max_samples // num_buckets` items; any shortfall
    (from buckets smaller than that quota) is topped up from the remaining,
    not-yet-selected items so the result never contains duplicates.
    """
    random.seed(seed)
    buckets = defaultdict(list)
    for x in data:
        buckets[key_fn(x)].append(x)

    if not buckets:
        return []

    base = max_samples // len(buckets)
    selected = []
    leftovers = []

    for key in buckets:
        bucket = buckets[key]
        if len(bucket) <= base:
            # Whole bucket is used; nothing is left over for the top-up pool.
            selected.extend(bucket)
        else:
            # Sample by index so the chosen items are excluded from leftovers
            # exactly once (avoids re-adding a selected sample -> duplicates).
            chosen = set(random.sample(range(len(bucket)), base))
            selected.extend(bucket[i] for i in chosen)
            leftovers.extend(bucket[i] for i in range(len(bucket)) if i not in chosen)

    # Top up toward max_samples from items NOT already selected (no overlap).
    remaining = max_samples - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(random.sample(leftovers, min(remaining, len(leftovers))))

    random.shuffle(selected)
    return selected


def balance_by_dataset_name(data, category, max_samples, seed=42):
    """Return up to `max_samples` items, balanced across `dataset_name`.

    For `category == "test"`, the `tot_semantic_test` dataset is excluded.
    """
    if category == "test":
        data = [x for x in data if x["dataset_name"] != "tot_semantic_test"]

    return _balance_by_key(
        data,
        key_fn=lambda x: x["dataset_name"],
        max_samples=max_samples,
        seed=seed,
    )


def balance_by_lang_and_dataset(data, category, max_samples, seed=42):
    """Return up to `max_samples` items, balanced across every
    (language, dataset_name) cell.

    E.g. 4 languages x 5 datasets = 20 cells -> each cell contributes
    `max_samples // 20` items, with the shortfall from small cells topped up
    at random from the remaining pool (never duplicating a selected item).

    Samples without a `language` field fall into the "unknown" cell.
    For `category == "test"`, the `tot_semantic_test` dataset is excluded.
    """
    if category == "test":
        data = [x for x in data if x["dataset_name"] != "tot_semantic_test"]

    return _balance_by_key(
        data,
        key_fn=lambda x: (x.get("language", "unknown"), x["dataset_name"]),
        max_samples=max_samples,
        seed=seed,
    )