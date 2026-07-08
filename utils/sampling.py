"""
utils/sampling.py

Class-balanced subsampling for TISER datasets, shared by training
(train_qlora.py) and evaluation (inference.py).

Kept in its own module (rather than io_gpu.py) so the sampling logic has no
heavy dependencies and can be unit-tested in isolation — see tests/test_sampling.py.
"""

import random
from collections import defaultdict


def balance_by_dataset_name(data, category, max_samples, seed=42):
    """Return up to `max_samples` items, balanced across `dataset_name`.

    Each dataset bucket contributes `max_samples // num_datasets` items; any
    shortfall (from buckets smaller than that quota) is topped up from the
    remaining, not-yet-selected items so the result never contains duplicates.
    For `category == "test"`, the `tot_semantic_test` dataset is excluded.
    """
    random.seed(seed)
    buckets = defaultdict(list)
    for x in data:
        if category == "test" and x["dataset_name"] == "tot_semantic_test":
            continue
        buckets[x["dataset_name"]].append(x)

    names = list(buckets.keys())
    base = max_samples // len(names)
    selected = []
    leftovers = []

    for name in names:
        bucket = buckets[name]
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
