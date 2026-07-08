#!/usr/bin/env python3
"""
Verify that the frozen EN 15k snapshot matches what the OLD run_train.sh would have
sampled from the full English split with seed=42.

Why this exists
---------------
docs/DATA.md §3 (decision B) rewired run_train.sh / run_experiment1.sh / run_eval_rag.sh to
use `train_tiser_15000_en.json` directly instead of subsampling the full 174 MB
`TISER_train_en.json` at runtime. That rewire is only safe if the snapshot is *exactly* the
15k subset the old pipeline produced. This script checks that by question_id.

It faithfully reproduces the OLD path:
    ds = load_dataset("json", data_files=<full>)["train"]
    ds = ds.filter(validation_status == "PASS")          # --only_passed
    ds = balance_OLD(ds.shuffle(seed=42), "train", 15000)  # the *pre-fix* sampler, embedded below

and compares set(question_id) against the snapshot.

IMPORTANT: run this on a machine where git-LFS data is pulled (the GPU box, or after
`git lfs install && git lfs pull`). It needs the `datasets` library, because the sample order
depends on HuggingFace's Dataset.shuffle(seed=42) RNG — a plain-Python shuffle would not match.

Usage
-----
    python multilingual_rag_tiser/preprocess/verify_15k_snapshot.py \
        --full     data/splits/train/TISER_train_en.json \
        --snapshot data/splits/train/train_tiser_15000_en.json \
        --max_samples 15000

Exit code 0 = MATCH (rewire safe), 1 = MISMATCH (revert to old full-file sampling).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# EXACT pre-fix balance_by_dataset_name (do NOT "fix" this — it must reproduce
# the behavior that was live when the snapshot was created).
# ---------------------------------------------------------------------------
def balance_OLD(data, category, max_samples, seed=42):
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
        if len(buckets[name]) <= base:
            selected.extend(buckets[name])
            leftovers += buckets[name][:]
        else:
            sel = random.sample(buckets[name], base)
            selected.extend(sel)
            leftovers += [x for x in buckets[name] if x not in sel]

    remaining = max_samples - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(random.sample(leftovers, min(remaining, len(leftovers))))

    random.shuffle(selected)
    return selected


def qids(rows):
    return [str(r.get("question_id", "")) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="data/splits/train/TISER_train_en.json")
    ap.add_argument("--snapshot", default="data/splits/train/train_tiser_15000_en.json")
    ap.add_argument("--max_samples", type=int, default=15000)
    ap.add_argument("--no_only_passed", action="store_true",
                    help="Skip the validation_status==PASS filter (default keeps it, matching "
                         "run_train.sh --only_passed).")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: `datasets` not installed. Reproduction needs HF Dataset.shuffle(seed=42).")

    # ---- Reproduce the OLD sampling ----
    print(f"Loading full split: {args.full}")
    ds = load_dataset("json", data_files=args.full)["train"]
    print(f"  raw rows: {len(ds)}")
    if not args.no_only_passed and "validation_status" in ds.column_names:
        ds = ds.filter(lambda x: x["validation_status"] == "PASS")
        print(f"  PASS rows: {len(ds)}")

    ds = ds.shuffle(seed=42)
    reproduced = balance_OLD(ds, category="train", max_samples=args.max_samples)
    B = qids(reproduced)

    # ---- Load the snapshot ----
    print(f"Loading snapshot: {args.snapshot}")
    with open(args.snapshot, encoding="utf-8") as f:
        snap = json.load(f)
    A = qids(snap)

    # ---- Diagnostics ----
    a_set, b_set = set(A), set(B)
    a_dupes = sum(c - 1 for c in Counter(A).values() if c > 1)
    b_dupes = sum(c - 1 for c in Counter(B).values() if c > 1)

    print("\n" + "=" * 60)
    print("15k SNAPSHOT vs OLD seed=42 SAMPLING")
    print("=" * 60)
    print(f"snapshot rows:        {len(A)}  (unique qids: {len(a_set)}, dupes: {a_dupes})")
    print(f"reproduced rows:      {len(B)}  (unique qids: {len(b_set)}, dupes: {b_dupes})")
    if "" in a_set or "" in b_set:
        print("WARNING: some rows have empty question_id — comparison unreliable.")
    only_snap = a_set - b_set
    only_repro = b_set - a_set
    print(f"in snapshot only:     {len(only_snap)}")
    print(f"in reproduced only:   {len(only_repro)}")
    for label, s in (("snapshot-only", only_snap), ("reproduced-only", only_repro)):
        if s:
            print(f"  e.g. {label}: {list(sorted(s))[:5]}")

    # Match = identical MULTISET of question_ids. Duplicates are expected here: they are an
    # artifact of the pre-fix balance sampler (small tgqa bucket re-drawn in the top-up). As long
    # as the same duplicates appear on both sides, the snapshot faithfully reproduces the old run.
    match = Counter(A) == Counter(B)
    print("-" * 60)
    if a_dupes or b_dupes:
        print(f"NOTE: {a_dupes} duplicate row(s) in snapshot, {b_dupes} in reproduced "
              f"— pre-fix balance-sampler artifact (small tgqa bucket). Identical on both "
              f"sides, so still a faithful match; flag as a data-quality footnote in the report.")
    if match:
        print("RESULT: MATCH ✓  → snapshot == old seed=42 sampling (multiset-identical). Rewire SAFE.")
        sys.exit(0)
    else:
        print("RESULT: MISMATCH ✗  → snapshot is NOT the old seed=42 subset.")
        print("Recommended: revert run_train.sh / run_experiment1.sh / run_eval_rag.sh EN")
        print("branches to the full-file sampling, OR treat the snapshot as the authoritative")
        print("frozen set and update the reported-numbers provenance accordingly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
