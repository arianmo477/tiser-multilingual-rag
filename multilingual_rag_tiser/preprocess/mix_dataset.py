
"""
Mix TISER language datasets with balanced per-dataset totals.

For each `dataset_name` category (story / l2 / l3 / wiki_easy / wiki_hard),
allocate samples evenly across the provided languages. Within each
(dataset, language) cell, pick samples uniformly at random.

TGQA is treated specially: it's smaller than the others, so its total is
capped at what every language can supply, kept evenly divisible across
languages. The remaining budget is redistributed across the other datasets.
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from utils.io_gpu import load_json, save_json


# ============================================================================
# Helpers
# ============================================================================

def sample_key(sample):
    dataset = str(sample.get("dataset_name", "") or "").strip()
    qid = str(sample.get("question_id", "") or "").strip()
    if qid:
        return f"{dataset}::{qid}"
    question = str(sample.get("question", "") or "").strip()
    return f"{dataset}::{question[:200]}"


def infer_lang_from_path(path):
    """Best-effort language detection from filename / directory."""
    path = str(path)
    for lang in ("en", "it", "de", "fr", "fa", "es"):
        if f"_{lang}." in path or f"_{lang}_" in path or f"/{lang}/" in path:
            return lang
    return "unknown"


def group_by_dataset(samples):
    """dataset_name -> list of unique samples. Removes intra-file duplicates."""
    groups = defaultdict(list)
    seen = set()
    for sample in samples:
        key = sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        dataset_name = sample.get("dataset_name", "unknown")
        groups[dataset_name].append(sample)
    return groups


def take_samples(group_for_lang, lang, dataset_name, n):
    bucket = list(group_for_lang[dataset_name])
    random.shuffle(bucket)
    if len(bucket) < n:
        raise ValueError(
            f"Not enough samples for {lang} / {dataset_name}. "
            f"Need {n}, have {len(bucket)}."
        )
    chosen = []
    for sample in bucket[:n]:
        sample = dict(sample)
        sample["language"] = lang
        chosen.append(sample)
    return chosen


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="train")
    parser.add_argument("--dataset1", required=True)
    parser.add_argument("--dataset2", required=True)
    parser.add_argument("--dataset3", default=None)
    parser.add_argument("--dataset4", default=None)
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect provided dataset paths
    paths = [args.dataset1, args.dataset2]
    if args.dataset3 is not None:
        paths.append(args.dataset3)
    if args.dataset4 is not None:
        paths.append(args.dataset4)

    langs = [infer_lang_from_path(p) for p in paths]
    if "unknown" in langs:
        raise ValueError(
            f"Could not infer language for one of the paths. "
            f"Paths: {paths}, Detected: {langs}"
        )
    if len(set(langs)) != len(langs):
        raise ValueError(f"Duplicate languages: {langs}")

    n_langs = len(langs)

    print("=" * 60)
    print(f"Mixing {n_langs}-language dataset with balanced totals")
    print("=" * 60)
    print(f"Max samples: {args.max_samples}")
    print(f"Paths:       {paths}")
    print(f"Languages:   {langs}")
    print(f"Output:      {args.output}")
    print("=" * 60)

    # Load and group each language's data
    raw = []
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        data = load_json(path)
        print(f"Loaded {path}: {len(data)} samples")
        raw.append(data)

    grouped = {lang: group_by_dataset(data) for lang, data in zip(langs, raw)}

    # Intersection of dataset_names — every language must have the category
    dataset_names_sets = [set(grouped[lang].keys()) for lang in langs]
    dataset_names = sorted(set.intersection(*dataset_names_sets))
    if not dataset_names:
        raise ValueError("No shared dataset_names found across all languages.")

    print("=" * 60)
    print("Available samples")
    print("=" * 60)
    for dataset_name in dataset_names:
        counts = ", ".join(
            f"{lang.upper()}={len(grouped[lang][dataset_name])}"
            for lang in langs
        )
        print(f"  {dataset_name}: {counts}")

    # ------------------------------------------------------------
    # Decide per-dataset totals
    # ------------------------------------------------------------
    # TGQA is smaller than the rest; we cap it and redistribute the rest.
    # The TGQA total is forced to be divisible by n_langs so each language
    # gets exactly the same number of TGQA samples (it's an L1-only set).
    tgqa_name = "tgqa_split_train"
    if tgqa_name not in dataset_names:
        # No special TGQA handling needed — just balance everything equally.
        normal_per_dataset = args.max_samples // len(dataset_names)
        dataset_totals = {d: normal_per_dataset for d in dataset_names}
        # Distribute leftover from integer division to first few datasets
        leftover = args.max_samples - normal_per_dataset * len(dataset_names)
        for i in range(leftover):
            dataset_totals[dataset_names[i]] += 1
    else:
        normal_per_dataset = args.max_samples // len(dataset_names)
        tgqa_per_lang = min(
            min(len(grouped[lang][tgqa_name]) for lang in langs),
            normal_per_dataset // n_langs,
        )
        tgqa_total = tgqa_per_lang * n_langs

        other_datasets = [d for d in dataset_names if d != tgqa_name]
        remaining = args.max_samples - tgqa_total
        base = remaining // len(other_datasets)
        extra = remaining % len(other_datasets)

        dataset_totals = {tgqa_name: tgqa_total}
        for i, d in enumerate(other_datasets):
            dataset_totals[d] = base + (1 if i < extra else 0)

    # ------------------------------------------------------------
    # Allocate each dataset's total across languages
    # ------------------------------------------------------------
    # For each dataset, give base = total // N to every language, then
    # give +1 to the first `extra` languages. Rotate which languages get
    # the +1 across datasets to keep the global language distribution close
    # to even.
    allocation = {}
    rotation_offset = 0

    for dataset_name in dataset_names:
        total = dataset_totals[dataset_name]
        base = total // n_langs
        extra = total % n_langs

        per_lang = {}
        for i, lang in enumerate(langs):
            # Languages at positions [offset, offset+1, ..., offset+extra-1]
            # get the +1 — cycled.
            shifted = (i - rotation_offset) % n_langs
            per_lang[lang] = base + (1 if shifted < extra else 0)
        allocation[dataset_name] = per_lang

        # Shift the rotation so the next dataset's +1's start at a different
        # language. Keeps overall language counts close.
        rotation_offset = (rotation_offset + extra) % n_langs

    print("=" * 60)
    print("Allocation plan")
    print("=" * 60)
    for dataset_name in dataset_names:
        per_lang = allocation[dataset_name]
        parts = ", ".join(f"{lang.upper()}={per_lang[lang]}" for lang in langs)
        total = sum(per_lang.values())
        print(f"  {dataset_name}: {parts} (total={total})")

        # Verify each language can supply
        for lang in langs:
            n_needed = per_lang[lang]
            n_have = len(grouped[lang][dataset_name])
            if n_have < n_needed:
                raise ValueError(
                    f"Not enough {lang.upper()} samples for {dataset_name}. "
                    f"Need {n_needed}, have {n_have}."
                )

    # ------------------------------------------------------------
    # Select samples
    # ------------------------------------------------------------
    mixed = []
    for dataset_name in dataset_names:
        for lang in langs:
            n = allocation[dataset_name][lang]
            selected = take_samples(
                group_for_lang=grouped[lang],
                lang=lang,
                dataset_name=dataset_name,
                n=n,
            )
            mixed.extend(selected)

    random.shuffle(mixed)

    if len(mixed) != args.max_samples:
        raise ValueError(f"Expected {args.max_samples}, got {len(mixed)}.")

    # ------------------------------------------------------------
    # Save + report
    # ------------------------------------------------------------
    save_json(args.output, mixed)

    lang_dist = Counter(s["language"] for s in mixed)
    dataset_dist = Counter(s["dataset_name"] for s in mixed)
    joint_dist = Counter((s["language"], s["dataset_name"]) for s in mixed)

    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Total samples: {len(mixed)}")
    print(f"Language distribution: {dict(lang_dist)}")
    print(f"Dataset distribution: {dict(dataset_dist)}")
    print("Joint distribution:")
    for dataset_name in dataset_names:
        parts = ", ".join(
            f"{lang.upper()}={joint_dist[(lang, dataset_name)]}"
            for lang in langs
        )
        print(f"  {dataset_name}: {parts}")
    print(f"Saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()