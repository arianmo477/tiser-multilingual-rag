#!/usr/bin/env python3
"""
Translation quality validation for TISER multilingual data.

Scores each translated field against its English source using:
  - Cosine similarity of multilingual sentence embeddings (semantic)
  - Exact match for numeric / boolean answers
  - Structural error checks (empty parens, malformed tags)
  - English-leftover penalty (untranslated tokens in target)

Samples pass if the MINIMUM field score >= threshold. Failed samples
are written separately for retranslation or manual inspection.
"""

import argparse
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.io_gpu import load_json, save_json
from utils.translation_quality import (
    english_leftover_penalty,
    exact_score,
    fields_for_sample,
    is_number_or_bool,
    normalize_text,
    structure_errors,
)


# =============================================================================
# Scoring
# =============================================================================

def cosine_scores(model, src_texts, tgt_texts, batch_size):
    """Cosine similarity of normalized embeddings — same as dot product."""
    if not src_texts:
        return []

    src_clean = [normalize_text(x) for x in src_texts]
    tgt_clean = [normalize_text(x) for x in tgt_texts]

    print("Encoding English fields...")
    src_emb = model.encode(
        src_clean, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    )

    print("Encoding translated fields...")
    tgt_emb = model.encode(
        tgt_clean, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    )

    return np.sum(src_emb * tgt_emb, axis=1).tolist()


def build_pairs(data):
    """
    Collect (src, tgt) text pairs to score in one batched pass.

    Skips numeric/boolean answers (handled by exact_score later) and
    empty sources. Returns aligned lists of pairs and (sample_idx, field)
    metadata so scores can be routed back after batched encoding.
    """
    pairs, meta = [], []

    for i, sample in enumerate(data):
        for field in fields_for_sample(sample):
            src = sample.get(f"{field}_en", "")
            tgt = sample.get(field, "")

            if field == "answer" and is_number_or_bool(src):
                continue
            if not str(src).strip():
                continue

            pairs.append((src, tgt))
            meta.append((i, field))

    return pairs, meta


def score_field(sample, field, base_score):
    """
    Combine semantic score with structural penalties.

    Returns a dict with the final score, its components, and pass status.
    Hard structural errors cap the score below any reasonable threshold.
    """
    errors = structure_errors(sample, field)
    penalty = english_leftover_penalty(sample.get(field, ""))
    final = max(0.0, float(base_score) - penalty)

    if errors:
        final = min(final, 0.94)  # cap below typical 0.95 threshold

    return {
        "score": round(final, 6),
        "semantic_score": round(float(base_score), 6),
        "penalty": round(float(penalty), 6),
        "errors": errors,
    }


def score_dataset(data, model_name, threshold, batch_size):
    """Score every field in every sample; split into passed/failed."""
    print(f"Loading scorer model: {model_name}")
    model = SentenceTransformer(model_name)

    # ---- Batched semantic scoring ----
    pairs, meta = build_pairs(data)
    print(f"Scoring {len(pairs)} text pairs...")

    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]
    sim_scores = cosine_scores(model, src_texts, tgt_texts, batch_size)

    # ---- Route scores back to (sample, field) grid ----
    scores = {i: {f: None for f in fields_for_sample(s)}
              for i, s in enumerate(data)}
    for (i, field), score in zip(meta, sim_scores):
        scores[i][field] = float(score)

    # ---- Exact scoring overrides for numeric / boolean answers ----
    for i, sample in enumerate(data):
        if "answer" in scores[i] and is_number_or_bool(sample.get("answer_en", "")):
            scores[i]["answer"] = exact_score(
                sample.get("answer_en", ""), sample.get("answer", ""),
            )

    # ---- Per-sample aggregation ----
    report, passed, failed = [], [], []

    for i, sample in enumerate(tqdm(data, desc="Validating", dynamic_ncols=True)):
        fields = fields_for_sample(sample)
        field_results = {}
        failed_fields = []

        for field in fields:
            base = scores[i].get(field) or 0.0
            result = score_field(sample, field, base)
            result["passed"] = result["score"] >= threshold
            field_results[field] = result

            if not result["passed"]:
                failed_fields.append(field)

        min_score = min(r["score"] for r in field_results.values())
        ok = min_score >= threshold

        report.append({
            "idx": i,
            "question_id": sample.get("question_id"),
            "dataset_name": sample.get("dataset_name"),
            "fields_scored": fields,
            "min_score": round(min_score, 6),
            "passed": ok,
            "failed_fields": failed_fields,
            "fields": field_results,
        })

        (passed if ok else failed).append(sample)

    return report, passed, failed


# =============================================================================
# Reporting
# =============================================================================

def print_summary(report, threshold):
    total = len(report)
    failed_items = [r for r in report if not r["passed"]]
    passed_count = total - len(failed_items)

    print("=" * 80)
    print("TRANSLATION QUALITY SUMMARY")
    print("=" * 80)
    print(f"Threshold: {threshold}")
    print(f"Total:     {total}")
    print(f"Passed:    {passed_count}")
    print(f"Failed:    {len(failed_items)}")
    if total:
        print(f"Pass rate: {passed_count / total:.2%}")

    # Which fields fail most often? Guides pipeline debugging.
    field_fail_counts = {}
    for r in failed_items:
        for f in r["failed_fields"]:
            field_fail_counts[f] = field_fail_counts.get(f, 0) + 1

    print("\nFailed by field:")
    if not field_fail_counts:
        print("  none")
    else:
        for f, n in sorted(field_fail_counts.items()):
            print(f"  {f}: {n}")

    print("\nFirst failed samples:")
    for r in failed_items[:10]:
        print(
            f"  idx={r['idx']} id={r['question_id']} "
            f"min_score={r['min_score']} failed={r['failed_fields']}"
        )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Validate translation quality.")
    p.add_argument("--input", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--passed_out", default=None)
    p.add_argument("--failed_out", default=None)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--model_name",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    p.add_argument(
        "--no_exit_fail", action="store_true",
        help="Do not exit with code 1 when failed samples exist.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data = load_json(args.input)
    report, passed, failed = score_dataset(
        data=data,
        model_name=args.model_name,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    save_json(args.report, report)
    if args.passed_out:
        save_json(args.passed_out, passed)
    if args.failed_out:
        save_json(args.failed_out, failed)

    print_summary(report, args.threshold)

    print(f"\nReport written to: {args.report}")
    if args.passed_out:
        print(f"Passed samples written to: {args.passed_out}")
    if args.failed_out:
        print(f"Failed samples written to: {args.failed_out}")

    if failed and not args.no_exit_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()