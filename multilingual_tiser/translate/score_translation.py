#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils.io_gpu import load_json, save_json
from utils.translation_quality import (
    fields_for_sample,
    normalize_text,
    is_number_or_bool,
    exact_score,
    structure_errors,
    english_leftover_penalty,
  
)

def cosine_scores(model, src_texts, tgt_texts, batch_size):
    if not src_texts:
        return []

    src_clean = [normalize_text(x) for x in src_texts]
    tgt_clean = [normalize_text(x) for x in tgt_texts]

    print("Encoding English fields...")
    src_emb = model.encode(
        src_clean,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Encoding translated fields...")
    tgt_emb = model.encode(
        tgt_clean,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.sum(src_emb * tgt_emb, axis=1).tolist()


def build_pairs(data):
    pairs = []
    meta = []

    for i, sample in enumerate(data):
        for field in fields_for_sample(sample):
            src = sample.get(f"{field}_en", "")
            tgt = sample.get(field, "")

            # numeric / boolean answer is scored exactly later
            if field == "answer" and is_number_or_bool(src):
                continue

            if not str(src).strip():
                continue

            pairs.append((src, tgt))
            meta.append((i, field))

    return pairs, meta


def init_score_table(data):
    scores = {}

    for i, sample in enumerate(data):
        scores[i] = {}
        for field in fields_for_sample(sample):
            scores[i][field] = None

    return scores


def score_dataset(data, model_name, threshold, batch_size):
    print(f"Loading scorer model: {model_name}")
    model = SentenceTransformer(model_name)

    pairs, meta = build_pairs(data)
    print(f"Scoring {len(pairs)} text pairs...")

    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]

    sim_scores = cosine_scores(model, src_texts, tgt_texts, batch_size)
    scores = init_score_table(data)

    for (i, field), score in zip(meta, sim_scores):
        scores[i][field] = float(score)

    # Exact scoring for numeric / boolean answers
    for i, sample in enumerate(data):
        src_ans = sample.get("answer_en", "")
        tgt_ans = sample.get("answer", "")

        if "answer" in scores[i] and is_number_or_bool(src_ans):
            scores[i]["answer"] = exact_score(src_ans, tgt_ans)

    report = []
    passed = []
    failed = []

    for i, sample in enumerate(tqdm(data, desc="Validating", dynamic_ncols=True)):
        field_results = {}
        failed_fields = []

        fields = fields_for_sample(sample)

        for field in fields:
            base_score = scores[i].get(field)

            if base_score is None:
                base_score = 0.0

            errors = structure_errors(sample, field)
            penalty = english_leftover_penalty(sample.get(field, ""))

            final_score = max(0.0, float(base_score) - penalty)

            # Hard structural errors cap the score below pass threshold.
            if errors:
                final_score = min(final_score, 0.94)

            field_results[field] = {
                "score": round(final_score, 6),
                "semantic_score": round(float(base_score), 6),
                "penalty": round(float(penalty), 6),
                "errors": errors,
                "passed": final_score >= threshold,
            }

            if final_score < threshold:
                failed_fields.append(field)

        min_score = min(field_results[f]["score"] for f in fields)
        ok = min_score >= threshold

        item = {
            "idx": i,
            "question_id": sample.get("question_id"),
            "dataset_name": sample.get("dataset_name"),
            "fields_scored": fields,
            "min_score": round(min_score, 6),
            "passed": ok,
            "failed_fields": failed_fields,
            "fields": field_results,
        }

        report.append(item)

        if ok:
            passed.append(sample)
        else:
            failed.append(sample)

    return report, passed, failed


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

    field_fail_counts = {}

    for r in failed_items:
        for f in r["failed_fields"]:
            field_fail_counts[f] = field_fail_counts.get(f, 0) + 1

    print()
    print("Failed by field:")
    if not field_fail_counts:
        print("  none")
    else:
        for f, n in sorted(field_fail_counts.items()):
            print(f"  {f}: {n}")

    print()
    print("First failed samples:")
    for r in failed_items[:10]:
        print(
            f"  idx={r['idx']} id={r['question_id']} "
            f"min_score={r['min_score']} failed={r['failed_fields']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--passed_out", default=None)
    ap.add_argument("--failed_out", default=None)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--model_name",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    ap.add_argument(
        "--no_exit_fail",
        action="store_true",
        help="Do not exit with code 1 when failed samples exist.",
    )
    args = ap.parse_args()

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

    print()
    print(f"Report written to: {args.report}")

    if args.passed_out:
        print(f"Passed samples written to: {args.passed_out}")

    if args.failed_out:
        print(f"Failed samples written to: {args.failed_out}")

    if failed and not args.no_exit_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()