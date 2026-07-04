#!/usr/bin/env python3
"""
Aggregate the four cells of the RAG ablation (Experiment 1) into one table.

Experiment 1 asks: does retrieval and fine-tuning inject the *same* task
knowledge? We compare few-shot RAG on the base (non-fine-tuned) model against
the fine-tuned model, with and without RAG:

    base_norag   base model,       RAG off
    base_rag     base model,       RAG on  (few-shot exemplars)
    ft_norag     fine-tuned model, RAG off   (your main result)
    ft_rag       fine-tuned model, RAG on

Expected finding: RAG helps the base model a lot (it supplies the format /
reasoning pattern the base model never learned) but adds little or nothing to
the fine-tuned model — i.e. retrieval and fine-tuning are substitutes.

Each input is a results JSON produced by inference.py (a list of per-sample
dicts carrying em/norm_em/soft_em/f1/chrf/english_leak + question_id).

Usage:
    python multilingual_rag_tiser/evaluation/compare_rag_ablation.py \\
        base_norag=experiments/.../gen_base_norag.json \\
        base_rag=experiments/.../gen_base_rag.json \\
        ft_norag=experiments/.../gen_ft_norag.json \\
        ft_rag=experiments/.../gen_ft_rag.json \\
        --per_dataset \\
        --out experiments/experiment1_rag_ablation/summary.json

Order does not matter; any subset of the four labels is accepted. If the label
set is {base_norag, base_rag, ft_norag, ft_rag} the RAG deltas are printed too.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = ["f1", "chrf", "norm_em", "em", "soft_em", "english_leak"]
# Canonical column order for the printed table.
KNOWN_ORDER = ["base_norag", "base_rag", "ft_norag", "ft_rag"]


def load_cell(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of result dicts, got {type(rows)}")
    return rows


def mean_metrics(rows):
    n = len(rows)
    if n == 0:
        return {m: 0.0 for m in METRICS} | {"n": 0}
    out = {m: sum(float(r.get(m, 0)) for r in rows) / n for m in METRICS}
    out["n"] = n
    return out


def parse_pairs(pairs):
    cells = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Expected label=path, got: {p!r}")
        label, path = p.split("=", 1)
        cells[label.strip()] = path.strip()
    return cells


def align_on_question_id(cell_rows):
    """Restrict every cell to the question_ids present in ALL cells, so the
    comparison is over an identical sample set. Returns (aligned, common_ids,
    warnings)."""
    id_sets = {}
    for label, rows in cell_rows.items():
        ids = [r.get("question_id") for r in rows]
        id_sets[label] = set(ids)

    common = set.intersection(*id_sets.values()) if id_sets else set()
    common.discard(None)

    warnings = []
    for label, ids in id_sets.items():
        missing = len(ids - common)
        if missing:
            warnings.append(
                f"  {label}: {missing} sample(s) not shared by all cells "
                f"(dropped from aligned comparison)"
            )

    aligned = {}
    for label, rows in cell_rows.items():
        seen = set()
        kept = []
        for r in rows:
            qid = r.get("question_id")
            if qid in common and qid not in seen:
                kept.append(r)
                seen.add(qid)
        aligned[label] = kept
    return aligned, common, warnings


def order_labels(labels):
    known = [l for l in KNOWN_ORDER if l in labels]
    extra = sorted(l for l in labels if l not in KNOWN_ORDER)
    return known + extra


def fmt_row(label, m):
    return (
        f"{label:<12} "
        f"F1={m['f1']:.4f}  chrF={m['chrf']:.2f}  "
        f"NormEM={m['norm_em']:.4f}  EM={m['em']:.4f}  "
        f"SoftEM={m['soft_em']:.4f}  EngLeak={m['english_leak']:.4f}  N={m['n']}"
    )


def print_table(title, per_cell, labels):
    print(f"\n{title}")
    print("-" * len(title))
    for label in labels:
        print(fmt_row(label, per_cell[label]))


def print_deltas(per_cell, labels):
    """Print RAG effect (rag - norag) for base and ft, if both present."""
    pairs = [("base", "base_norag", "base_rag"),
             ("ft", "ft_norag", "ft_rag")]
    printed_header = False
    for name, no_rag, rag in pairs:
        if no_rag in labels and rag in labels:
            if not printed_header:
                print("\nRAG effect (rag - norag), positive = RAG helps")
                print("-" * 46)
                printed_header = True
            for m in ["f1", "em", "norm_em", "chrf"]:
                d = per_cell[rag][m] - per_cell[no_rag][m]
                sign = "+" if d >= 0 else ""
                print(f"  [{name:<4}] {m:<8} {sign}{d:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="label=path entries")
    ap.add_argument("--per_dataset", action="store_true",
                    help="Also break results down by dataset_name.")
    ap.add_argument("--no_align", action="store_true",
                    help="Skip question_id alignment (compare raw files as-is).")
    ap.add_argument("--out", default=None, help="Optional path to save a JSON summary.")
    args = ap.parse_args()

    cells = parse_pairs(args.pairs)
    for label, path in cells.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"[{label}] missing results file: {path}")

    cell_rows = {label: load_cell(path) for label, path in cells.items()}
    labels = order_labels(list(cell_rows))

    if args.no_align:
        aligned = cell_rows
        common_ids = None
        warnings = []
    else:
        aligned, common_ids, warnings = align_on_question_id(cell_rows)

    if warnings:
        print("Alignment warnings:")
        for w in warnings:
            print(w)

    overall = {label: mean_metrics(aligned[label]) for label in labels}

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — RAG ABLATION (base vs fine-tuned, RAG on/off)")
    print("=" * 70)
    if common_ids is not None:
        print(f"Aligned on {len(common_ids)} shared question_id(s).")
    print_table("Overall", overall, labels)
    print_deltas(overall, labels)

    per_dataset = {}
    if args.per_dataset:
        ds_names = set()
        for rows in aligned.values():
            ds_names.update(r.get("dataset_name", "unknown") for r in rows)
        for ds in sorted(ds_names):
            per_cell = {}
            for label in labels:
                subset = [r for r in aligned[label]
                          if r.get("dataset_name", "unknown") == ds]
                per_cell[label] = mean_metrics(subset)
            per_dataset[ds] = per_cell
            print_table(f"Dataset: {ds}", per_cell, labels)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "files": cells,
            "aligned_n": len(common_ids) if common_ids is not None else None,
            "overall": overall,
            "per_dataset": per_dataset,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {out}")

    print("=" * 70)


if __name__ == "__main__":
    main()
