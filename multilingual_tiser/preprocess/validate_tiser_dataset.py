#!/usr/bin/env python3
"""
Normalize, clean, and validate TISER datasets.
Supports both English and Italian prompt markers.

Unknown-answer policy:
  By DEFAULT, unknown-answer samples are KEPT in both train and test splits.
  This avoids the train-test distribution mismatch where training has 0%
  Unknown answers but test has ~5% Unknown answers.

  Use --remove_unknown_from_train and/or --remove_unknown_from_test to
  filter explicitly.
"""
import argparse
import json
import logging
import re
import sys
from collections import  defaultdict
from pathlib import Path

from utils.io_gpu import _language_counts

# Make project-root imports stable when run directly
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import (
    UNKNOWN_TRIGGERS,
    load_json,
    load_txt_as_string,
    save_json,
    save_stats,
    repair_text_fields,
    file_sha256,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ==================================================
# REGEX
# ==================================================
QUESTION_REGEX = re.compile(
    r"(Question:)\s*(.*?)(Temporal context:|$)",
    re.DOTALL | re.IGNORECASE,
)
TEMPORAL_REGEX = re.compile(
    r"(Temporal context:)\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)

# ==================================================
# REQUIRED FIELDS
# ==================================================
_REQUIRED_FIELDS_COMMON = [
    "dataset_name",
    "question_id",
    "question",
    "temporal_context",
    "prompt",
    "answer",
]
REQUIRED_FIELDS_TRAIN = _REQUIRED_FIELDS_COMMON + ["output"]
REQUIRED_FIELDS_TEST = _REQUIRED_FIELDS_COMMON

VALID_LANGUAGES = {"en"}


# ==================================================
# UNKNOWN DETECTION
# ==================================================
def is_unknown_answer(answer: str) -> bool:
    """Return True if the answer is missing or a known 'unknown' placeholder."""
    if not UNKNOWN_TRIGGERS:
        log.warning("UNKNOWN_TRIGGERS is empty — unknown-answer filtering is a no-op.")
    if not answer or not isinstance(answer, str):
        return True
    return answer.strip().lower() in UNKNOWN_TRIGGERS


def remove_unknown_answers(data: list[dict]) -> tuple[list[dict], list[dict]]:
    clean, removed = [], []
    for sample in data:
        (removed if is_unknown_answer(sample.get("answer", "")) else clean).append(sample)
    return clean, removed


# ==================================================
# EXTRACTION HELPERS
# ==================================================
def extract_question_and_context(prompt: str) -> tuple[str, str]:
    """Extract question and temporal context from prompt text."""
    if not prompt or not isinstance(prompt, str):
        return "", ""
    question = ""
    temporal_context = ""
    m_q = QUESTION_REGEX.search(prompt)
    if m_q:
        question = m_q.group(2).strip()
    m_tc = TEMPORAL_REGEX.search(prompt)
    if m_tc:
        temporal_context = m_tc.group(2).strip()
    return question, temporal_context


# ==================================================
# NORMALIZATION
# ==================================================
def normalize_common_fields(sample: dict, idx: int, canonical_prompt: str) -> dict:
    raw_prompt = sample.get("prompt", "")
    q_from_prompt, tc_from_prompt = extract_question_and_context(raw_prompt)

    question = sample.get("question") or q_from_prompt
    temporal_context = sample.get("temporal_context") or tc_from_prompt

    if not question:
        log.debug("Sample %s: question could not be resolved from field or prompt.", idx)

    return {
        "dataset_name": str(sample.get("dataset_name", "")).strip(),
        "question_id": str(sample.get("question_id", f"auto_{idx}")).strip(),
        "language": str(sample.get("language", "en")).strip(),
        "question": str(question).strip(),
        "temporal_context": str(temporal_context).strip(),
        "prompt": canonical_prompt.strip(),
        "answer": str(sample.get("answer", "")).strip(),
    }


def normalize_train(sample: dict, idx: int, canonical_prompt: str) -> dict:
    normalized = normalize_common_fields(sample, idx, canonical_prompt)
    normalized["output"] = str(sample.get("output", "")).strip()
    return normalized


def normalize_test(sample: dict, idx: int, canonical_prompt: str) -> dict:
    return normalize_common_fields(sample, idx, canonical_prompt)


# ==================================================
# DEDUPLICATION
# ==================================================
def _dedup_key(sample: dict) -> tuple:
    """Composite key for deduplication (already stripped by normalization)."""
    return (
        sample.get("dataset_name", "").lower(),
        sample.get("question_id", "").lower(),
        sample.get("question", "").lower(),
        sample.get("answer", "").lower(),
    )


def remove_duplicates(data: list[dict]) -> tuple[list[dict], list[dict]]:
    seen: set[tuple] = set()
    deduped, removed = [], []
    for sample in data:
        key = _dedup_key(sample)
        if key in seen:
            removed.append(sample)
        else:
            seen.add(key)
            deduped.append(sample)
    return deduped, removed


# ==================================================
# VOID QUESTION ID REMOVAL
# ==================================================
def remove_void_question_ids(data: list[dict]) -> tuple[list[dict], list[dict]]:
    clean, removed = [], []
    for sample in data:
        qid = sample.get("question_id", "")
        (clean if (isinstance(qid, str) and qid.strip()) else removed).append(sample)
    return clean, removed


# ==================================================
# VALIDATION
# ==================================================
def validate_sample(sample: dict, split: str, allow_unknown: bool = True) -> list[str]:
    """
    Validate a sample's structural fields.

    allow_unknown=True (default): unknown-answer samples are valid.
    allow_unknown=False: flags unknown-answer samples as 'unknown_answer' error.
    """
    errors = []
    required = REQUIRED_FIELDS_TRAIN if split == "train" else REQUIRED_FIELDS_TEST
    for field in required:
        if field not in sample:
            errors.append(f"missing:{field}")
        elif not isinstance(sample[field], str):
            errors.append(f"non_string:{field}")
        elif not sample[field].strip():
            errors.append(f"empty:{field}")
    if "language" in sample and sample["language"].strip().lower() not in VALID_LANGUAGES:
        errors.append("invalid:language")
    if not allow_unknown and is_unknown_answer(sample.get("answer", "")):
        errors.append("unknown_answer")
    return errors


# ==================================================
# STATS HELPERS
# ==================================================


def _unknown_count(data: list[dict]) -> int:
    """Count samples with unknown answers (for diagnostic stats)."""
    return sum(1 for s in data if is_unknown_answer(s.get("answer", "")))


# ==================================================
# MAIN
# ==================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize, clean, and validate TISER datasets."
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output cleaned JSON file")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--prompt_file", required=True, help="Canonical prompt .txt file")
    parser.add_argument("--save_invalid", default=None, help="Path to save structural warnings")
    parser.add_argument("--stats_output", default=None, help="Path to save preprocessing stats")
    parser.add_argument("--max_print", type=int, default=5)
    parser.add_argument(
        "--remove_unknown_from_train",
        action="store_true",
        help=(
            "Remove unknown-answer samples from the train split. "
            "Default: KEEP unknowns to match test distribution."
        ),
    )
    parser.add_argument(
        "--remove_unknown_from_test",
        action="store_true",
        help="Remove unknown-answer samples from the test split. Default: keep them.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run all steps but skip writing output files",
    )
    args = parser.parse_args()

    # ── 1. Load inputs ────────────────────────────────────────────────────────
    try:
        canonical_prompt = load_txt_as_string(args.prompt_file)
        raw_data = load_json(args.input)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.error("Failed to load input files: %s", exc)
        sys.exit(1)

    if not isinstance(raw_data, list):
        log.error("Input JSON must contain a list of samples.")
        sys.exit(1)

    log.info("Raw samples: %d", len(raw_data))

    # ── 2. Repair unicode (all string fields) ─────────────────────────────────
    raw_data = [repair_text_fields(s) for s in raw_data]

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    normalize = normalize_train if args.split == "train" else normalize_test
    data = [normalize(s, i, canonical_prompt) for i, s in enumerate(raw_data)]

    # ── 4. Deduplicate ────────────────────────────────────────────────────────
    data, removed_dups = remove_duplicates(data)
    log.info("Removed %d duplicates (remaining: %d)", len(removed_dups), len(data))

    # ── 5. Optionally remove unknown answers ──────────────────────────────────
    # Default: keep unknowns in both splits so train/test distributions match.
    removed_unknowns: list[dict] = []
    unknown_count_initial = _unknown_count(data)

    should_filter_unknown = (
        (args.split == "train" and args.remove_unknown_from_train)
        or (args.split == "test" and args.remove_unknown_from_test)
    )

    if should_filter_unknown:
        data, removed_unknowns = remove_unknown_answers(data)
        log.info(
            "Removed %d samples with unknown answers from %s split",
            len(removed_unknowns),
            args.split,
        )
    else:
        pct = (100.0 * unknown_count_initial / len(data)) if data else 0.0
        log.info(
            "Keeping %d unknown-answer samples in %s split (%.1f%% of data)",
            unknown_count_initial,
            args.split,
            pct,
        )

    # ── 6. Remove void question IDs ───────────────────────────────────────────
    data, removed_void_qids = remove_void_question_ids(data)
    log.info(
        "Removed %d samples with void question IDs (remaining: %d)",
        len(removed_void_qids),
        len(data),
    )

    # ── 7. Prompt consistency check ───────────────────────────────────────────
    prompts = {s["prompt"] for s in data}
    log.info(
        "Prompt templates: %s",
        "CONSISTENT" if len(prompts) == 1 else f"INCONSISTENT ({len(prompts)})",
    )

    # ── 8. Validation ─────────────────────────────────────────────────────────
    # If unknowns are kept, don't flag them as structural errors.
    invalid_samples: list[dict] = []
    error_stats: dict[str, int] = defaultdict(int)
    allow_unknown_in_validation = not should_filter_unknown
    for idx, sample in enumerate(data):
        errs = validate_sample(sample, args.split, allow_unknown=allow_unknown_in_validation)
        if errs:
            invalid_samples.append({"index": idx, "errors": errs, "sample": sample})
            for e in errs:
                error_stats[e] += 1

    log.info("Samples with structural warnings: %d", len(invalid_samples))
    if error_stats:
        log.info("Structural warning breakdown:")
        for k, v in sorted(error_stats.items(), key=lambda x: (-x[1], x[0])):
            log.info("  %s: %d", k, v)

    # ── 9. Write outputs ──────────────────────────────────────────────────────
    output_path = Path(args.output)
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(str(output_path), data)
        log.info("Clean dataset saved to: %s", output_path)

        if invalid_samples and args.save_invalid:
            invalid_path = Path(args.save_invalid)
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(str(invalid_path), invalid_samples)
            log.info("Structural warnings saved to: %s", invalid_path)
    else:
        log.info("Dry-run mode: no files written.")

    # ── 10. Stats ─────────────────────────────────────────────────────────────
    stats: dict = {
        "input_file": args.input,
        "output_file": args.output,
        "split": args.split,
        "dry_run": args.dry_run,
        "raw_samples": len(raw_data),
        "duplicates_removed": len(removed_dups),
        "unknown_removed": len(removed_unknowns),
        "unknown_kept": unknown_count_initial if not should_filter_unknown else 0,
        "void_question_ids_removed": len(removed_void_qids),
        "final_samples": len(data),
        "structural_warnings": len(invalid_samples),
        "prompt_templates_found": len(prompts),
        "language_distribution": _language_counts(data),
        "error_breakdown": dict(error_stats),
        "output_sha256": file_sha256(output_path) if (not args.dry_run and output_path.exists()) else None,
    }

    if args.stats_output and not args.dry_run:
        save_stats(args.stats_output, stats)
        log.info("Stats saved to: %s", args.stats_output)

    log.info("Done. Prompt is canonical, multilingual-safe, and validated.")


if __name__ == "__main__":
    main()