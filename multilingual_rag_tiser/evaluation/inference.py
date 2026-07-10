#!/usr/bin/env python3
"""
Inference / Evaluation for TISER with optional RAG ablation.

Generation is iterative: generate once, then extend any response that did not
emit </answer>, up to --max_extensions times.

RAG modes (both expected to HURT — reported as negative finding):
  few_shot         Prepend retrieved solved examples as demonstrations.
  context_stuffing Append retrieved temporal contexts to the sample's own.

Without --use_rag: clean baseline evaluation.
"""

import argparse
import gc
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from utils.io_gpu import load_prompt_for_lang
from utils.metrics import calculate_metrics, clean_output, extract_answer
from utils.prompt import build_prompt
from utils.rag_utils import RAGContextBuilder
from utils.sampling import balance_by_dataset_name, balance_by_lang_and_dataset


# =============================================================================
# Generation
# =============================================================================

class AnswerTagStopping(StoppingCriteria):
    """Stop when </answer> appears in the generated tail."""

    def __init__(self, tokenizer, prompt_len, lookback=40):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.lookback = lookback

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        if len(gen_ids) < 5:
            return False
        tail = self.tokenizer.decode(gen_ids[-self.lookback:], skip_special_tokens=True)
        return "</answer>" in tail


def _generate(model, tokenizer, prompts, max_new_tokens, use_stop=True):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]
    stopping = None
    if use_stop and inputs.input_ids.shape[0] == 1:
        stopping = StoppingCriteriaList([AnswerTagStopping(tokenizer, prompt_len)])

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
        )
    return tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)


def generate_iterative(model, tokenizer, prompts, args):
    """Generate, then extend responses that didn't emit </answer>."""
    responses = _generate(model, tokenizer, prompts, args.max_new_tokens)

    for i in range(len(responses)):
        for _ in range(args.max_extensions):
            if "</answer>" in responses[i]:
                break
            extension = _generate(
                model, tokenizer, [prompts[i] + responses[i]], max_new_tokens=256,
            )[0]
            responses[i] += extension

    return responses


# =============================================================================
# Metrics buckets
# =============================================================================

METRIC_NAMES = ["em", "norm_em", "soft_em", "f1", "chrf", "english_leak"]


def empty_bucket():
    return {m: 0.0 for m in METRIC_NAMES} | {"n": 0}


def add_to_bucket(bucket, row):
    for m in METRIC_NAMES:
        bucket[m] += row.get(m, 0)
    bucket["n"] += 1


def fmt_bucket(b):
    n = b["n"]
    if n == 0:
        return "(empty)"
    return (
        f"F1={b['f1']/n:.4f}  chrF={b['chrf']/n:.2f}  "
        f"NormEM={b['norm_em']/n:.4f}  EM={b['em']/n:.4f}  "
        f"SoftEM={b['soft_em']/n:.4f}  EngLeak={b['english_leak']/n:.4f}  N={n}"
    )


# =============================================================================
# Setup helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TISER inference with optional RAG.")

    # Model
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", default=None)

    # Data
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument(
        "--balance",
        choices=["dataset", "lang_dataset"],
        default="dataset",
        help="How to balance the --max_eval_samples subset: across dataset_name "
             "only (default), or across every (language, dataset_name) cell — "
             "use lang_dataset for mixed-language test files.",
    )
    p.add_argument(
        "--only_passed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only validation_status==PASS samples. "
             "Use --no-only_passed to evaluate on the full set.",
    )

    # Generation (iterative: extend until </answer> or max_extensions reached)
    p.add_argument("--max_extensions", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--prompt_name", type=str, default="tiser_full")
    p.add_argument("--log_every", type=int, default=1)

    # RAG (ablation — expected to hurt)
    p.add_argument("--use_rag", action="store_true")
    p.add_argument("--rag_mode", choices=["few_shot", "context_stuffing"], default="few_shot")
    p.add_argument("--rag_index_dir", type=str, default=None)
    p.add_argument("--rag_top_k", type=int, default=1)
    p.add_argument("--rag_min_score", type=float, default=0.60)
    p.add_argument("--rag_model_name", type=str,
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    args = p.parse_args()

    if args.use_rag:
        if not args.rag_index_dir:
            raise ValueError("--use_rag requires --rag_index_dir")
        if not Path(args.rag_index_dir).exists():
            raise FileNotFoundError(f"RAG index not found: {args.rag_index_dir}")

    return args


def print_config(args):
    print(f"Max new tokens: {args.max_new_tokens} (+{args.max_extensions} extensions max)")
    if args.use_rag:
        print(f"RAG mode:       {args.rag_mode} (ablation)")
        print(f"RAG index:      {args.rag_index_dir}")
        print(f"RAG top-k:      {args.rag_top_k}")
        print(f"RAG min score:  {args.rag_min_score}")
    else:
        print("RAG:            disabled (clean baseline)")


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model: {args.base_model}")
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    if args.adapter_dir:
        print(f"Loading adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        print("No adapter — base model only.")
        model = base

    model.eval()
    return model, tokenizer


def load_data(args):
    ds = load_dataset("json", data_files=args.test_file)["train"]
    if args.only_passed and "validation_status" in ds.column_names:
        ds = ds.filter(lambda x: x["validation_status"] == "PASS")
    if args.max_eval_samples:
        balance_fn = (
            balance_by_lang_and_dataset
            if args.balance == "lang_dataset"
            else balance_by_dataset_name
        )
        ds = balance_fn(
            ds.shuffle(seed=42), category="test", max_samples=args.max_eval_samples,
        )
    data = [dict(x) for x in ds]

    lang_counts = Counter(s.get("language") for s in data)
    if lang_counts:
        print(f"\nLanguage distribution: {dict(lang_counts)}")
    print(f"Samples: {len(data)}")
    return data


def apply_rag(ex, base_prompt, rag):
    """
    Returns (prompt, sample, hit_flag) after applying RAG augmentation.
    Called only when rag is not None.
    """
    few_shot_prefix, modified_sample = rag.build(ex)

    if rag.mode == "few_shot" and few_shot_prefix:
        return few_shot_prefix + base_prompt, modified_sample, True

    if rag.mode == "context_stuffing" and (
        modified_sample.get("temporal_context") != ex.get("temporal_context")
    ):
        return base_prompt, modified_sample, True

    return base_prompt, ex, False


# =============================================================================
# Reporting
# =============================================================================

def print_summary(overall, results, args, rag, rag_hits, rag_total):
    print("\n" + "=" * 70)
    print("FINAL OVERALL")
    print("=" * 70)
    print(fmt_bucket(overall))

    if rag is not None:
        hit_pct = 100 * rag_hits / rag_total if rag_total else 0
        print(
            f"\nRAG ({args.rag_mode}): {rag_hits}/{rag_total} samples retrieved "
            f"above min_score={args.rag_min_score} ({hit_pct:.1f}%)"
        )
        if hit_pct < 20:
            print("  ↳ Low hit rate — min_score may be too high.")
        elif hit_pct > 80:
            print("  ↳ High hit rate — noise present for most samples.")

    by_lang = defaultdict(empty_bucket)
    for r in results:
        add_to_bucket(by_lang[r["language"]], r)
    if len(by_lang) > 1:
        print("\nPer-language:")
        for lang in sorted(by_lang):
            print(f"  [{lang}] {fmt_bucket(by_lang[lang])}")

    by_ds = defaultdict(empty_bucket)
    for r in results:
        add_to_bucket(by_ds[r.get("dataset_name", "unknown")], r)
    if len(by_ds) > 1:
        print("\nPer-dataset:")
        for ds in sorted(by_ds):
            print(f"  [{ds}] {fmt_bucket(by_ds[ds])}")

    print("=" * 70)
    print(f"Results saved to: {args.output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    print_config(args)

    model, tokenizer = load_model(args)

    rag = None
    if args.use_rag:
        rag = RAGContextBuilder(
            index_dir=args.rag_index_dir,
            model_name=args.rag_model_name,
            top_k=args.rag_top_k,
            min_score=args.rag_min_score,
            mode=args.rag_mode,
        )

    data = load_data(args)

    results = []
    overall = empty_bucket()
    rag_total = 0
    rag_hits = 0

    for i in tqdm(range(0, len(data), args.batch_size)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        batch = data[i:i + args.batch_size]
        lang = batch[0].get("language", "en")
        base_prompt = load_prompt_for_lang(args.prompt_name, lang)

        prompts = []
        for ex in batch:
            if rag is not None:
                rag_total += 1
                prompt, sample, hit = apply_rag(ex, base_prompt, rag)
                if hit:
                    rag_hits += 1
            else:
                prompt, sample = base_prompt, ex
            prompts.append(build_prompt(sample, prompt=prompt))

        decoded = generate_iterative(model, tokenizer, prompts, args)

        for j, pred in enumerate(decoded):
            ex = batch[j]
            full = clean_output("<reasoning>" + pred)
            extracted = extract_answer(full)
            metrics = calculate_metrics(
                extracted,
                [str(ex.get("answer", ""))],
                gold_en=str(ex.get("answer_en", "")),
            )
            add_to_bucket(overall, metrics)

            results.append({
                "question_id": ex.get("question_id"),
                "dataset_name": ex.get("dataset_name", ""),
                "language": ex.get("language", "en"),
                "gold_target": str(ex.get("answer", "")),
                "gold_en": str(ex.get("answer_en", "")),
                "extracted": extracted,
                "model_output": full,
                "rag_used": rag is not None,
                "rag_mode": args.rag_mode if rag is not None else None,
                **metrics,
            })

            if overall["n"] % args.log_every == 0:
                n = overall["n"]
                tqdm.write(
                    f"Step {n} | F1={overall['f1']/n:.4f}  "
                    f"chrF={overall['chrf']/n:.2f}  "
                    f"NormEM={overall['norm_em']/n:.4f}  "
                    f"EM={overall['em']/n:.4f}"
                )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary(overall, results, args, rag, rag_hits, rag_total)


if __name__ == "__main__":
    main()