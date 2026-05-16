#!/usr/bin/env python3
"""
Inference / Evaluation Script for TISER Multilingual Evaluation — Class-Based Refactor

Per-sample prompt template lookup with English fallback.
Reports EM, normalized EM, soft EM, token F1, chrF, and english-leak rate.
Per-language and per-dataset breakdowns emitted automatically.
"""

import argparse
import gc
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

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
    PreTrainedTokenizer,
)

from utils.io_gpu import balance_by_dataset_name, load_prompt_for_lang
from utils.metrics import calculate_metrics, clean_output, extract_answer
from utils.prompt import build_prompt



# ─────────────────────────────────────────────
# Stopping Criterion
# ─────────────────────────────────────────────

class AnswerTagStoppingCriteria(StoppingCriteria):
    """Stops generation as soon as </answer> appears in the decoded tail."""

    STOP_STRING = "</answer>"

    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_len: int, lookback: int = 40):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.lookback = lookback

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        gen_ids = input_ids[0][self.prompt_len:]
        if len(gen_ids) < 5:
            return False
        tail = gen_ids[-self.lookback:]
        decoded = self.tokenizer.decode(tail, skip_special_tokens=True)
        return self.STOP_STRING in decoded


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────

class ModelLoader:
    """Loads a 4-bit quantised base model and optionally attaches a LoRA adapter."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    # ------------------------------------------------------------------
    def _compute_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # ------------------------------------------------------------------
    def _bnb_config(self, dtype: torch.dtype) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # ------------------------------------------------------------------
    def load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # ------------------------------------------------------------------
    def load_model(self):
        print(f"Loading model: {self.args.base_model}")
        dtype = self._compute_dtype()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            quantization_config=self._bnb_config(dtype),
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )

        if self.args.adapter_dir:
            print(f"Loading adapter: {self.args.adapter_dir}")
            model = PeftModel.from_pretrained(base_model, self.args.adapter_dir)
        else:
            print("No adapter — using base model only.")
            model = base_model

        model.eval()
        return model


# ─────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────

class EvalDataLoader:
    """Loads, optionally filters, and returns the evaluation data as a plain list."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    # ------------------------------------------------------------------
    def load(self) -> List[Dict]:
        dataset = load_dataset("json", data_files=self.args.test_file)["train"]

        if self.args.only_passed and "validation_status" in dataset.column_names:
            dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")

        if self.args.max_eval_samples:
            dataset = balance_by_dataset_name(
                dataset.shuffle(seed=42),
                category="test",
                max_samples=self.args.max_eval_samples,
            )

        data_list = [dict(x) for x in dataset]

        langs = [s.get("language") for s in data_list]
        if any(langs):
            counts = Counter(l or "MISSING" for l in langs)
            print(f"Language distribution: {dict(counts)}")

        print(f"Samples loaded: {len(data_list)}")
        return data_list


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class ResponseGenerator:
    """Wraps batch and iterative generation strategies."""

    def __init__(self, model, tokenizer: PreTrainedTokenizer, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    # ------------------------------------------------------------------
    def _generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        n_tokens = max_new_tokens or self.args.max_new_tokens
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        prompt_len = inputs.input_ids.shape[1]
        stopping = StoppingCriteriaList([
            AnswerTagStoppingCriteria(self.tokenizer, prompt_len=prompt_len)
        ])

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                # StoppingCriteria requires batch size == 1 to be reliable
                stopping_criteria=stopping if inputs.input_ids.shape[0] == 1 else None,
            )

        return self.tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    def _generate_iterative(self, prompts: List[str]) -> List[str]:
        responses = self._generate_batch(prompts)

        for i in range(len(responses)):
            ext = 0
            while "</answer>" not in responses[i] and ext < self.args.max_extensions:
                cont_prompt = prompts[i] + responses[i]
                cont_inputs = self.tokenizer(
                    cont_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(self.model.device)

                prompt_len = cont_inputs.input_ids.shape[1]
                stopping = StoppingCriteriaList([
                    AnswerTagStoppingCriteria(self.tokenizer, prompt_len=prompt_len)
                ])

                with torch.inference_mode():
                    out = self.model.generate(
                        **cont_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        stopping_criteria=stopping,
                    )

                new_text = self.tokenizer.decode(
                    out[0][prompt_len:], skip_special_tokens=True
                )
                responses[i] += new_text
                ext += 1
                if "</answer>" in new_text:
                    break

        return responses

    # ------------------------------------------------------------------
    def generate(self, prompts: List[str]) -> List[str]:
        if self.args.strategy == "iterative":
            return self._generate_iterative(prompts)
        return self._generate_batch(prompts)


# ─────────────────────────────────────────────
# Metrics Aggregator
# ─────────────────────────────────────────────

METRIC_NAMES = ["em", "norm_em", "soft_em", "f1", "chrf", "english_leak"]


class MetricsAggregator:
    """Accumulates per-sample metrics and formats summary reports."""

    def __init__(self):
        self.overall = self._empty_bucket()
        self.by_lang: Dict[str, Dict] = defaultdict(self._empty_bucket)
        self.by_dataset: Dict[str, Dict] = defaultdict(self._empty_bucket)

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_bucket() -> Dict:
        return {m: 0.0 for m in METRIC_NAMES} | {"n": 0}

    # ------------------------------------------------------------------
    @staticmethod
    def _add(bucket: Dict, row: Dict) -> None:
        for m in METRIC_NAMES:
            bucket[m] += row.get(m, 0)
        bucket["n"] += 1

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(bucket: Dict) -> str:
        n = bucket["n"]
        if n == 0:
            return "(empty)"
        return (
            f"F1={bucket['f1']/n:.4f}  chrF={bucket['chrf']/n:.2f}  "
            f"NormEM={bucket['norm_em']/n:.4f}  EM={bucket['em']/n:.4f}  "
            f"SoftEM={bucket['soft_em']/n:.4f}  "
            f"EngLeak={bucket['english_leak']/n:.4f}  N={n}"
        )

    # ------------------------------------------------------------------
    def update(self, result: Dict) -> None:
        self._add(self.overall, result)
        self._add(self.by_lang[result.get("language", "unknown")], result)
        self._add(self.by_dataset[result.get("dataset_name", "unknown")], result)

    # ------------------------------------------------------------------
    def log_step(self, log_every: int) -> None:
        n = self.overall["n"]
        if n % log_every == 0:
            tqdm.write(
                f"Step {n} | "
                f"F1={self.overall['f1']/n:.4f}  "
                f"chrF={self.overall['chrf']/n:.2f}  "
                f"NormEM={self.overall['norm_em']/n:.4f}  "
                f"EM={self.overall['em']/n:.4f}"
            )

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        sep = "=" * 70
        print(f"\n{sep}\nFINAL OVERALL\n{sep}")
        print(self._fmt(self.overall))

        if len(self.by_lang) > 1:
            print("\nPer-language:")
            for lang in sorted(self.by_lang):
                print(f"  [{lang}] {self._fmt(self.by_lang[lang])}")

        if len(self.by_dataset) > 1:
            print("\nPer-dataset:")
            for ds in sorted(self.by_dataset):
                print(f"  [{ds}] {self._fmt(self.by_dataset[ds])}")

        print(sep)


# ─────────────────────────────────────────────
# Evaluation Pipeline
# ─────────────────────────────────────────────

class EvaluationPipeline:
    """Orchestrates the full inference + evaluation loop."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    # ------------------------------------------------------------------
    def _preview_prompt(self, data_list: List[Dict]) -> None:
        if not data_list:
            return
        first = data_list[0]
        lang = first.get("language", "en")
        prompt = load_prompt_for_lang(self.args.prompt_name, lang)
        print(f"\n--- Sample prompt (lang={lang}, first 300 chars) ---")
        print(build_prompt(first, prompt=prompt)[:300])
        print("---\n")

    # ------------------------------------------------------------------
    def _save_results(self, results: List[Dict]) -> None:
        os.makedirs(os.path.dirname(self.args.output_file) or ".", exist_ok=True)
        with open(self.args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {self.args.output_file}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"max_new_tokens: {self.args.max_new_tokens}, strategy: {self.args.strategy}")

        # 1. Model & tokenizer
        loader = ModelLoader(self.args)
        tokenizer = loader.load_tokenizer()
        model = loader.load_model()

        # 2. Data
        data_list = EvalDataLoader(self.args).load()
        self._preview_prompt(data_list)

        # 3. Generator & aggregator
        generator = ResponseGenerator(model, tokenizer, self.args)
        aggregator = MetricsAggregator()
        results: List[Dict] = []

        # 4. Inference loop
        for i in tqdm(range(0, len(data_list), self.args.batch_size)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            batch = data_list[i : i + self.args.batch_size]
            lang = batch[0].get("language", "en")
            prompt = load_prompt_for_lang(self.args.prompt_name, lang)
            prompts = [build_prompt(ex, prompt=prompt) for ex in batch]

            preds = generator.generate(prompts)

            for j, pred in enumerate(preds):
                full = clean_output("<reasoning>" + pred)
                gold_target = str(batch[j].get("answer", ""))
                gold_en = str(batch[j].get("answer_en", ""))
                sample_lang = batch[j].get("language", "en")
                dataset_name = batch[j].get("dataset_name", "")

                extracted = extract_answer(full)
                metrics = calculate_metrics(extracted, [gold_target], gold_en=gold_en)

                result = {
                    "question_id": batch[j].get("question_id"),
                    "dataset_name": dataset_name,
                    "language": sample_lang,
                    "gold_target": gold_target,
                    "gold_en": gold_en,
                    "extracted": extracted,
                    "model_output": full,
                    **metrics,
                }
                results.append(result)
                aggregator.update(result)
                aggregator.log_step(self.args.log_every)

        # 5. Save & report
        self._save_results(results)
        aggregator.print_summary()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument(
        "--adapter_dir", default=None,
        help="Path to LoRA adapter. Omit for base model only.",
    )
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--strategy", choices=["base", "iterative"], default="iterative")
    parser.add_argument("--max_extensions", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--only_passed", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--prompt_name", type=str, default="tiser_full")
    return parser.parse_args()


if __name__ == "__main__":
    EvaluationPipeline(parse_args()).run()