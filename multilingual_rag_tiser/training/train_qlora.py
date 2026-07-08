#!/usr/bin/env python3
"""
QLoRA Training Script — Class-Based Refactor (8GB VRAM + Validation Safe)

Key features:
- 8GB-friendly defaults
- 4-bit QLoRA
- Validation split support
- Best checkpoint by eval loss
- CPU offload support
- group_by_length enabled
"""

import argparse
import os
from typing import Any, List, Optional, Dict
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.io_gpu import balance_by_dataset_name, load_prompt_for_lang
from utils.prompt import build_prompt

# -------------------------
# OPTIMIZATION FLAGS
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────

@dataclass
class DataCollatorForCausalLMWithPadding:
    """Pads input_ids / attention_mask and aligns labels with -100 fill."""

    tokenizer: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        features_no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(features_no_labels, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ─────────────────────────────────────────────
# Dataset Handler
# ─────────────────────────────────────────────

class QLoRADatasetHandler:
    """Loads, filters, tokenizes and splits the dataset."""

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    def load(self) -> Dataset:
        print(f"Loading dataset from: {self.args.train_file}")
        dataset = load_dataset("json", data_files=self.args.train_file)["train"]

        if self.args.only_passed and "validation_status" in dataset.column_names:
            dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")
            print(f"Filtered to PASS samples: {len(dataset)}")

        if self.args.max_train_samples:
            print(f"Balancing dataset to max {self.args.max_train_samples} samples...")
            dataset = balance_by_dataset_name(
                dataset.shuffle(seed=42),
                category="train",
                max_samples=self.args.max_train_samples,
            )

        if isinstance(dataset, list):
            print("Converting list back to Hugging Face Dataset...")
            dataset = Dataset.from_list(dataset)

        return dataset

    # ------------------------------------------------------------------
    def _preprocess(self, ex: Dict) -> Dict:
        lang = ex.get("language") or "en"
        prompt = load_prompt_for_lang(self.args.prompt_name, lang)
        full_prompt = build_prompt(ex, prompt=prompt)

        if not full_prompt or not full_prompt.strip():
            raise ValueError(
                f"Empty prompt generated with prompt_name={self.args.prompt_name}. "
                f"Example keys: {list(ex.keys())}"
            )

        output = (ex.get("output", "") or ex.get("answer", "") or "").strip()

        prompt_tokens = self.tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        output_tokens = self.tokenizer(
            output + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_tokens + output_tokens
        labels = [-100] * len(prompt_tokens) + output_tokens

        if len(input_ids) > self.args.max_length:
            input_ids = input_ids[: self.args.max_length]
            labels = labels[: self.args.max_length]

        # Drop samples where truncation removed all output tokens
        if all(x == -100 for x in labels):
            return {"input_ids": [], "labels": [], "attention_mask": []}

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _has_valid_labels(example: Dict) -> bool:
        return any(x != -100 for x in example["labels"])

    # ------------------------------------------------------------------
    def tokenize(self, dataset: Dataset) -> Dataset:
        print("Tokenizing dataset...")
        dataset = dataset.map(
            self._preprocess,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

        if len(dataset) == 0:
            raise ValueError("No valid samples remain after preprocessing.")

        return dataset

    # ------------------------------------------------------------------
    def split(self, dataset: Dataset):
        """Returns (train_dataset, eval_dataset | None)."""
        if self.args.validation_split > 0 and len(dataset) > 10:
            splits = dataset.train_test_split(test_size=self.args.validation_split, seed=42)
            train_dataset = splits["train"]
            eval_dataset = splits["test"]

            before = len(eval_dataset)
            eval_dataset = eval_dataset.filter(self._has_valid_labels)
            removed = before - len(eval_dataset)
            if removed:
                print(f"Removed {removed} bad samples from val set")

            if len(eval_dataset) == 0:
                print("WARNING: val set empty after filtering — disabling evaluation")
                eval_dataset = None

            print(f"Train samples:      {len(train_dataset)}")
            print(f"Validation samples: {len(eval_dataset) if eval_dataset else 0}")
            return train_dataset, eval_dataset

        print(f"Train samples: {len(dataset)}")
        print("No validation split used.")
        return dataset, None

    # ------------------------------------------------------------------
    def prepare(self):
        """Full pipeline: load → tokenize → split."""
        raw = self.load()
        tokenized = self.tokenize(raw)
        return self.split(tokenized)


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────

class QLoRAModelBuilder:
    """Loads the base model in 4-bit and wraps it with LoRA adapters."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    # ------------------------------------------------------------------
    @staticmethod
    def _find_lora_targets(model) -> List[str]:
        names = {n.split(".")[-1] for n, _ in model.named_modules()}
        preferred = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        return [m for m in preferred if m in names]

    # ------------------------------------------------------------------
    def _bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    # ------------------------------------------------------------------
    def build(self):
        print("Loading model in 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            device_map="auto",
            max_memory={
                0: f"{self.args.gpu_memory_gb}GiB",
                "cpu": f"{self.args.cpu_memory_gb}GiB",
            },
            quantization_config=self._bnb_config(),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=(self.args.gradient_checkpointing == 1),
        )

        targets = self._find_lora_targets(model)
        print(f"Targeting modules for LoRA: {targets}")
        if not targets:
            raise ValueError("Could not find LoRA target modules in the model.")

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model


# ─────────────────────────────────────────────
# Trainer Builder
# ─────────────────────────────────────────────

class QLoRATrainerBuilder:
    """Assembles the HuggingFace Trainer from model, datasets and args."""

    def __init__(
        self,
        args: argparse.Namespace,
        model,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    # ------------------------------------------------------------------
    def _training_args(self) -> TrainingArguments:
        has_eval = self.eval_dataset is not None
        return TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.lr,
            per_device_train_batch_size=self.args.per_device_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.args.grad_accum,
            bf16=False,
            fp16=True,
            optim="paged_adamw_8bit",
            logging_steps=self.args.logging_steps,
            save_strategy="steps",
            save_steps=self.args.eval_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=self.args.eval_steps if has_eval else None,
            save_total_limit=2,
            gradient_checkpointing=(self.args.gradient_checkpointing == 1),
            dataloader_num_workers=self.args.dataloader_num_workers,
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=has_eval,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    # ------------------------------------------------------------------
    def build(self) -> Trainer:
        return Trainer(
            model=self.model,
            args=self._training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForCausalLMWithPadding(self.tokenizer),
        )


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class QLoRATrainingPipeline:
    """Top-level pipeline: wires everything together and runs training."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    # ------------------------------------------------------------------
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        print(f"Loading tokenizer for: {self.args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    # ------------------------------------------------------------------
    def run(self):
        # 1. Tokenizer
        tokenizer = self._load_tokenizer()

        # 2. Data
        data_handler = QLoRADatasetHandler(self.args, tokenizer)
        train_dataset, eval_dataset = data_handler.prepare()

        # 3. Model
        model = QLoRAModelBuilder(self.args).build()

        # 4. Trainer
        trainer = QLoRATrainerBuilder(
            self.args, model, tokenizer, train_dataset, eval_dataset
        ).build()

        # 5. Train
        print("Starting training...")
        trainer.train()

        # 6. Save
        print(f"Saving model to {self.args.output_dir}")
        model.save_pretrained(self.args.output_dir)
        tokenizer.save_pretrained(self.args.output_dir)
        print("Training complete.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model / Data / Prompt
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--only_passed", action="store_true")
    parser.add_argument("--prompt_name", type=str, default="tiser_full")

    # Hyperparameters (defaults match the documented 8GB-QLoRA config in run_train.sh)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Validation
    parser.add_argument("--validation_split", type=float, default=0.1)

    # Memory / Offload
    parser.add_argument("--gpu_memory_gb", type=int, default=7)
    parser.add_argument("--cpu_memory_gb", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    QLoRATrainingPipeline(parse_args()).run()