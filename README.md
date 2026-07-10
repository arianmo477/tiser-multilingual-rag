# tiser-multilingual

Multilingual temporal reasoning with a small language model — a two-part extension of [TISER](https://aclanthology.org/2025.acl-long.1358/) (Bazaga et al., ACL 2025):

1. **Small-model adaptation** — Qwen2.5-3B-Instruct + QLoRA on a single 8 GB GPU, matching the spirit of the original paper's reasoning pipeline without requiring 40+ GB of VRAM.

2. **Multilingual extension** — an NLLB-based translation pipeline that produces structurally faithful Italian, German, and French training data, enabling a single fine-tuned model to reason temporally across languages.

The original TISER paper uses Qwen2.5-7B and Mistral-7B on English only. This project reaches strong English-only performance with a 3 B model and extends the same temporal-reasoning pipeline to multilingual EN · IT · DE · FR training. A third component adds **retrieval-augmented generation (RAG) as a controlled ablation**, testing when retrieval helps a benchmark whose samples are already self-contained.

---

## Results

Most evaluations use approximately **500 samples** across the five TISER categories: `tgqa`, `tempreason_l2`, `tempreason_l3`, `timeqa_easy`, and `timeqa_hard`. Some runs are exactly balanced with 100 samples per category, while others use a near-balanced split depending on the available filtered samples.

### English-only model — EN (15 000 training samples)

Evaluated on 500 English samples balanced across the five TISER categories:

| | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 0.921 | 90.18 | 0.844 | 0.838 | 0.912 | 0.000 |

Per dataset:

| Dataset | F1 | chrF | NormEM | EM | SoftEM | N |
|---|---:|---:|---:|---:|---:|---:|
| timeqa_easy | 0.974 | 97.98 | 0.960 | 0.960 | 0.980 | 100 |
| timeqa_hard | 0.981 | 98.39 | 0.950 | 0.950 | 0.960 | 100 |
| tempreason_l3 | 0.956 | 96.03 | 0.950 | 0.950 | 0.950 | 100 |
| tempreason_l2 | 0.840 | 85.05 | 0.810 | 0.810 | 0.810 | 100 |
| tgqa | 0.854 | 73.46 | 0.550 | 0.520 | 0.860 | 100 |



### Four-language model — EN + IT + DE + FR (15 000 training samples)

Evaluated on 500 samples: 125 per language.

| | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 0.888 | 88.14 | 0.798 | 0.792 | 0.856 | 0.006 |
| English | 0.925 | 91.75 | 0.840 | 0.840 | 0.912 | 0.000 |
| German | 0.905 | 88.40 | 0.816 | 0.816 | 0.888 | 0.008 |
| Italian | 0.887 | 89.48 | 0.800 | 0.784 | 0.856 | 0.008 |
| French | 0.834 | 82.94 | 0.736 | 0.728 | 0.768 | 0.008 |

Per dataset:

| Dataset | F1 | EM | N |
|---|---:|---:|---:|
| timeqa_easy | 0.980 | 0.950 | 100 |
| timeqa_hard | 0.928 | 0.880 | 100 |
| tempreason_l3 | 0.899 | 0.860 | 100 |
| tempreason_l2 | 0.838 | 0.770 | 100 |
| tgqa | 0.793 | 0.500 | 100 |

### Cross-lingual transfer — EN-only model on non-English test sets

Is multilingual fine-tuning necessary? Evaluating the **English-only** model
directly on the IT/DE/FR test sets (500 samples each) shows transfer is real
but partial, and English leakage rises an order of magnitude:

| Test set | F1 | chrF | NormEM | EM | SoftEM | EngLeak | Multilingual model F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IT | 0.808 | 82.42 | 0.678 | 0.674 | 0.748 | 0.062 | 0.887 |
| FR | 0.795 | 80.66 | 0.658 | 0.652 | 0.714 | 0.058 | 0.834 |
| DE | 0.783 | 79.33 | 0.674 | 0.672 | 0.758 | 0.048 | 0.905 |

The multilingual fine-tune gains 4–12 F1 over zero-shot transfer and cuts
EngLeak from ~5–6 % to ≤ 0.8 %.

### RAG ablation (Experiment 1)

Four cells on identical samples (N = 100, seed 42, few-shot mode,
`top_k=1`, `min_score=0.60`), aligned on shared `question_id`. The mixed test
set contains 25 samples per language. Retrieval hit-rate was 100 % on English
and 83 % on the mixed set (same-language exemplars are enforced, so
cross-language hits are filtered out):

| Test | Cell | F1 | chrF | NormEM | EM | SoftEM |
|---|---|---:|---:|---:|---:|---:|
| EN | base_norag | 0.688 | 69.11 | 0.60 | 0.59 | 0.65 |
| EN | base_rag | 0.737 | 73.15 | 0.61 | 0.60 | 0.66 |
| EN | ft_norag | 0.922 | 89.99 | 0.86 | 0.84 | 0.92 |
| EN | ft_rag | 0.899 | 87.57 | 0.82 | 0.81 | 0.89 |
| Mixed | base_norag | 0.569 | 60.73 | 0.42 | 0.42 | 0.53 |
| Mixed | base_rag | 0.562 | 60.82 | 0.37 | 0.37 | 0.57 |
| Mixed | ft_norag | 0.850 | 84.08 | 0.74 | 0.74 | 0.80 |
| Mixed | ft_rag | 0.882 | 86.89 | 0.78 | 0.77 | 0.84 |

On English, retrieval and fine-tuning act as **substitutes**: RAG lifts the
un-tuned base model (+0.049 F1, supplying the reasoning format) and slightly
hurts the fine-tuned model (−0.024 F1). On the mixed four-language set the
picture partially reverses: RAG is neutral on the base model and adds
+0.032 F1 to the fine-tuned multilingual model, concentrated in the
categories where fine-tuning has the most headroom. Since decoding is greedy,
samples without an exemplar are identical to the no-RAG cell, so the gain is
carried entirely by the retrieved 83 % (≈ +0.039 F1 per retrieved sample).

**Notes.**

The English-only model gives the strongest overall result, which is expected because it is trained and evaluated only on the original English distribution. The multilingual models trade a small amount of English performance for cross-lingual temporal reasoning ability.

French scores slightly lower than the other languages due to translation quality issues in the NLLB pipeline, especially carrier-prefix artifacts and isolated-entity translation noise. EngLeak remains very low across multilingual models, meaning the fine-tuned model almost never regresses to English when answering in a non-English language. `tgqa` is the hardest category because answers require matching free-form event descriptions rather than mostly entity names or numeric values.

---

## What this adds over the original TISER paper

| | TISER original | tiser-multilingual |
|---|---|---|
| Model | 7 B Qwen2.5 / Mistral | **3 B** Qwen2.5-Instruct |
| VRAM | ~40 GB | **8 GB** QLoRA 4-bit NF4 |
| Languages | English only | EN · IT · DE · FR |
| Training data | English only | English + NLLB-translated multilingual data |
| Retrieval | — | RAG ablation (few-shot / context-stuffing) |
| Metrics | EM, F1 | EM, NormEM, SoftEM, F1, chrF, EngLeak |

---

## Installation

```bash
git clone https://github.com/arianmo477/MultilingualTemporalReasoningTISER.git
cd MultilingualTemporalReasoningTISER

# Recommended: create the environment from environment.yml
conda env create -f environment.yml
conda activate tiser

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Or install manually with pip (note `faiss-cpu` is required for the RAG extension
and `accelerate` for 4-bit loading):

```bash
conda create -n tiser python=3.10 && conda activate tiser
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes tokenizers
pip install sentence-transformers sentencepiece sacremoses faiss-cpu tqdm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Datasets are stored with **git-LFS** (`git lfs install && git lfs pull`); see
[`docs/DATA.md`](docs/DATA.md) for which files are required and how to rebuild
derived artifacts such as the RAG index.

Unit tests cover the balanced samplers and the metrics:

```bash
python -m pytest tests/
```

## Model checkpoints

The fine-tuned LoRA adapters are available on the Hugging Face Hub. They apply
on top of `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4):

| Adapter | Languages | Overall F1 (500-sample eval) | Link |
|---|---|---:|---|
| `en_15000_8gb_val_qlora` | EN | 0.921 (EN test) | [HF Hub](https://huggingface.co/arianmo47/en_15000_8gb_val_qlora) |
| `de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora` | EN+IT+DE+FR | 0.888 (mixed test) | [HF Hub](https://huggingface.co/arianmo47/de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora) |

Load one directly from the Hub:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER = "arianmo47/de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora"

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
```

The Hub id can also be passed directly to the evaluation script's
`--adapter_dir` argument (`PeftModel.from_pretrained` accepts both local paths
and Hub ids).

### Project structure
```bash
tiser-multilingual/
│
├── multilingual_rag_tiser/
│   ├── training/
│   │   ├── train_qlora.py            # QLoRA fine-tuning
│   │   └── run_train.sh              # Training launcher
│   │
│   ├── evaluation/
│   │   ├── inference.py              # Iterative inference + metrics (optional RAG)
│   │   ├── compare_rag_ablation.py   # Aggregate the 4 RAG-ablation cells
│   │   ├── run_eval_rag.sh           # Single-run eval launcher
│   │   └── run_experiment1.sh        # Full 2x2 RAG ablation launcher
│   │
│   ├── rag/
│   │   └── build_rag_index.py        # FAISS index over training questions
│   │
│   ├── translate/
│   │   ├── translate_dataset.py      # NLLB translation pipeline
│   │   ├── score_translation.py      # Translation quality scorer
│   │   ├── run_translate.sh          # Translation launcher
│   │   └── run_score.sh              # Scoring launcher
│   │
│   └── preprocess/
│       ├── validate_tiser_dataset.py # Normalize/clean/validate EN source data
│       ├── mix_dataset.py            # Balanced multilingual mixer
│       ├── verify_15k_snapshot.py    # Provenance check for the frozen EN 15k
│       ├── run_validate_dataset.sh   # Validation launcher
│       └── run_mix_dataset.sh        # Mixer launcher
│
├── utils/
│   ├── metrics.py                    # EM, NormEM, SoftEM, F1, chrF, EngLeak
│   ├── sampling.py                   # Balanced subsampling (dataset / lang x dataset)
│   ├── prompt.py                     # Prompt builder shared by train + eval
│   ├── translation_utils.py          # Language packs, hallucination filter
│   ├── translation_quality.py        # Field scoring helpers for the QC scorer
│   ├── months.py                     # Month-name localization tables
│   ├── rag_utils.py                  # Retrieval + prompt injection (RAGContextBuilder)
│   └── io_gpu.py                     # JSON I/O, prompt loading, GPU helpers
│
├── tests/                            # pytest: sampling + metrics
│
├── data/
│   ├── prompts/
│   │   ├── tiser_full_en.txt         # English prompt
│   │   ├── tiser_full_it.txt         # Italian prompt
│   │   ├── tiser_full_de.txt         # German prompt
│   │   └── tiser_full_fr.txt         # French prompt
│   └── splits/
│       ├── train/                    # Per-language + mixed training JSONs
│       └── test/                     # Per-language test JSONs
│
└── experiments/                      # Checkpoints and result JSONs
```

### Training

The base model is always `Qwen/Qwen2.5-3B-Instruct` (override with
`BASE_MODEL=...`). Generation and evaluation are always iterative.

English-only model:
```bash
bash multilingual_rag_tiser/training/run_train.sh en tiser_full 15000
```

Single non-English models (uses the quality-filtered `_passed` file when present):
```bash
bash multilingual_rag_tiser/training/run_train.sh it tiser_full 15000
bash multilingual_rag_tiser/training/run_train.sh de tiser_full 15000
bash multilingual_rag_tiser/training/run_train.sh fr tiser_full 15000
```

Multilingual models — build the mixed file, then train on it (the launcher
auto-selects `(language, dataset)`-balanced subsampling for mixed files):
```bash
# EN + IT
bash multilingual_rag_tiser/preprocess/run_mix_dataset.sh train en,it 15000
bash multilingual_rag_tiser/training/run_train.sh en_it_mixed tiser_full 15000

# EN + IT + DE + FR
bash multilingual_rag_tiser/preprocess/run_mix_dataset.sh train de,it,fr,en 15000
bash multilingual_rag_tiser/training/run_train.sh de_it_fr_en_mixed tiser_full 15000
```

Key training hyperparameters:

| Hyperparameter        | Value                          |
| --------------------- | ------------------------------ |
| Base model            | `Qwen/Qwen2.5-3B-Instruct`     |
| Quantization          | NF4 4-bit, double quantization |
| LoRA rank / alpha     | 16 / 32                        |
| LoRA dropout          | 0.05                           |
| Max sequence length   | 1 536 tokens                   |
| Per-device batch size | 1                              |
| Gradient accumulation | 16                             |
| Optimizer             | `paged_adamw_8bit`             |
| Learning rate         | 3 × 10⁻⁴                       |
| Epochs                | 2                              |
| Validation split      | 10 %                           |

### Evaluation

`run_eval_rag.sh` takes `<lang> [adapter|none] [prompt] [max_samples]`.
**RAG is on by default** (`USE_RAG=1`); set `USE_RAG=0` for a clean baseline
evaluation.

Single language, no RAG:
```bash
USE_RAG=0 bash multilingual_rag_tiser/evaluation/run_eval_rag.sh en \
    experiments/qwen/en_15000_8gb_val_qlora/ tiser_full 500
```

Multilingual models — the script takes a single language, so evaluate the same
adapter once per language:
```bash
for L in en it de fr; do
  USE_RAG=0 bash multilingual_rag_tiser/evaluation/run_eval_rag.sh "$L" \
      experiments/qwen/de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora/ \
      tiser_full 500
done
```

Cross-lingual transfer of a single-language adapter works the same way — just
pass a different test language than the adapter was trained on.

Each invocation evaluates one language, writes one result JSON, and prints
per-language and per-dataset breakdowns at the end of the run. When
`--max_eval_samples` subsampling is used on a mixed-language test file, pass
`BALANCE=lang_dataset` (the ablation launcher does this automatically) so
every `(language, dataset)` cell is equally represented.

### RAG ablation (Experiment 1)

Run all four cells (base/ft × rag/norag) on identical samples and aggregate:
```bash
bash multilingual_rag_tiser/evaluation/run_experiment1.sh en \
    experiments/qwen/en_15000_8gb_val_qlora/ 100
```

The script builds the FAISS index if missing, runs the four cells, and calls
the aggregator. To re-aggregate existing outputs (no inference), set
`SKIP_RUN=1`. Manual aggregation:
```bash
python multilingual_rag_tiser/evaluation/compare_rag_ablation.py \
    base_norag=path/to/gen_base_norag.json \
    base_rag=path/to/gen_base_rag.json \
    ft_norag=path/to/gen_ft_norag.json \
    ft_rag=path/to/gen_ft_rag.json \
    --per_dataset --per_language
```

### Metric definitions
| Metric      | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| **EM**      | Strict exact match after text normalization                                 |
| **NormEM**  | Order-insensitive EM using sorted token bags; catches word-order variants   |
| **SoftEM**  | Substring containment match in either direction                             |
| **F1**      | Token-level F1, SQuAD-style                                                 |
| **chrF**    | Character n-gram F-score, robust to morphology                              |
| **EngLeak** | Fraction of non-English answers where the model returned the English answer |

## Translation pipeline

The multilingual datasets are generated from the English TISER data using
NLLB-200-distilled-1.3B. Supported targets: `it`, `de`, `fr`.

```bash
# via the launcher (resume-safe, uses the shared entity cache)
bash multilingual_rag_tiser/translate/run_translate.sh train it
bash multilingual_rag_tiser/translate/run_translate.sh train de
bash multilingual_rag_tiser/translate/run_translate.sh train fr

# or directly
python multilingual_rag_tiser/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_fr.json \
    --target_lang fr \
    --batch_size 2
```

### Design decisions

The translation pipeline has four layers to preserve quality on the structured TISER format.

**Entity cache** — every parenthesized entity or event, such as (Boston Red Sox) or (Westinghouse Electric), is translated once and reused everywhere it appears. This enforces identical surface forms across samples. The cache is saved to disk and reused across runs, making large translation jobs safe to interrupt and resume.

**Proper-noun shortcut** — short capitalized entities, such as person names, team abbreviations, and many club names, bypass the translation model entirely. NLLB is unreliable on isolated names and can hallucinate descriptive sentences.

**Hallucination detection** — after each NLLB call, outputs that elaborate a short input into a descriptive sentence are detected and discarded. In that case, the source string is used as a safer fallback.

**Language packs** — high-frequency templated phrases, such as "X plays for Y from Z to W", "True or false:", and common question stems, are translated using hand-written per-language rules instead of relying only on NLLB. This keeps temporal contexts structurally consistent and helps preserve dates.

### Translation quality scoring

Translated samples can be checked before training:

```bash
bash multilingual_rag_tiser/translate/run_score.sh train it
```

The scorer checks the translated fields (`question`, `temporal_context`,
`answer`, `output`) for semantic similarity, missing years, empty parentheses,
missing TISER tags, answer consistency, and leftover English phrases. Samples
below the quality threshold are split into `passed` and `failed` files; the
training launcher automatically prefers the `passed` file when it exists.

## TISER reasoning format

The model produces a four-section structured trace:
```
<reasoning>
  [Step-by-step temporal reasoning in plain paragraph form]
  <timeline>
    [Key events extracted from the context, with dates]
  </timeline>
  <reflection>
    [Self-review and correction of the reasoning]
  </reflection>
</reasoning>

<answer>
  [Final concise answer]
</answer>
```

Generation stops at `</answer>` via a custom StoppingCriteria. If `</answer>`
does not appear within `max_new_tokens`, generation is extended iteratively in
256-token increments, up to two extensions by default.

### Prompt templates
| File                  | Used for                                             |
| --------------------- | ---------------------------------------------------- |
| `tiser_full_en.txt`   | English CoT prompt for training and evaluation       |
| `tiser_full_it.txt`   | Italian prompt with native-language answer directive |
| `tiser_full_de.txt`   | German prompt                                        |
| `tiser_full_fr.txt`   | French prompt                                        |

Per-sample prompt selection at training and inference time ensures each sample
uses the prompt in its own language (`tiser_full_{lang}.txt`), with the
English prompt as the fallback.

### Hardware

All experiments run on a single consumer GPU.
| Component    | Value                       |
| ------------ | --------------------------- |
| GPU          | 8 GB VRAM                   |
| CPU          | AMD Ryzen, Alienware m17 R5 |
| RAM          | 16 GB                       |
| Python       | 3.10                        |
| PyTorch      | 2.x                         |
| Transformers | 4.x                         |

# Citation

If you use this code or results, please cite the original TISER paper:
```bibtex
@inproceedings{bazaga2025tiser,
  title     = {Learning to Reason Over Time: Timeline Self-Reflection
               for Improved Temporal Reasoning in Language Models},
  author    = {Bazaga, Adrián and others},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association
               for Computational Linguistics},
  year      = {2025},
}
```
# License

Released for research purposes.

The TISER dataset and Qwen2.5 model weights are subject to their own licenses. Please check the original authors' repositories and model cards for details.