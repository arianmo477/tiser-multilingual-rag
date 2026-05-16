# tiser-multilingual

Multilingual temporal reasoning with a small language model — a two-part
extension of [TISER](https://aclanthology.org/2025.acl-long.1358/)
(Bazaga et al., ACL 2025):

1. **Small-model adaptation** — Qwen2.5-3B-Instruct + QLoRA on a single
   8 GB GPU, matching the spirit of the original paper's reasoning pipeline
   without requiring 40+ GB of VRAM.

2. **Multilingual extension** — an NLLB-based translation pipeline that
   produces structurally faithful Italian, German, French, and Persian
   training data, enabling a single fine-tuned model to reason temporally
   across languages.

The original TISER paper uses Qwen2.5-7B and Mistral-7B on English only.
This project reaches competitive performance with a 3 B model and extends
it to four languages simultaneously.

---

## Results

All evaluations use a mixed test set of 500 samples balanced across the
five TISER categories (story / L2 / L3 / wiki\_easy / wiki\_hard).

### Two-language model — EN + IT (15 000 training samples)

Evaluated on 500 samples (250 EN · 250 IT):

| | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---|---|---|---|---|---|
| **Overall** | 0.892 | 88.07 | 0.802 | 0.794 | 0.860 | 0.010 |
| English | 0.911 | 89.34 | 0.836 | 0.832 | 0.900 | 0.000 |
| Italian | 0.873 | 86.80 | 0.768 | 0.756 | 0.820 | 0.020 |

Per dataset:

| Dataset | F1 | EM | N |
|---|---|---|---|
| timeqa\_easy | 0.957 | 0.930 | 100 |
| timeqa\_hard | 0.925 | 0.860 | 100 |
| tempreason\_l3 | 0.930 | 0.880 | 100 |
| tempreason\_l2 | 0.817 | 0.760 | 100 |
| tgqa | 0.831 | 0.540 | 100 |

### Four-language model — EN + IT + DE + FR (15 000 training samples)

Evaluated on 500 samples (125 per language):

| | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---|---|---|---|---|---|
| **Overall** | 0.888 | 88.14 | 0.798 | 0.792 | 0.856 | 0.006 |
| English | 0.925 | 91.75 | 0.840 | 0.840 | 0.912 | 0.000 |
| German | 0.905 | 88.40 | 0.816 | 0.816 | 0.888 | 0.008 |
| Italian | 0.887 | 89.48 | 0.800 | 0.784 | 0.856 | 0.008 |
| French | 0.834 | 82.94 | 0.736 | 0.728 | 0.768 | 0.008 |

Per dataset:

| Dataset | F1 | EM | N |
|---|---|---|---|
| timeqa\_easy | 0.980 | 0.950 | 100 |
| timeqa\_hard | 0.928 | 0.880 | 100 |
| tempreason\_l3 | 0.899 | 0.860 | 100 |
| tempreason\_l2 | 0.838 | 0.770 | 100 |
| tgqa | 0.793 | 0.500 | 100 |

**Notes.**
French scores slightly lower than the other languages due to translation
quality issues in the NLLB pipeline (smaller model + carrier-prefix
artifacts). EngLeak ≈ 0.006–0.010 across models, meaning the fine-tuned
model almost never regresses to English when answering in a non-English
language. tgqa is the hardest category because answers require matching
free-form event descriptions rather than entity names.

---

## What this adds over the original TISER paper

| | TISER (original) | tiser-multilingual |
|---|---|---|
| Model | 7 B (Qwen2.5 / Mistral) | **3 B** Qwen2.5-Instruct |
| VRAM | ~40 GB | **8 GB** (QLoRA 4-bit NF4) |
| Languages | English only | EN · IT · DE · FR |
| Training data | English only | Translated via NLLB |
| Metrics | EM, F1 | EM, NormEM, SoftEM, F1, chrF, EngLeak |

---

## Installation

```bash
git clone https://github.com/<your-username>/tiser-multilingual.git
cd tiser-multilingual

conda create -n tiser python=3.10
conda activate tiser

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes datasets tqdm
pip install transformers[sentencepiece]    # required for NLLB

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

---

## Project structure

tiser-multilingual/
│
├── tiser_lite/
│   ├── training/
│   │   ├── train_qlora.py          # QLoRA fine-tuning
│   │   └── run_train.sh            # Training launcher
│   │
│   ├── evaluation/
│   │   ├── inference.py            # Inference + metric computation
│   │   └── run_eval.sh             # Multi-language eval launcher
│   │
│   ├── translate/
│   │   └── translate_dataset.py    # NLLB translation pipeline
│   │
│   └── preprocess/
│       ├── mix_dataset.py          # Balanced multilingual mixer
│       └── run_mix_dataset.sh      # Mixer launcher
│
├── utils/
│   ├── metrics.py                  # EM, NormEM, SoftEM, F1, chrF, EngLeak
│   ├── prompt.py                   # Prompt builder (shared by train + eval)
│   ├── translation_utils.py        # Language packs, hallucination filter
│   └── io_gpu.py                   # JSON I/O, balanced sampling
│
├── data/
│   ├── prompts/
│   │   ├── tiser_full.txt          # English prompt
│   │   ├── tiser_full_it.txt       # Italian
│   │   ├── tiser_full_de.txt       # German
│   │   └── tiser_full_fr.txt       # French
│   └── splits/
│       ├── train/                  # Per-language + mixed training JSONs
│       └── test/                   # Per-language test JSONs
│
└── experiments/                    # Checkpoints and result JSONs



---

## Training

### Two-language model (EN + IT)

```bash
# Build the mixed training file
bash tiser_lite/preprocess/run_mix_dataset.sh train en,it 15000

# Fine-tune
bash tiser_lite/training/run_train.sh qwen en_it_mixed tiser_full 15000
```

### Four-language model (EN + IT + DE + FR)

```bash
# Build mixed file (requires translated files to exist — see Translation)
bash tiser_lite/preprocess/run_mix_dataset.sh train en,it,de,fr 15000

# Fine-tune
bash tiser_lite/training/run_train.sh qwen de_it_fr_en_mixed tiser_full 15000
```

### Key training hyperparameters

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | NF4 4-bit, double quant |
| LoRA rank / alpha | 16 / 32 |
| Max sequence length | 1 536 tokens |
| Per-device batch size | 1 × 16 gradient accumulation steps |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | 3 × 10⁻⁴ |
| Epochs | 2 |
| Validation split | 10 % |

---

## Evaluation

### Single language

```bash
bash tiser_lite/evaluation/run_eval.sh qwen en \
    experiments/qwen/en_it_mixed_tiser_full_15000_8gb_val_qlora/ \
    iterative tiser_full 500
```

### All four languages at once

```bash
bash tiser_lite/evaluation/run_eval.sh qwen en,it,de,fr \
    experiments/qwen/de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora/ \
    iterative tiser_full 500
```

This produces one result JSON per language and prints per-language and
per-dataset breakdowns automatically at the end of each run.

### Aggregating multiple result files

```bash
python tiser_lite/temp.py \
    experiments/qwen/*/results/gen_*.json \
    --per_language \
    --per_dataset
```

### Metric definitions

| Metric | Description |
|---|---|
| **EM** | Strict exact match after text normalization |
| **NormEM** | Order-insensitive EM (sorted token bag — catches word-order variants) |
| **SoftEM** | Substring containment match in either direction |
| **F1** | Token-level F1 (SQuAD-style) |
| **chrF** | Character n-gram F-score — robust to morphological variation |
| **EngLeak** | Fraction of non-English answers where model returned English |

---

## Translation pipeline

Translates the English TISER dataset using
[NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B).

```bash
# Italian
python tiser_lite/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_it.json \
    --target_lang it --batch_size 2

# German
python tiser_lite/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_de.json \
    --target_lang de --batch_size 2

# French
python tiser_lite/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_fr.json \
    --target_lang fr --batch_size 2
```

Supported target languages: `it` · `de` · `fr` · `fa` · `es`

### Design decisions

The pipeline has four layers to preserve translation quality on the
structured TISER format:

**Entity cache** — every parenthesized entity (`(Boston Red Sox)`,
`(Westinghouse Electric)`) is translated once and reused everywhere it
appears, enforcing identical surface forms across all samples. The cache
is saved to disk and reused across runs, making large translation jobs
safe to interrupt and resume.

**Proper-noun shortcut** — short capitalized entities (person names,
team abbreviations) bypass the model entirely. NLLB is unreliable on
isolated names and frequently hallucinates descriptive sentences.

**Hallucination detection** — after each NLLB call, outputs that elaborate
a short input into a descriptive sentence are detected and discarded;
the source string is used as a fallback instead.

**Language packs** — high-frequency templated phrases
(`"X plays for Y from Z to W"`, `"True or false:"`, question stems)
are substituted using hand-written per-language rules rather than passed
through NLLB, ensuring consistent phrasing across thousands of samples.

---

## TISER reasoning format

The model produces a four-section structured trace:

```xml
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

Generation stops at `</answer>` via a custom `StoppingCriteria`. If
`</answer>` does not appear within `max_new_tokens`, the iterative
strategy extends generation in 256-token increments (up to 2 extensions
by default).

---

## Prompt templates

| File | Used for |
|---|---|
| `tiser_full.txt` | Main CoT prompt — training and evaluation |
| `tiser_full_it.txt` | Italian — includes native-language answer directive |
| `tiser_full_de.txt` | German |
| `tiser_full_fr.txt` | French |
| `tiser_compact.txt` | Compact variant with stricter length guidance |
| `standard.txt` | Direct-answer baseline (no CoT) |
| `answer_recovery.txt` | Second-pass prompt when generation lacks `</answer>` |

Per-sample prompt selection at training and inference time ensures each
sample always sees the prompt in its own language. The fallback is always
the base English prompt.

---

## Hardware

All experiments run on a single consumer GPU:

| | |
|---|---|
| GPU | 8 GB VRAM |
| CPU | AMD Ryzen (Alienware m17 R5) |
| RAM | 16 GB |
| Python | 3.10 |
| PyTorch | 2.x |
| Transformers | 4.x |

---

## Citation

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

---

## License

Released for research purposes.
The TISER dataset and Qwen2.5 model weights are subject to their own
licenses — see the original authors' repositories for details.