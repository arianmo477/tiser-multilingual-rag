# tiser-multilingual

Multilingual temporal reasoning with a small language model — a two-part extension of [TISER](https://aclanthology.org/2025.acl-long.1358/) (Bazaga et al., ACL 2025):

1. **Small-model adaptation** — Qwen2.5-3B-Instruct + QLoRA on a single 8 GB GPU, matching the spirit of the original paper's reasoning pipeline without requiring 40+ GB of VRAM.

2. **Multilingual extension** — an NLLB-based translation pipeline that produces structurally faithful Italian, German, French, and Persian training data, enabling a single fine-tuned model to reason temporally across languages.

The original TISER paper uses Qwen2.5-7B and Mistral-7B on English only. This project reaches strong English-only performance with a 3 B model and extends the same temporal-reasoning pipeline to multilingual EN · IT · DE · FR training.

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

### Italian-only model — IT, 15k training samples

Evaluated on 500 Italian samples across the five TISER categories.

**Overall**

| F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---:|---:|---:|---:|---:|---:|
| 0.881 | 87.67 | 0.790 | 0.772 | 0.838 | 0.006 |

**Per dataset**

| Dataset | F1 | chrF | NormEM | EM | SoftEM | EngLeak | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| timeqa_easy | 0.974 | 97.46 | 0.955 | 0.955 | 0.966 | 0.000 | 88 |
| timeqa_hard | 0.923 | 94.48 | 0.885 | 0.885 | 0.885 | 0.000 | 96 |
| tempreason_l3 | 0.892 | 90.14 | 0.860 | 0.822 | 0.879 | 0.028 | 107 |
| tempreason_l2 | 0.807 | 80.34 | 0.733 | 0.695 | 0.733 | 0.000 | 105 |
| tgqa | 0.828 | 77.96 | 0.548 | 0.539 | 0.750 | 0.000 | 104 |

### Two-language model — EN + IT (15 000 training samples)

Evaluated on 500 samples: 250 English and 250 Italian.

| | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 0.892 | 88.07 | 0.802 | 0.794 | 0.860 | 0.010 |
| English | 0.911 | 89.34 | 0.836 | 0.832 | 0.900 | 0.000 |
| Italian | 0.873 | 86.80 | 0.768 | 0.756 | 0.820 | 0.020 |

Per dataset:

| Dataset | F1 | EM | N |
|---|---:|---:|---:|
| timeqa_easy | 0.957 | 0.930 | 100 |
| timeqa_hard | 0.925 | 0.860 | 100 |
| tempreason_l3 | 0.930 | 0.880 | 100 |
| tempreason_l2 | 0.817 | 0.760 | 100 |
| tgqa | 0.831 | 0.540 | 100 |

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


### Project structure
```bash
tiser-multilingual/
│
├── multilingual_rag_tiser/
│   ├── training/
│   │   ├── train_qlora.py          # QLoRA fine-tuning
│   │   └── run_train.sh            # Training launcher
│   │
│   ├── evaluation/
│   │   ├── inference.py            # Inference + metric computation
│   │   └── run_eval_rag.sh             # Multi-language eval launcher
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
│   ├── prompt.py                   # Prompt builder shared by train + eval
│   ├── translation_utils.py        # Language packs, hallucination filter
│   └── io_gpu.py                   # JSON I/O, balanced sampling
│
├── data/
│   ├── prompts/
│   │   ├── tiser_full_en.txt          # English prompt
│   │   ├── tiser_full_it.txt       # Italian prompt
│   │   ├── tiser_full_de.txt       # German prompt
│   │   └── tiser_full_fr.txt       # French prompt
│   └── splits/
│       ├── train/                  # Per-language + mixed training JSONs
│       └── test/                   # Per-language test JSONs
│
└── experiments/                    # Checkpoints and result JSONs
```


### Training
 English-only model
```bash
bash multilingual_rag_tiser/training/run_train.sh qwen en tiser_full 15000
```

 Two-language model: EN + IT
```bash
#Build the mixed training file
bash multilingual_rag_tiser/preprocess/run_mix_dataset.sh train en,it 15000
# Fine-tune
bash multilingual_rag_tiser/training/run_train.sh qwen en_it_mixed tiser_full 15000
```
# Key training hyperparameters

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
Single language
```bash
bash multilingual_rag_tiser/evaluation/run_eval_rag.sh qwen en \
    experiments/qwen/en_15000_8gb_val_qlora/ \
    iterative tiser_full 500
```
Two-language model — the script takes a single language, so evaluate the same
adapter once per language:
```bash
for L in en it; do
  bash multilingual_rag_tiser/evaluation/run_eval_rag.sh qwen "$L" \
      experiments/qwen/en_it_mixed_tiser_full_15000_8gb_val_qlora/ \
      iterative tiser_full 500
done
```
Four-language model
```bash
for L in en it de fr; do
  bash multilingual_rag_tiser/evaluation/run_eval_rag.sh qwen "$L" \
      experiments/qwen/de_it_fr_en_mixed_tiser_full_15000_8gb_val_qlora/ \
      iterative tiser_full 500
done
```
Each invocation evaluates one language, writes one result JSON, and prints per-language and
per-dataset breakdowns automatically at the end of the run. To aggregate the RAG ablation
cells into one comparison:
```bash
python multilingual_rag_tiser/evaluation/compare_rag_ablation.py \
    base_norag=path/to/gen_base_norag.json \
    base_rag=path/to/gen_base_rag.json \
    ft_norag=path/to/gen_ft_norag.json \
    ft_rag=path/to/gen_ft_rag.json \
    --per_dataset
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

Translation pipeline

The multilingual datasets are generated from the English TISER data using NLLB-200-distilled-1.3B

# Italian
```bash
python multilingual_rag_tiser/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_it.json \
    --target_lang it \
    --batch_size 2
```

# German
```bash
python multilingual_rag_tiser/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_de.json \
    --target_lang de \
    --batch_size 2
```
# French
```bash
python multilingual_rag_tiser/translate/translate_dataset.py \
    --input  data/splits/train/TISER_train_en.json \
    --output data/splits/train/TISER_train_fr.json \
    --target_lang fr \
    --batch_size 2
```
Design decisions

The translation pipeline has four layers to preserve quality on the structured TISER format.

Entity cache — every parenthesized entity or event, such as (Boston Red Sox) or (Westinghouse Electric), is translated once and reused everywhere it appears. This enforces identical surface forms across samples. The cache is saved to disk and reused across runs, making large translation jobs safe to interrupt and resume.

Proper-noun shortcut — short capitalized entities, such as person names, team abbreviations, and many club names, bypass the translation model entirely. NLLB is unreliable on isolated names and can hallucinate descriptive sentences.

Hallucination detection — after each NLLB call, outputs that elaborate a short input into a descriptive sentence are detected and discarded. In that case, the source string is used as a safer fallback.

Language packs — high-frequency templated phrases, such as "X plays for Y from Z to W", "True or false:", and common question stems, are translated using hand-written per-language rules instead of relying only on NLLB. This keeps temporal contexts structurally consistent and helps preserve dates.

Translation quality scoring

Translated samples can be checked before training:

bash multilingual_rag_tiser/translate/run_score.sh train

The scorer checks the translated fields:

question
temporal_context
answer
output

It validates semantic similarity, missing years, empty parentheses, missing TISER tags, answer consistency, and leftover English phrases. Samples below the quality threshold can be separated into passed and failed files before training.

TISER reasoning format

The model produces a four-section structured trace:
```bash
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

Generation stops at </answer> via a custom StoppingCriteria. If </answer> does not appear within max_new_tokens, the iterative strategy extends generation in 256-token increments, up to two extensions by default.

### Prompt templates
| File                  | Used for                                             |
| --------------------- | ---------------------------------------------------- |
| `tiser_full.txt`      | Main CoT prompt for training and evaluation          |
| `tiser_full_it.txt`   | Italian prompt with native-language answer directive |
| `tiser_full_de.txt`   | German prompt                                        |
| `tiser_full_fr.txt`   | French prompt                                        |
| `tiser_compact.txt`   | Compact variant with stricter length guidance        |
| `standard.txt`        | Direct-answer baseline without CoT                   |
| `answer_recovery.txt` | Second-pass prompt when generation lacks `</answer>` |


Per-sample prompt selection at training and inference time ensures each sample uses the prompt in its own language. The fallback is always the base English prompt.

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
```bash
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