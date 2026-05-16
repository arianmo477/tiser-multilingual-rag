# tiser-multilingual

**Multilingual temporal reasoning with a small language model** — a two-part extension of [TISER](https://aclanthology.org/2025.acl-long.1358/) (Bazaga et al., ACL 2025).

### Features

1. **Small-model adaptation** — Qwen2.5-3B-Instruct + QLoRA, running on a single **8 GB GPU**.
2. **Multilingual extension** — an NLLB-based pipeline for structurally faithful Italian, German, French, and Persian data. One fine-tuned model can handle temporal reasoning across languages.

Original TISER used 7B models on English only. This project achieves strong performance with a 3B model and adds multilingual capability for EN · IT · DE · FR.

---

## Results

All evaluations use a balanced test set of **500 samples** across the five TISER categories: `tgqa`, `tempreason_l2`, `tempreason_l3`, `timeqa_easy`, and `timeqa_hard`.

### English-only model — EN, 15k training samples

**Overall**

| F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---:|---:|---:|---:|---:|---:|
| 0.921 | 90.18 | 0.844 | 0.838 | 0.912 | 0.000 |

**Per dataset**

| Dataset | F1 | chrF | NormEM | EM | SoftEM | N |
|---|---:|---:|---:|---:|---:|---:|
| timeqa_easy | 0.974 | 97.98 | 0.960 | 0.960 | 0.980 | 100 |
| timeqa_hard | 0.981 | 98.39 | 0.950 | 0.950 | 0.960 | 100 |
| tempreason_l3 | 0.956 | 96.03 | 0.950 | 0.950 | 0.950 | 100 |
| tempreason_l2 | 0.840 | 85.05 | 0.810 | 0.810 | 0.810 | 100 |
| tgqa | 0.854 | 73.46 | 0.550 | 0.520 | 0.860 | 100 |

### Two-language model — EN + IT, 15k training samples

Evaluated on 500 samples: 250 English and 250 Italian.

**Overall**

| F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---:|---:|---:|---:|---:|---:|
| 0.892 | 88.07 | 0.802 | 0.794 | 0.860 | 0.010 |

**Per language**

| Language | F1 | NormEM | EM | SoftEM |
|---|---:|---:|---:|---:|
| English | 0.911 | 0.836 | 0.832 | 0.900 |
| Italian | 0.873 | 0.768 | 0.756 | 0.820 |

**Per dataset**

| Dataset | F1 | EM | N |
|---|---:|---:|---:|
| timeqa_easy | 0.957 | 0.930 | 100 |
| timeqa_hard | 0.925 | 0.860 | 100 |
| tempreason_l3 | 0.930 | 0.880 | 100 |
| tempreason_l2 | 0.817 | 0.760 | 100 |
| tgqa | 0.831 | 0.540 | 100 |

### Four-language model — EN + IT + DE + FR, 15k training samples

Evaluated on 500 samples: 125 per language.

**Overall**

| F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---:|---:|---:|---:|---:|---:|
| 0.888 | 88.14 | 0.798 | 0.792 | 0.856 | 0.006 |

**Per language**

| Language | F1 | NormEM | EM | SoftEM |
|---|---:|---:|---:|---:|
| English | 0.925 | 0.840 | 0.840 | 0.912 |
| German | 0.905 | 0.816 | 0.816 | 0.888 |
| Italian | 0.887 | 0.800 | 0.784 | 0.856 |
| French | 0.834 | 0.736 | 0.728 | 0.768 |

**Per dataset**

| Dataset | F1 | EM | N |
|---|---:|---:|---:|
| timeqa_easy | 0.980 | 0.950 | 100 |
| timeqa_hard | 0.928 | 0.880 | 100 |
| tempreason_l3 | 0.899 | 0.860 | 100 |
| tempreason_l2 | 0.838 | 0.770 | 100 |
| tgqa | 0.793 | 0.500 | 100 |

**Notes.**

The English-only model gives the strongest overall result, which is expected because it is trained and evaluated only on the original English distribution. The multilingual models trade a small amount of English performance for cross-lingual temporal reasoning ability.

French performance is slightly lower due to translation artifacts in the NLLB pipeline, especially carrier-prefix artifacts and isolated-entity translation noise. `tgqa` remains the hardest category because answers require matching free-form event descriptions rather than mostly entity names or numeric values. EngLeak is very low, meaning the model rarely falls back to English when answering in a non-English language.

---

## Comparison with Original TISER

| Aspect | Original TISER | tiser-multilingual |
|---|---|---|
| Model | 7B Qwen2.5 / Mistral | **3B** Qwen2.5-Instruct |
| VRAM | ~40 GB | **8 GB** QLoRA 4-bit |
| Languages | English only | EN · IT · DE · FR |
| Training data | English only | English + NLLB-translated data |
| Metrics | EM, F1 | EM, NormEM, SoftEM, F1, chrF, EngLeak |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/tiser-multilingual.git
cd tiser-multilingual

conda create -n tiser python=3.10
conda activate tiser

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes datasets tqdm
pip install "transformers[sentencepiece]" sentence-transformers sacremoses

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"


### Project Structure
tiser-multilingual/
├── tiser_lite/
│   ├── training/
│   │   ├── train_qlora.py
│   │   └── run_train.sh
│   ├── evaluation/
│   │   ├── inference.py
│   │   └── run_eval.sh
│   ├── translate/
│   │   └── translate_dataset.py
│   ├── preprocess/
│   │   ├── mix_dataset.py
│   │   └── run_mix_dataset.sh
│   └── utils/
│       ├── metrics.py
│       ├── prompt.py
│       ├── translation_utils.py
│       └── io_gpu.py
├── data/
│   ├── prompts/
│   │   ├── tiser_full.txt
│   │   ├── tiser_full_it.txt
│   │   ├── tiser_full_de.txt
│   │   └── tiser_full_fr.txt
│   └── splits/
│       ├── train/
│       └── test/
└── experiments/