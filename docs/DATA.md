# Data guide — sources, generated artifacts, and what the experiments need

This repo stores its datasets with **git-LFS** (`data/.gitattributes` sends every `*.json` to
LFS). The total is **~2.1 GB across 47 files**, which is a problem: GitHub's free LFS tier is
**1 GB storage + 1 GB/month bandwidth**, so a fresh `git clone` by a grader can exceed the quota
and receive **pointer files instead of data** — i.e. the results become unreproducible.

This guide classifies every data file as **source**, **generated artifact**, or **required for
the final experiments**, so we can prune the repo safely without losing reproducibility.

> Nothing here has been deleted. This is the plan + the exact commands to review before pruning.

---

## 1. How to get the data (any machine that runs training/eval)

The GPU box (not the dev laptop) is the only machine that needs the full data.

```bash
# one-time
git lfs install
# pull everything that is still LFS-tracked
git lfs pull
# …or pull only what you need (much smaller):
git lfs pull --include="data/splits/test/**,data/splits/train/train_tiser_15000_en.json"
```

If `git lfs` is missing: `brew install git-lfs` (macOS) / `apt-get install git-lfs` (Debian).

---

## 2. Classification

### 2a. SOURCE — upstream / preprocessing input (re-obtainable, NOT needed to reproduce results)
These are only read by the **validation/preprocessing** scripts
(`preprocess/validate_tiser_dataset.py`), not by training or evaluation. The reported models are
trained and evaluated from the per-language *splits*, so these can be pruned and re-obtained from
the original TISER release (`amazon-science/TISER`) if preprocessing must be re-run.

| File | Size | Note |
|---|---:|---|
| `data/TISER_train.json` | 208.5 MB | Pre-split upstream train |
| `data/TISER_test.json` | 88.0 MB | Pre-split upstream test |

### 2b. GENERATED — regenerable artifacts (safe to prune; rebuilt by our own code)

**RAG indexes** — rebuilt by `multilingual_rag_tiser/rag/build_rag_index.py`
(already added to `.gitignore`). ⚠️ The `.faiss` files are currently **raw git blobs, not LFS**,
so they bloat core history.

| File | Size | Rebuild with |
|---|---:|---|
| `data/rag/train_en/documents.json` | 176.8 MB | `build_rag_index.py --input <en train> --output_dir data/rag/train_en --language en` |
| `data/rag/train_en/index.faiss` | 77.8 MB | (same command) |
| `data/rag/train_it/documents.json` | 55.0 MB | `… --input <it train> --output_dir data/rag/train_it --language it` |
| `data/rag/train_it/index.faiss` | 23.6 MB | (same) |
| `data/rag/train_mixed/documents.json` | 55.2 MB | `… --input <mixed train> --output_dir data/rag/train_mixed` |
| `data/rag/train_mixed/index.faiss` | 23.0 MB | (same) |

`run_experiment1.sh` / `run_eval_rag.sh` rebuild the index automatically if it is missing.

**Diagnostic byproducts** — output of validation/scoring; no experiment reads them. Extract any
summary stats you want for the report (translation quality section) *before* pruning.

| File(s) | Size | Produced by |
|---|---:|---|
| `data/invalid_samples_train.json` | 219.1 MB | validation |
| `data/invalid_samples_test.json` | 1.7 MB | validation |
| `data/splits/{train,test}/{it,de,fr}/translation_quality_report.json` | ~66.6 MB total | `score_translation.py` |
| `data/splits/{train,test}/{it,de,fr}/TISER_*_failed.json` | ~107 MB total | `score_translation.py` |

**Intermediate translations** — full NLLB output before quality scoring. Training uses the
`*_passed.json` subsets, not these. Regenerable via the translation pipeline (GPU, expensive), so
keep only if you don't want to re-translate.

| File | Size | Note |
|---|---:|---|
| `data/splits/train/TISER_train_it.json` | 135.3 MB | pre-scoring IT |
| `data/splits/train/TISER_train_de.json` | 115.5 MB | pre-scoring DE |
| `data/splits/train/TISER_train_fr.json` | 100.4 MB | pre-scoring FR |
| `data/splits/train/de/TISER_train_de_passed.json` | 86.8 MB | only feeds the mixed file build |
| `data/splits/train/fr/TISER_train_fr_passed.json` | 79.3 MB | only feeds the mixed file build |
| `data/splits/train/TISER_train_de_it_en_mixed.json` | 70.7 MB | EN+IT+DE model is **not** in the reported results |

### 2c. REQUIRED — keep (needed to reproduce the reported models + Experiment 1)

**Training inputs (one per reported model):**

| Model (README) | File | Size |
|---|---|---:|
| EN-only 15k | `data/splits/train/train_tiser_15000_en.json` **(frozen snapshot — §3)** | 52.1 MB |
| IT-only 15k | `data/splits/train/it/TISER_train_it_passed.json` | 80.1 MB |
| EN+IT | `data/splits/train/TISER_train_it_en_mixed.json` | 65.9 MB |
| EN+IT+DE+FR | `data/splits/train/TISER_train_de_it_fr_en_mixed.json` | 73.1 MB |

**Evaluation splits (one per language):**

| Lang | File | Size |
|---|---|---:|
| EN | `data/splits/test/TISER_test_en.json` | 53.7 MB |
| IT | `data/splits/test/it/TISER_test_it_passed.json` | 1.1 MB |
| DE | `data/splits/test/de/TISER_test_de_passed.json` | 2.7 MB |
| FR | `data/splits/test/fr/TISER_test_fr_passed.json` | 1.4 MB |

**Small aids (keep — tiny):** `data/splits/**/event_translation_cache_*.json` (~3.5 MB total,
keeps translations cheap/consistent) and `data/prompts/*.txt` (not LFS).

Required total ≈ **526 MB** (or ≈ **404 MB** if the EN model switches to the 15k snapshot, §3) —
comfortably under the 1 GB LFS quota.

---

## 3. EN 15k training file — DECIDED: option B (freeze the snapshot)

**Decision (2026-07-04): use the frozen snapshot** `data/splits/train/train_tiser_15000_en.json`
(52 MB) as the exact, reproducible EN training set, because it freezes the exact data behind the
reported numbers. `run_train.sh` now points the `en` branch at this snapshot (falling back to the
full split only if the snapshot is missing). Passing `MAX_SAMPLES=15000` on a pre-balanced 15k
file is content-preserving with the fixed `balance_by_dataset_name` (returns the same set,
reshuffled by seed=42).

**VERIFIED (2026-07-04):** `multilingual_rag_tiser/preprocess/verify_15k_snapshot.py` confirmed the
snapshot is *multiset-identical* to the old `run_train.sh` sampling of the full split with seed=42
(15,000 rows, 0 symmetric difference). Note: the EN full split has **no `validation_status`
field**, so `--only_passed` is a no-op for EN (all 50,643 rows are balanced to 15k). The snapshot
carries **53 duplicate `tgqa` rows** — an artifact of the pre-fix balance sampler (tgqa pool =
2,188 < the 3,000/category target, so the buggy top-up re-drew 53). This is the exact data the
model trained on; report it as a data-quality footnote (EN 15k used all available tgqa + a ~0.35%
over-weighting).

**Consequence — DONE:** the full `data/splits/train/TISER_train_en.json` (174 MB) is no longer
referenced by any script. Both training (`run_train.sh`) and the RAG index source
(`run_experiment1.sh` / `run_eval_rag.sh`, `RAG_TRAIN_FILE` for `en`) now use the 15k snapshot
(with the full split kept only as a `find_first` fallback). The full file is therefore **prunable**
— it is listed in §5's pending prune block.

---

## 4. How to rebuild RAG indexes

```bash
# English
python multilingual_rag_tiser/rag/build_rag_index.py \
  --input data/splits/train/TISER_train_en.json \
  --output_dir data/rag/train_en --language en

# Italian
python multilingual_rag_tiser/rag/build_rag_index.py \
  --input data/splits/train/it/TISER_train_it_passed.json \
  --output_dir data/rag/train_it --language it

# Mixed (no language filter)
python multilingual_rag_tiser/rag/build_rag_index.py \
  --input data/splits/train/TISER_train_de_it_fr_en_mixed.json \
  --output_dir data/rag/train_mixed
```

---

## 5. Prune plan

**Executed in this pass (safe scope):** only the generated RAG artifacts below are untracked with
`git rm --cached` (files stay on disk; now `.gitignored`; rebuilt on the next eval run). No commit
was made. Everything else in this section remains a **reviewed-but-not-executed** plan.

```bash
# generated RAG artifacts (also .gitignored) — DONE this pass
git rm --cached data/rag/train_en/index.faiss data/rag/train_en/documents.json
git rm --cached data/rag/train_it/index.faiss data/rag/train_it/documents.json
git rm --cached data/rag/train_mixed/index.faiss data/rag/train_mixed/documents.json

# --- everything below: PENDING, not executed ---

# diagnostics
git rm --cached data/invalid_samples_train.json data/invalid_samples_test.json
git rm --cached data/splits/train/*/translation_quality_report.json
git rm --cached data/splits/test/*/translation_quality_report.json
git rm --cached data/splits/train/*/TISER_*_failed.json
git rm --cached data/splits/test/*/TISER_*_failed.json

# source (document upstream instead)
git rm --cached data/TISER_train.json data/TISER_test.json

# full EN train split — superseded by the frozen 15k snapshot (no script references it now)
git rm --cached data/splits/train/TISER_train_en.json

# intermediate translations (regenerable; keep passed subsets + mixed files)
git rm --cached data/splits/train/TISER_train_it.json \
                data/splits/train/TISER_train_de.json \
                data/splits/train/TISER_train_fr.json \
                data/splits/train/de/TISER_train_de_passed.json \
                data/splits/train/fr/TISER_train_fr_passed.json \
                data/splits/train/TISER_train_de_it_en_mixed.json
```

This untracks **~1.56 GB**, leaving **~0.5 GB** of required data. Then commit and push.

> ⚠️ `git rm --cached` stops *future* bloat but the blobs remain in **history** (the `.faiss`
> files especially, since they're raw git). To actually shrink the clone size you'd rewrite
> history with `git filter-repo` (or start a clean history for submission). Decide before the
> final push — a history rewrite changes commit hashes and needs a force-push.
