# Report Draft & Reference — Multilingual + RAG Extensions of TISER

> Working reference for the LaTeX report (IEEE conference format, English, max 4 pages
> excl. bibliography/appendix). Fill the LaTeX from this; keep this file as the running
> source of truth. **[TODO]** markers flag numbers/claims that must be produced by
> actually running experiments before they go in the paper.
>
> Repo link (required in report): **[TODO: paste GitHub URL]**
> Base paper: Bazaga et al., *Learning to Reason Over Time: Timeline Self-Reflection for
> Improved Temporal Reasoning in Language Models*, ACL 2025.
> https://aclanthology.org/2025.acl-long.1358/ · Original code: amazon-science/TISER

---

## 0. Grading map (keep in view)

| Component | Max pts | Where addressed |
|---|---|---|
| Report | 2 | this doc → LaTeX |
| Reproducibility | 2 | §7, dataset/LFS notes, run scripts |
| Oral | 1 | slides (separate) |
| Extension 1 — Multilingual | 2.5 | §4.1, §6.1 (teammate) |
| Extension 2 — RAG | 2.5 | §4.2, §6.2 (**this author**) |

---

## 1. Abstract (~250 words) — draft

Temporal reasoning requires a model to order events, resolve relative time expressions, and
answer questions whose correctness hinges on dates, entities, and event ordering. TISER
(Bazaga et al., 2025) improves temporal reasoning by making a model emit an explicit trace —
step-by-step reasoning, a constructed timeline, and a self-reflection step — before its final
answer, rather than answering directly. The original work targets 7B English-only models. We
reproduce the TISER reasoning format on a **3B** model (Qwen2.5-3B-Instruct) fine-tuned with
**QLoRA (4-bit NF4)** so the pipeline fits an **8 GB consumer GPU**, and we extend it in two
independent directions. **(1) Multilingual extension:** an NLLB-200 translation pipeline
produces structurally faithful Italian, German, and French training data, with quality control
via entity caching, proper-noun shortcutting, hallucination detection, and language-specific
temporal templates; we test whether one fine-tuned model preserves the TISER structure across
languages. **(2) RAG extension:** we add retrieval-augmented generation as an ablation, encoding
training questions with a multilingual sentence encoder into a FAISS index and injecting the most
similar solved example as a few-shot demonstration (or its context, in a stuffing variant). Our
central question is whether retrieval helps a benchmark whose samples are **already
self-contained**. We find **[TODO: fill after experiments]** that on the fine-tuned model
retrieval is redundant, while on the non-fine-tuned base model few-shot retrieval recovers much
of the format/reasoning gap — i.e. retrieval and fine-tuning act as *substitutes*. Results:
English F1 ≈ 0.92; multilingual models trade ≈3–4 F1 points for cross-lingual ability with near-
zero English leakage. **[Refine word count to ~250 once numbers are final.]**

---

## 2. Introduction / Problem statement

**Task.** Given a natural-language *temporal question* and a *temporal context* (a set of dated
facts / events), produce the correct answer, using an explicit reasoning trace.

**Formal I/O.**
- **Input:** `(context C, question Q, language L)`, where `C` is a self-contained set of temporal
  facts and `Q` is a temporal query (entity-at-time, ordering, duration, boolean, or "unknown").
- **Task addressed:** closed-context temporal question answering with structured reasoning.
- **Expected output:** a structured trace then a final answer:
  ```
  <reasoning> … <timeline> … </timeline> <reflection> … </reflection> </reasoning>
  <answer> final answer </answer>
  ```
  Evaluation targets only the `<answer>` payload; the trace is the mechanism, not the scored
  object.

**Datasets (TISER categories):** `tgqa`, `tempreason_l2`, `tempreason_l3`, `timeqa_easy`,
`timeqa_hard`. `tgqa` answers are free-form event descriptions (hardest); the others are largely
entity/date/boolean answers.

**Contributions.**
1. Small-model TISER: 3B + QLoRA on 8 GB, matching the reasoning pipeline of a 7B setup.
2. Multilingual TISER via NLLB translation + quality control (teammate).
3. RAG ablation isolating *when* retrieval helps temporal reasoning (this author).

---

## 3. Related work
- TISER (Bazaga et al., 2025) — timeline self-reflection. Base method.
- TimeQA / TempReason / TGQA — source temporal-QA datasets.
- QLoRA (Dettmers et al., 2023) — 4-bit NF4 fine-tuning.
- NLLB-200 (Meta, 2022) — multilingual MT backbone.
- Sentence-Transformers / FAISS — dense retrieval.
- RAG (Lewis et al., 2020) — retrieval-augmented generation. Note: designed for *open-domain*
  knowledge gaps, which is precisely the regime TISER samples do **not** have.

---

## 4. Methodology

### 4.0 Base pipeline (shared)
- Base model: `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 + double-quant (QLoRA), LoRA r=16/α=32,
  dropout 0.05, max seq len 1536 (train), lr 3e-4, 2 epochs, batch 1 × grad-accum 16,
  `paged_adamw_8bit`, 10% val split.
- Prompt: per-language template (`data/prompts/tiser_full_{en,it,de,fr}.txt`), falling back to
  English. `build_prompt` assembles `instruction + Context + Question`; the `<reasoning>` cue is
  prepended to the model output at decode time.
- Decoding: greedy, `repetition_penalty=1.1`, custom `AnswerTagStopping` stops at `</answer>`;
  the *iterative* strategy extends generation in 256-token steps (≤2 extensions) if `</answer>`
  is missing.

### 4.1 Multilingual extension (teammate — summary for coherence)
NLLB-200-distilled-1.3B translates EN→{IT,DE,FR}. Four quality layers: (a) **entity cache** —
each parenthesized entity/event translated once and reused for surface-form consistency, cached
to disk; (b) **proper-noun shortcut** — short capitalized names bypass NLLB (which hallucinates
on isolated names); (c) **hallucination detection** — outputs that expand a short input into a
descriptive sentence are discarded and the source is kept; (d) **language packs** — high-
frequency templated phrases and question stems use hand-written per-language rules to preserve
dates and temporal structure. A scorer validates semantic similarity, missing years, empty
parentheses, missing TISER tags, answer consistency, and English leftovers, splitting samples
into `passed`/`failed`.

### 4.2 RAG extension (**this author** — the graded part)

**Idea.** Instead of appending retrieved *evidence* (classic RAG), retrieve a **similar solved
training example** and inject it as a demonstration, because every target already carries its own
complete `temporal_context`.

**Index construction** (`multilingual_rag_tiser/rag/build_rag_index.py`):
1. Load a training split; drop corrupt/short samples (empty output, `()` translation artifacts).
2. Encode each sample's **question text** (not the context) with
   `paraphrase-multilingual-MiniLM-L12-v2`; L2-normalize embeddings.
3. Store in FAISS `IndexFlatIP` (inner product on normalized vectors = **cosine similarity**),
   alongside `documents.json` (full sample dicts, incl. `_en` fallbacks).
   Indexes: `data/rag/train_{en,it,mixed}/`.

**Retrieval + injection** (`utils/rag_utils.py`, used by `evaluation/inference.py`):
- Embed the target question, cosine-search the index, drop hits below `min_score` (≈0.55–0.60),
  drop self-matches by `question_id`, and (fixed) drop cross-language hits on a mixed index.
- Two modes (both hypothesized to *hurt* the fine-tuned model — a deliberate ablation):
  - **`few_shot` (default):** format the retrieved `question + context + solved output` as a
    demonstration block and **prepend** it before the instruction.
  - **`context_stuffing`:** append the retrieved sample's `temporal_context` onto the target's
    own context (naive RAG; expected to hurt more — injects facts about unrelated entities/dates
    into a closed context).

**Pseudocode (retrieval path):**
```
build_index(train):
    docs = [s for s in train if valid(s)]
    E    = normalize(encoder.encode([s.question for s in docs]))
    index = FAISS_IndexFlatIP(dim); index.add(E)
    save(index, docs)

answer(sample):
    q = encoder.encode(sample.question); q = normalize(q)
    hits = index.search(q, k)                       # cosine
    hits = [h for h in hits if h.score >= min_score
                and h.id != sample.id
                and (lang is None or h.language == lang)]
    if mode == few_shot:  prompt = demo(hits) + base_prompt
    else:                 sample.context += stuffed(hits)
    return generate(prompt, sample)
```

**Experiment 1 — the key RAG study (base vs fine-tuned × RAG on/off).**
Four cells on identical samples (same test file + fixed seed 42):
`base_norag`, `base_rag`, `ft_norag`, `ft_rag`.
Runner: `multilingual_rag_tiser/evaluation/run_experiment1.sh <model> <lang> <adapter> [N]`.
Aggregator: `compare_rag_ablation.py` aligns cells on shared `question_id` and reports the RAG
delta per model. **Hypothesis:** RAG helps `base` (supplies the reasoning format it never
learned) but is ≈neutral on `ft` → retrieval and fine-tuning are *substitutes*.

---

## 5. Experimental setup
- **Hardware:** single 8 GB consumer GPU; Ryzen CPU / 16 GB RAM (Alienware m17 R5).
- **Software:** Python 3.10, PyTorch 2.x, Transformers 4.x, PEFT, bitsandbytes, datasets,
  sentence-transformers, FAISS, NLLB (via transformers), sacremoses.
- **Validation:** 10% train split for fine-tuning; evaluation on held-out test splits, class-
  balanced across the 5 categories via `balance_by_dataset_name` (seed 42). Default eval keeps
  only `validation_status == PASS` samples (`--only_passed`; disable with `--no-only_passed`).
- **Metrics:** EM (strict normalized), NormEM (order-insensitive token bag), SoftEM (substring
  either direction), token-F1 (SQuAD-style), chrF (char n-gram F, morphology-robust), EngLeak
  (fraction of non-English answers that returned the English gold — leakage diagnostic).
- **Execution times:** **[TODO: record wall-clock per run — train hrs, eval min/500 samples.]**

---

## 6. Results

### 6.1 Main models (from README — verify/refresh before submission)
| Model | F1 | chrF | NormEM | EM | SoftEM | EngLeak |
|---|---:|---:|---:|---:|---:|---:|
| EN (15k) | 0.921 | 90.18 | 0.844 | 0.838 | 0.912 | 0.000 |
| IT (15k) | 0.881 | 87.67 | 0.790 | 0.772 | 0.838 | 0.006 |
| EN+IT | 0.892 | 88.07 | 0.802 | 0.794 | 0.860 | 0.010 |
| EN+IT+DE+FR | 0.888 | 88.14 | 0.798 | 0.792 | 0.856 | 0.006 |

Interpretation: EN best (in-distribution); multilingual trades ≈3–4 F1 for cross-lingual ability;
FR weakest (NLLB carrier-prefix / isolated-entity noise); `tgqa` hardest (free-form answers);
EngLeak near-zero → model stays in the target language.

### 6.2 RAG ablation (Experiment 1) — **[TODO: run and fill]**
| Cell | F1 | EM | NormEM | chrF | N |
|---|---:|---:|---:|---:|---:|
| base_norag | | | | | |
| base_rag | | | | | |
| ft_norag | | | | | |
| ft_rag | | | | | |

RAG effect (rag − norag): base **[TODO]**, ft **[TODO]**.
Expected narrative: positive Δ on base, ≈0 (or negative) Δ on ft. Also report retrieval hit-rate
at `min_score` and note truncation risk (long exemplar under the 4096 cap).

**Reportable conclusion (fill with real sign/magnitude):** on a *self-contained* benchmark, once
the model is fine-tuned, retrieval adds no evidence it lacks and can only cost context tokens;
retrieval's value is confined to the un-tuned regime. This distinguishes closed-context temporal
reasoning from open-domain temporal QA, where retrieval is beneficial.

---

## 7. Reproducibility notes (matters for 2 pts)
- **Package name is `multilingual_rag_tiser/`** — the README in places says `tiser_lite/` /
  `multilingual_tiser/`; those paths are stale. Fix commands in the README before submission.
- **git-LFS:** every `data/*.json` is LFS-tracked (`data/.gitattributes: *.json filter=lfs`).
  Reproducing requires `git lfs install && git lfs pull`. State this explicitly in the report;
  a grader without git-lfs gets pointer files, not data. Consider narrowing the LFS pattern so
  small config/result JSONs (e.g. `summary.json`) aren't pushed to LFS.
- **Dataset size asymmetry** (from LFS pointers, uncompressed): EN test ≈ 53.7 MB vs IT test
  ≈ 1.5 MB (passed 1.09 MB / failed 0.46 MB). The non-English test pools are much smaller, so a
  balanced 500-sample eval can hit categories with fewer than `500/5=100` samples — this is
  exactly the case the (now-fixed) `balance_by_dataset_name` duplication bug affected. **Re-run
  multilingual evals after the fix** and check whether reported numbers move.
- **Runner scripts:** `run_train.sh`, `run_eval_rag.sh`, `run_experiment1.sh`,
  `preprocess/run_mix_dataset.sh`, `translate/run_translate.sh`, `translate/run_score.sh`.

---

## 8. Conclusions (draft skeleton)
- **Remarkable outcomes:** 3B+QLoRA reaches ≈0.92 F1 on English TISER at 8 GB; multilingual model
  keeps ≈0.89 F1 with near-zero English leakage; RAG **[TODO: substitution finding]**.
- **Challenges:** NLLB translation quality on dates/entities (esp. FR); fitting generation +
  4-bit model in 8 GB; free-form `tgqa` answers; avoiding eval-set contamination in retrieval.
- **Lessons learned:** retrieval is not universally beneficial — its value depends on whether the
  task has an information gap; on self-contained benchmarks, fine-tuning already supplies what
  retrieval would.

---

## 9. Open items before submission
- [ ] Run Experiment 1 on the GPU box → fill §6.2 tables + deltas.
- [ ] Record execution times (§5).
- [ ] Decide `few_shot` vs also reporting `context_stuffing`.
- [ ] Re-run multilingual evals post balance-fix; confirm §6.1 numbers.
- [ ] Fix stale README paths; add git-lfs instructions.
- [ ] Insert GitHub repo URL (abstract/intro).
- [ ] Trim to 4 pages IEEE; move extra tables to appendix.
