# Development Guide

This repository is a **course project** for the Deep NLP course at Politecnico di
Torino — a TISER-based temporal-reasoning study with multilingual and RAG
extensions. It is developed by the project team and is not seeking external
contributions, but the setup and workflow below document how to work with the code.

## Environment setup

```bash
conda env create -f environment.yml
conda activate tiser
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

- **GPU work** (training, inference) needs a CUDA GPU with `bitsandbytes` (4-bit QLoRA).
- **CPU-only work** (RAG index build, retrieval, tests, result aggregation) runs
  without a GPU.
- Datasets are stored with git-LFS — see [`docs/DATA.md`](docs/DATA.md) for what to
  pull and how to rebuild derived artifacts (e.g. the RAG index).

## Running the tests

```bash
python -m pytest tests/
```

Tests cover the class-balanced sampler (`utils/sampling.py`) and the evaluation
metrics (`utils/metrics.py`); they need no GPU or dataset.

## Repository layout

| Path | Purpose |
|---|---|
| `multilingual_rag_tiser/` | training, evaluation (incl. RAG), translation, preprocessing |
| `utils/` | shared metrics, prompt building, sampling, I/O, translation helpers |
| `data/` | prompts and (git-LFS) datasets — see `docs/DATA.md` |
| `docs/` | data guide and report draft |
| `tests/` | unit tests |

## Conventions

- Keep changes focused; run `python -m pytest tests/` before committing.
- Don't commit trained adapters, RAG indexes, or other regenerable artifacts —
  they are gitignored / documented in `docs/DATA.md`.
