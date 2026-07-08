# NOTICE — Attribution

This project is a **derivative academic work** built on top of **TISER**:

> Adrián Bazaga et al. *Learning to Reason Over Time: Timeline Self-Reflection for
> Improved Temporal Reasoning in Language Models.* Proceedings of ACL 2025.
> Paper: <https://aclanthology.org/2025.acl-long.1358/>
> Original code: <https://github.com/amazon-science/TISER>
> (MIT No Attribution license, Copyright Amazon.com, Inc.)

The upstream TISER code and data are used under their original MIT-0 license — see
[`LICENSE`](LICENSE).

## Our contributions

Developed for the Deep NLP course at Politecnico di Torino, on top of TISER:

- **Small-model adaptation** — Qwen2.5-3B-Instruct fine-tuned with QLoRA (4-bit
  NF4) to run the TISER reasoning pipeline on a single 8 GB GPU.
- **Multilingual extension** — an NLLB-200 translation pipeline with quality
  control (entity caching, hallucination detection, language-specific templates)
  producing Italian, German, and French training data.
- **RAG extension** — retrieval-augmented few-shot / context-stuffing ablation
  over the TISER task.

## Third-party components

The Qwen2.5 model weights, NLLB-200, and the underlying TimeQA / TempReason / TGQA
datasets remain subject to their own respective licenses; consult the original
authors' repositories and model cards.

---

*Note for the team: MIT-0 permits reuse without retaining attribution, so this
NOTICE is added for academic transparency rather than legal necessity. If the
course requires a specific attribution/authorship format, adjust `LICENSE` and
this file accordingly.*
