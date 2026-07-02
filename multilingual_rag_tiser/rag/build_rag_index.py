#!/usr/bin/env python3
"""
Build a FAISS index for RAG few-shot exemplar retrieval.

Documents are encoded by QUESTION TEXT (not temporal_context),
using a multilingual sentence encoder. The index uses IndexFlatIP
with L2-normalized embeddings, so search scores are cosine similarities.

Each document in documents.json stores the full sample so inference.py
can format complete few-shot examples at retrieval time.

Usage:
    python multilingual_rag_tiser/rag/build_rag_index.py \\
        --input  data/splits/train/TISER_train_de_it_fr_en_mixed.json \\
        --output_dir data/rag/train_mixed \\
        --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    # Language-filtered (only index Italian samples for Italian retrieval):
    python multilingual_rag_tiser/rag/build_rag_index.py \\
        --input  data/splits/train/TISER_train_it.json \\
        --output_dir data/rag/train_it \\
        --language it
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def is_corrupt(text: str, threshold: int = 4) -> bool:
    """Return True if the text has too many empty parentheses — broken translation."""
    return text.count("()") > threshold


def select_text(sample: dict, field: str) -> str:
    """
    Prefer the target-language field; fall back to English if corrupt or missing.
    """
    native = (sample.get(field) or "").strip()
    english = (sample.get(f"{field}_en") or "").strip()

    if not native or is_corrupt(native):
        return english
    return native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Training JSON to index.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to save index.faiss and documents.json.")
    ap.add_argument(
        "--model_name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence encoder. Must be multilingual if the data is multilingual.",
    )
    ap.add_argument(
        "--language", default=None,
        help="If set, only index samples with this language value.",
    )
    ap.add_argument(
        "--batch_size", type=int, default=256,
        help="Encoding batch size.",
    )
    ap.add_argument(
        "--min_output_len", type=int, default=50,
        help="Skip samples whose output trace is shorter than this (probably corrupt).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and filter
    # -----------------------------------------------------------------------
    print(f"Loading: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw)} raw samples")

    documents = []
    skipped_lang = 0
    skipped_corrupt = 0

    for s in raw:
        # Language filter
        if args.language and s.get("language") not in (args.language, None):
            skipped_lang += 1
            continue

        question = select_text(s, "question")
        context = select_text(s, "temporal_context")
        output = select_text(s, "output")
        answer = select_text(s, "answer")

        # Skip if question or output are empty / corrupt
        if not question or not output:
            skipped_corrupt += 1
            continue
        if len(output) < args.min_output_len:
            skipped_corrupt += 1
            continue
        if is_corrupt(question, threshold=3):
            skipped_corrupt += 1
            continue

        documents.append({
            "question_id": s.get("question_id", ""),
            "dataset_name": s.get("dataset_name", ""),
            "language": s.get("language", "en"),
            "question": question,
            "question_en": s.get("question_en", question),
            "temporal_context": context,
            "temporal_context_en": s.get("temporal_context_en", context),
            "answer": answer,
            "answer_en": s.get("answer_en", answer),
            "output": output,
            "output_en": s.get("output_en", output),
        })

    print(f"  Kept: {len(documents)}")
    print(f"  Skipped (language filter): {skipped_lang}")
    print(f"  Skipped (corrupt / empty): {skipped_corrupt}")

    if not documents:
        raise ValueError("No valid documents to index. Check your input file and --language filter.")

    # -----------------------------------------------------------------------
    # Encode with question text
    # -----------------------------------------------------------------------
    print(f"\nLoading encoder: {args.model_name}")
    encoder = SentenceTransformer(args.model_name)

    texts = [d["question"] for d in documents]
    print(f"Encoding {len(texts)} questions...")

    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # L2-normalized → IndexFlatIP = cosine similarity
        show_progress_bar=True,
    ).astype("float32")

    print(f"  Embedding shape: {embeddings.shape}")

    # -----------------------------------------------------------------------
    # Build FAISS index
    # IndexFlatIP + normalized vectors = exact cosine similarity search
    # -----------------------------------------------------------------------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"  FAISS index: {index.ntotal} vectors, dim={dimension}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    index_path = out_dir / "index.faiss"
    docs_path = out_dir / "documents.json"

    faiss.write_index(index, str(index_path))
    print(f"Saved index: {index_path}")

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"Saved documents: {docs_path}")

    # -----------------------------------------------------------------------
    # Sanity check: retrieve a few examples
    # -----------------------------------------------------------------------
    print("\nSanity check — top-3 results for first document:")
    q_emb = embeddings[0:1]
    scores, idxs = index.search(q_emb, 4)
    for score, idx in zip(scores[0], idxs[0]):
        if idx == 0:
            continue
        print(f"  [{score:.4f}] {documents[idx]['question'][:80]}")

    print("\nDone.")
    print(f"Index: {index_path}")
    print(f"Docs:  {docs_path}")


if __name__ == "__main__":
    main()