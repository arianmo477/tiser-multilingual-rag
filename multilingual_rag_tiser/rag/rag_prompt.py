#!/usr/bin/env python3
"""
Few-shot RAG prompt utilities for TISER.

This does NOT append retrieved temporal contexts as extra evidence.
Instead, it retrieves a similar solved training example and prepends it as
a demonstration.

This is better for TISER because every target sample already has its own
complete temporal_context.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.docs_path = self.index_dir / "documents.json"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documents file not found: {self.docs_path}")

        print(f"Loading RAG index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Loading RAG documents: {self.docs_path}")
        with open(self.docs_path, encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"Loading RAG embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 1,
        min_score: float = 0.55,
        language: str | None = None,
        exclude_question_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # Retrieve more than top_k because we may filter by language/id.
        search_k = max(top_k * 10, top_k)

        q_emb = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, search_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            score = float(score)
            if score < min_score:
                continue

            doc = self.documents[idx]

            if language and doc.get("language") and doc.get("language") != language:
                continue

            if exclude_question_id and doc.get("question_id") == exclude_question_id:
                continue

            doc = dict(doc)
            doc["score"] = score
            doc["doc_id"] = int(idx)
            results.append(doc)

            if len(results) >= top_k:
                break

        return results


def format_example(doc: Dict[str, Any], n: int) -> str:
    question = str(doc.get("question", "") or "").strip()
    context = str(doc.get("temporal_context", "") or "").strip()
    answer = str(doc.get("answer", "") or "").strip()

    output = str(doc.get("output", "") or "").strip()

    # Keep examples short. Full output can be too long/noisy.
    if output and len(output) <= 1800:
        solved = output
    else:
        solved = f"<answer>\n{answer}\n</answer>"

    return (
        f"Retrieved solved example {n}:\n"
        f"Question:\n{question}\n\n"
        f"Temporal context:\n{context}\n\n"
        f"Correct output:\n{solved}"
    )


def fewshot_instruction(lang: str) -> str:
    if lang == "it":
        return (
            "Esempio risolto recuperato. Usalo solo come dimostrazione del formato "
            "e del tipo di ragionamento. Non copiare la risposta dell'esempio."
        )

    if lang == "de":
        return (
            "Abgerufenes gelöstes Beispiel. Verwende es nur als Demonstration des "
            "Formats und der Denkweise. Kopiere nicht die Antwort des Beispiels."
        )

    if lang == "fr":
        return (
            "Exemple résolu récupéré. Utilisez-le seulement comme démonstration du "
            "format et du raisonnement. Ne copiez pas la réponse de l'exemple."
        )

    return (
        "Retrieved solved example. Use it only as a demonstration of the format "
        "and reasoning style. Do not copy the example answer."
    )


def add_fewshot_to_sample(
    sample: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sample = dict(sample)

    if not retrieved_docs:
        sample["rag_num_docs"] = 0
        sample["rag_scores"] = []
        return sample

    lang = sample.get("language", "en") or "en"

    examples = [
        format_example(doc, i)
        for i, doc in enumerate(retrieved_docs, start=1)
    ]

    fewshot_block = "\n\n---\n\n".join(
        [fewshot_instruction(lang)] + examples
    )

    original_question = str(sample.get("question", "") or "").strip()

    sample["question"] = (
        fewshot_block
        + "\n\n---\n\n"
        + "Now answer the target question below.\n\n"
        + original_question
    )

    sample["rag_num_docs"] = len(retrieved_docs)
    sample["rag_scores"] = [float(d.get("score", 0.0)) for d in retrieved_docs]
    sample["rag_doc_ids"] = [d.get("question_id", "") for d in retrieved_docs]

    return sample


class RAGContextBuilder:
    """
    Used by inference.py.

    It keeps the same method name add_to_sample() so you do not need to change
    your inference pipeline.
    """

    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 1,
        min_score: float = 0.55,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.retriever = RAGRetriever(index_dir=index_dir, model_name=model_name)

    def add_to_sample(
        self,
        sample: Dict[str, Any],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> Dict[str, Any]:
        lang = sample.get("language", "en") or "en"

        query = str(sample.get("question", "") or "").strip()

        docs = self.retriever.retrieve(
            query=query,
            top_k=top_k if top_k is not None else self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            language=None if lang == "mixed" else lang,
            exclude_question_id=sample.get("question_id", None),
        )

        return add_fewshot_to_sample(sample, docs)
