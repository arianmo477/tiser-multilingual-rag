"""
RAG utilities for TISER — two retrieval modes for ablation study.

Two modes, both expected to HURT performance on TISER (and reported as such):

  few_shot (default)
    Retrieve similar solved training examples by question similarity.
    Prepend them as few-shot demonstrations before the main prompt.
    Hurts because: the model already learned the reasoning format from
    fine-tuning; extra demonstrations waste context tokens.

  context_stuffing
    Retrieve similar training examples and append their temporal_context
    to the test sample's own temporal_context.
    Hurts more because: retrieved contexts describe different entities at
    different times — directly contradictory temporal facts injected into
    what is a self-contained, closed-context benchmark.

Both are included to show that TISER's self-contained design makes RAG
inappropriate regardless of architecture. This is a reportable negative
finding that distinguishes self-contained temporal reasoning from
open-domain temporal QA where retrieval is beneficial.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
# Retriever
# ============================================================================

class RAGRetriever:
    """
    Retrieve training examples from a pre-built FAISS index.

    The index encodes question text (not temporal_context), so retrieval
    finds structurally similar questions regardless of entity content.
    IndexFlatIP with L2-normalized embeddings = cosine similarity.
    """

    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_dir = Path(index_dir)
        index_path = self.index_dir / "index.faiss"
        docs_path = self.index_dir / "documents.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents file not found: {docs_path}")

        print(f"Loading RAG index from: {index_dir}")
        self.index = faiss.read_index(str(index_path))

        with open(docs_path, encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"  {len(self.documents)} documents indexed")
        print(f"Loading RAG encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 1,
        min_score: float = 0.55,
        exclude_question_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k training examples similar to `query`.

        Args:
            query:               Question text of the target sample.
            top_k:               Max examples to return.
            min_score:           Cosine similarity floor. 0.50–0.65 is
                                 appropriate for multilingual models.
                                 Do NOT use 0.9 — too high for cross-lingual.
            exclude_question_id: Skip documents with this id (prevents
                                 test-set leakage when evaluating on
                                 training-domain samples).
            language:            If set, only return documents whose
                                 language matches (documents with no
                                 language tag are always allowed). Keeps a
                                 mixed-language index from returning a
                                 cross-lingual exemplar.

        Returns:
            List of dicts with keys: question, temporal_context, answer,
            output, language, question_id, dataset_name, score.
        """
        if not query or not query.strip():
            return []

        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Over-fetch so we can filter
        search_k = min(top_k * 4, len(self.documents))
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            score = float(score)
            if score < min_score:
                continue
            doc = self.documents[idx]
            if exclude_question_id and doc.get("question_id") == exclude_question_id:
                continue
            # Language filter: skip cross-lingual hits on a mixed index.
            # Documents with no language tag are always allowed.
            if language and doc.get("language") and doc.get("language") != language:
                continue
            # Skip documents with corrupt output (broken translations)
            output = doc.get("output", "") or ""
            if not output.strip() or output.count("()") > 5:
                continue
            results.append({**doc, "score": score})
            if len(results) >= top_k:
                break

        return results


# ============================================================================
# Few-shot formatting helpers
# ============================================================================

_FEW_SHOT_HEADERS = {
    "en": (
        "Here is a similar solved example for reference.",
        "Now solve this new question:",
    ),
    "it": (
        "Ecco un esempio simile già risolto come riferimento.",
        "Ora risolvi questa nuova domanda:",
    ),
    "de": (
        "Hier ist ein ähnliches gelöstes Beispiel als Referenz.",
        "Löse jetzt diese neue Frage:",
    ),
    "fr": (
        "Voici un exemple similaire déjà résolu pour référence.",
        "Résous maintenant cette nouvelle question :",
    )
    
}

_STUFFING_HEADERS = {
    "en": "Additional temporal context (may not be relevant):",
    "it": "Contesto temporale aggiuntivo (potrebbe non essere rilevante):",
    "de": "Zusätzlicher zeitlicher Kontext (möglicherweise nicht relevant):",
    "fr": "Contexte temporel supplémentaire (peut ne pas être pertinent) :",
    
}


def format_few_shot_prefix(
    retrieved_docs: List[Dict[str, Any]],
    lang: str = "en",
) -> str:
    """
    Format retrieved examples as a few-shot prefix.

    Shows the model HOW to reason about a similar question — not
    additional temporal facts. Falls back to English content if the
    native translation looks corrupt (too many empty parens).

    Returns empty string if no valid examples could be formatted.
    """
    if not retrieved_docs:
        return ""

    intro_header, outro_header = _FEW_SHOT_HEADERS.get(
        lang, _FEW_SHOT_HEADERS["en"]
    )

    blocks = [intro_header, ""]

    for i, doc in enumerate(retrieved_docs, start=1):
        question = (doc.get("question") or doc.get("question_en", "")).strip()
        context = (doc.get("temporal_context") or
                   doc.get("temporal_context_en", "")).strip()
        output = (doc.get("output") or doc.get("output_en", "")).strip()
        answer = (doc.get("answer") or doc.get("answer_en", "")).strip()

        if not question or not output:
            continue

        # Fall back to English if translation is corrupt
        if question.count("()") > 3:
            question = doc.get("question_en", question)
        if context.count("()") > 3:
            context = doc.get("temporal_context_en", context)
        if output.count("()") > 3:
            output = doc.get("output_en", output)
        if not answer or answer.count("()") > 1:
            answer = doc.get("answer_en", answer)

        if len(retrieved_docs) > 1:
            blocks.append(
                f"=== Example {i} (similarity: {doc.get('score', 0):.2f}) ==="
            )
        else:
            blocks.append(
                f"=== Example (similarity: {doc.get('score', 0):.2f}) ==="
            )

        blocks.append(f"Question: {question}")
        if context:
            blocks.append(f"Temporal context: {context}")
        blocks.append("")
        blocks.append(output)
        blocks.append("")

    if len(blocks) <= 2:
        return ""

    blocks.append("=" * 40)
    blocks.append(outro_header)
    blocks.append("")

    return "\n".join(blocks)


def format_stuffed_context(
    retrieved_docs: List[Dict[str, Any]],
    lang: str = "en",
) -> str:
    """
    Format retrieved temporal contexts for context-stuffing mode.

    Appends retrieved contexts to the sample's own temporal_context.
    This is the naive RAG approach — expected to HURT on TISER because
    retrieved contexts describe different entities at different times,
    injecting contradictory temporal facts into a self-contained context.

    Returns empty string if nothing usable was retrieved.
    """
    if not retrieved_docs:
        return ""

    header = _STUFFING_HEADERS.get(lang, _STUFFING_HEADERS["en"])
    blocks = [header]

    for doc in retrieved_docs:
        ctx = (doc.get("temporal_context") or
               doc.get("temporal_context_en", "")).strip()
        # Skip corrupt entries
        if not ctx or ctx.count("()") > 3:
            ctx = doc.get("temporal_context_en", "").strip()
        if ctx:
            # Tag the source so it's visible in the reasoning trace
            dataset = doc.get("dataset_name", "")
            score = doc.get("score", 0)
            blocks.append(
                f"[Retrieved from {dataset}, similarity={score:.2f}]"
            )
            blocks.append(ctx)

    if len(blocks) <= 1:
        return ""

    return "\n".join(blocks)


# ============================================================================
# Main builder used in inference.py
# ============================================================================

class RAGContextBuilder:
    """
    Drop-in wrapper for use in inference.py.

    Supports two modes:

      "few_shot" (default)
        Returns a prefix string to prepend before the prompt.
        Usage:
            prefix, modified_sample = rag.build(sample)
            prompt = build_prompt(modified_sample, prompt=prefix + base_prompt)

      "context_stuffing"
        Returns a modified sample with extra content in temporal_context.
        Usage:
            prefix, modified_sample = rag.build(sample)
            prompt = build_prompt(modified_sample, prompt=base_prompt)

    In both modes, prefix and/or modified_sample may be None/empty if
    no relevant documents were retrieved above min_score.
    """

    VALID_MODES = ("few_shot", "context_stuffing")

    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 1,
        min_score: float = 0.55,
        mode: str = "few_shot",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid RAG mode '{mode}'. Choose from: {self.VALID_MODES}"
            )

        self.top_k = top_k
        self.min_score = min_score
        self.mode = mode
        self.retriever = RAGRetriever(index_dir=index_dir, model_name=model_name)

        print(f"RAG mode: {self.mode}")

    def build(
        self,
        sample: Dict[str, Any],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build RAG-augmented inputs for this sample.

        Returns:
            (few_shot_prefix, modified_sample)

            few_shot mode:
                few_shot_prefix  — prepend to base_prompt (may be "")
                modified_sample  — same as input sample (unchanged)

            context_stuffing mode:
                few_shot_prefix  — always "" (not used)
                modified_sample  — copy of sample with augmented
                                   temporal_context (unchanged if no hit)
        """
        query = (sample.get("question") or "").strip()
        lang = sample.get("language", "en") or "en"
        qid = sample.get("question_id")

        retrieved = self.retriever.retrieve(
            query=query,
            top_k=top_k if top_k is not None else self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            exclude_question_id=qid,
            # "mixed" is a corpus tag, not a real language → don't filter.
            language=None if lang == "mixed" else lang,
        )

        if not retrieved:
            return "", dict(sample)

        if self.mode == "few_shot":
            prefix = format_few_shot_prefix(retrieved, lang=lang)
            return prefix, dict(sample)

        # context_stuffing mode
        extra = format_stuffed_context(retrieved, lang=lang)
        if not extra:
            return "", dict(sample)

        modified = dict(sample)
        base_ctx = (sample.get("temporal_context") or "").strip()
        modified["temporal_context"] = (
            base_ctx + "\n\n" + extra if base_ctx else extra
        )
        return "", modified