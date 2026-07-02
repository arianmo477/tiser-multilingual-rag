from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        
    def build_index(self, documents):
        """
        Build FAISS index from documents.
        documents: list of strings (temporal contexts)
        """
        self.documents = documents
        embeddings = self.encoder.encode(documents, show_progress_bar=False)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
    def retrieve(self, query, top_k=3):
        """
        Retrieve top_k most relevant documents for the query.
        Returns: list of tuples [(doc_id, document_text), ...]
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append((idx + 1, self.documents[idx]))
        
        return results


def simulate_multidoc_from_single(context_text, num_variants=3):
    """
    Helper function to simulate multi-document scenarios from single context.
    This creates slight variations to test conflict resolution.
    For real usage, you'd load actual separate documents.
    """
    import re
    
    variants = []
    base_context = context_text
    
    # Original document
    variants.append(base_context)
    
    # Create variants by shifting some dates slightly
    for i in range(1, num_variants):
        variant = base_context
        # Find year patterns and sometimes shift them
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', variant)
        if years and i < len(years):
            old_year = years[i]
            new_year = str(int(old_year) + i)
            variant = variant.replace(f"at {old_year}", f"at {new_year}", 1)
        variants.append(variant)
    
    return variants