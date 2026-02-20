import os
import pickle
import numpy as np
import requests
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

# Paths
DATA_DIR = os.path.join("data")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
DOCS_STORE_PATH = os.path.join(DATA_DIR, "docs_store.pkl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

# Ollama configuration
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

class Document:
    """Simple document class"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def get_embedding(text: str) -> np.ndarray:
    """
    Gets embedding from Ollama API.
    """
    try:
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
        response = requests.post(OLLAMA_EMBED_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            return np.array(embedding, dtype=np.float32)
        else:
            print(f"Ollama embedding error: {response.status_code}")
            return np.zeros(384, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(384, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_vector_results(query: str, k: int = 5, source_filter: str = None) -> List[tuple]:
    """
    Retrieves top-k documents using vector search (cosine similarity).
    Returns list of (doc, score).
    """
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(DOCS_STORE_PATH):
        return []
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Load stored embeddings and texts
        with open(EMBEDDINGS_PATH, "rb") as f:
            stored_embeddings = pickle.load(f)
        
        with open(DOCS_STORE_PATH, "rb") as f:
            data = pickle.load(f)
            all_texts = data.get("texts", [])
            doc_metadata = data.get("metadata", [])
        
        # Compute similarity scores
        scores = []
        for i, embedding in enumerate(stored_embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            # optional source filtering
            md = doc_metadata[i] if i < len(doc_metadata) else {}
            src = md.get("source")
            if source_filter and src != source_filter:
                continue
            scores.append((i, similarity))
        
        # Get top-k
        top_k_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in top_k_indices:
            if idx < len(all_texts):
                doc = Document(
                    page_content=all_texts[idx],
                    metadata=doc_metadata[idx] if idx < len(doc_metadata) else {}
                )
                results.append((doc, score))
        
        return results
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def get_keyword_results(query: str, k: int = 5, source_filter: str = None) -> List[tuple]:
    """
    Retrieves top-k documents from BM25 Index.
    Returns list of (doc, score).
    """
    if not os.path.exists(BM25_PATH) or not os.path.exists(DOCS_STORE_PATH):
        return []
    
    try:
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        
        with open(DOCS_STORE_PATH, "rb") as f:
            data = pickle.load(f)
            all_texts = data.get("texts", [])
            doc_metadata = data.get("metadata", [])
        
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_n:
            if idx < len(all_texts):
                md = doc_metadata[idx] if idx < len(doc_metadata) else {}
                src = md.get("source")
                if source_filter and src != source_filter:
                    continue

                # header boost: if query tokens overlap header tokens, increase score
                header = (md.get("header") or "").lower()
                header_tokens = header.split()
                overlap = len(set(tokenized_query) & set(header_tokens))
                boosted_score = scores[idx]
                if overlap > 0:
                    boosted_score *= (1.0 + 0.5 * overlap)

                doc = Document(
                    page_content=all_texts[idx],
                    metadata=md
                )
                results.append((doc, boosted_score))
        
        return results
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

def reciprocal_rank_fusion(vector_results, keyword_results, k=60, vector_weight: float = 0.6, keyword_weight: float = 0.4):
    """
    Combines results using RRF (Reciprocal Rank Fusion).
    """
    fused_scores = {}
    
    # Normalize and combine scores using weights
    # Build maps
    vec_map = {item[0].page_content: item for item in vector_results}
    kw_map = {item[0].page_content: item for item in keyword_results}

    all_keys = set(list(vec_map.keys()) + list(kw_map.keys()))

    # get raw scores
    vec_scores = [s for (_, s) in vector_results] if vector_results else [0.0]
    kw_scores = [s for (_, s) in keyword_results] if keyword_results else [0.0]
    max_vec = max(vec_scores) if vec_scores else 1.0
    max_kw = max(kw_scores) if kw_scores else 1.0

    for key in all_keys:
        doc = vec_map.get(key, kw_map.get(key))[0]
        v = vec_map.get(key)[1] if key in vec_map else 0.0
        q = kw_map.get(key)[1] if key in kw_map else 0.0
        # normalize
        vn = v / max_vec if max_vec else 0.0
        kn = q / max_kw if max_kw else 0.0
        combined = vector_weight * vn + keyword_weight * kn
        fused_scores[key] = {"doc": doc, "score": combined}
    
    # Sort by fused score
    reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return reranked

def hybrid_search(query: str, k: int = 5, source_filter: str = None) -> List[Dict[str, Any]]:
    """
    Performs hybrid search and returns top-k structured results.
    """
    vector_res = get_vector_results(query, k=k*2, source_filter=source_filter)
    keyword_res = get_keyword_results(query, k=k*2, source_filter=source_filter)

    reranked = reciprocal_rank_fusion(vector_res, keyword_res)
    top_k = reranked[:k]
    
    return [
        {
            "content": item["doc"].page_content,
            "source": item["doc"].metadata.get("source", "unknown"),
            "score": float(item["score"])
        }
        for item in top_k
    ]
