import os
import pickle
import numpy as np
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from backend.embeddings import get_embedding

# Paths
DATA_DIR = os.path.join("data")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
DOCS_STORE_PATH = os.path.join(DATA_DIR, "docs_store.pkl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
_SOURCE_ENTITY_CACHE: Dict[str, str] = {}
_ENTITY_PATTERN = re.compile(
    r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,6}\s+(?:Inc\.?|Ltd\.?|LLC|Corp\.?|Corporation|plc|PLC))"
)


def _extract_entities(text: str) -> List[str]:
    if not text:
        return []
    matches = _ENTITY_PATTERN.findall(text)
    if not matches:
        return []

    banned_fragments = [
        "nasdaq",
        "stock market",
        "securities exchange",
        "new york stock exchange",
        "exchange act",
    ]

    cleaned = []
    for raw in matches:
        name = " ".join(raw.split())
        if not name:
            continue
        lower = name.lower()
        if any(fragment in lower for fragment in banned_fragments):
            continue
        # Avoid over-long captures from legal boilerplate.
        if len(name.split()) > 7:
            continue
        cleaned.append(name)

    if not cleaned:
        return []

    uniq = []
    for name in cleaned:
        if name not in uniq:
            uniq.append(name)
    return uniq


def _extract_entity(text: str) -> str:
    cleaned = _extract_entities(text)
    if not cleaned:
        return ""

    for suffix in ["Inc.", "Inc", "Ltd.", "Ltd", "Corporation", "Corp.", "Corp", "LLC", "PLC", "plc"]:
        for name in cleaned:
            if name.endswith(suffix):
                return name

    return cleaned[0]


def _build_source_entity_cache(all_texts: List[str], doc_metadata: List[Dict[str, Any]]) -> Dict[str, str]:
    global _SOURCE_ENTITY_CACHE
    if _SOURCE_ENTITY_CACHE:
        return _SOURCE_ENTITY_CACHE

    source_counts: Dict[str, Dict[str, int]] = {}
    for idx, text in enumerate(all_texts):
        md = doc_metadata[idx] if idx < len(doc_metadata) else {}
        source = str(md.get("source", "")).strip()
        if not source:
            continue
        if source not in source_counts:
            source_counts[source] = {}

        entities: List[str] = []
        if isinstance(md, dict):
            existing = str(md.get("entity", "")).strip()
            if existing:
                entities.append(existing)
        if not entities:
            entities = _extract_entities(str(text)[:3000])

        for entity in entities:
            source_counts[source][entity] = source_counts[source].get(entity, 0) + 1

    out: Dict[str, str] = {}
    for source, counts in source_counts.items():
        if not counts:
            continue
        ranked = sorted(
            counts.items(),
            key=lambda kv: (
                -kv[1],
                0 if kv[0].endswith("Inc.") else 1,
                len(kv[0]),
            ),
        )
        out[source] = ranked[0][0]

    _SOURCE_ENTITY_CACHE = out
    return _SOURCE_ENTITY_CACHE

class Document:
    """Simple document class"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

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
        if len(query_embedding) == 0 or float(np.linalg.norm(query_embedding)) <= 1e-8:
            return []
        
        # Load stored embeddings and texts
        with open(EMBEDDINGS_PATH, "rb") as f:
            stored_embeddings = pickle.load(f)
        
        with open(DOCS_STORE_PATH, "rb") as f:
            data = pickle.load(f)
            all_texts = data.get("texts", [])
            doc_metadata = data.get("metadata", [])
        source_entity_map = _build_source_entity_cache(all_texts, doc_metadata)
        
        # Compute similarity scores
        scores = []
        for i, embedding in enumerate(stored_embeddings):
            if embedding is None or len(embedding) == 0:
                continue
            if float(np.linalg.norm(embedding)) <= 1e-8:
                continue
            similarity = cosine_similarity(query_embedding, embedding)
            # optional source filtering
            md = doc_metadata[i] if i < len(doc_metadata) else {}
            src = md.get("source")
            if source_filter and src != source_filter:
                continue
            scores.append((i, similarity))

        if not scores:
            return []
        
        # Get top-k
        top_k_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in top_k_indices:
            if idx < len(all_texts):
                metadata = doc_metadata[idx] if idx < len(doc_metadata) else {}
                if source_entity_map and isinstance(metadata, dict):
                    src = str(metadata.get("source", "")).strip()
                    if src and src in source_entity_map and not metadata.get("entity"):
                        metadata = dict(metadata)
                        metadata["entity"] = source_entity_map[src]
                doc = Document(
                    page_content=all_texts[idx],
                    metadata=metadata
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
        source_entity_map = _build_source_entity_cache(all_texts, doc_metadata)
        
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
                if source_entity_map and isinstance(md, dict):
                    src_key = str(src).strip() if src else ""
                    if src_key and src_key in source_entity_map and not md.get("entity"):
                        md = dict(md)
                        md["entity"] = source_entity_map[src_key]

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
            "entity": item["doc"].metadata.get("entity", ""),
            "score": float(item["score"])
        }
        for item in top_k
    ]
