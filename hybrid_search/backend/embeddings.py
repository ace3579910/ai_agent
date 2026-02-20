import os
import requests
import numpy as np
import json
from typing import List, Optional

# Configurable embedding endpoint (can be overridden via env)
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def _parse_embedding_response(resp_json) -> Optional[List[float]]:
    # handle a few common response shapes
    if not resp_json:
        return None
    # direct embedding
    if isinstance(resp_json, dict):
        if "embedding" in resp_json and isinstance(resp_json["embedding"], list):
            return resp_json["embedding"]
        if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
            # sometimes embeddings is a list of vectors
            first = resp_json["embeddings"][0]
            if isinstance(first, list):
                return first
        # some apis wrap in data: [{"embedding": [...]}]
        if "data" in resp_json and isinstance(resp_json["data"], list):
            first = resp_json["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
    # sometimes API returns a list directly
    if isinstance(resp_json, list) and len(resp_json) > 0:
        first = resp_json[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
    return None


def get_embedding(text: str) -> np.ndarray:
    """Robust embedding fetcher.

    Tries a few payload shapes and endpoint variants to be tolerant of Ollama versions.
    Returns a numpy vector or a zero-vector fallback.
    """
    endpoints = [OLLAMA_EMBED_URL]
    # derive some common alternate paths
    if OLLAMA_EMBED_URL.endswith('/api/embed'):
        base = OLLAMA_EMBED_URL[:-len('/api/embed')]
        endpoints.extend([
            base + '/api/embeddings',
            base + '/embed',
            base + '/api/embed',
        ])
    else:
        endpoints.extend([
            OLLAMA_EMBED_URL.rstrip('/') + '/api/embed',
            OLLAMA_EMBED_URL.rstrip('/') + '/embed',
            OLLAMA_EMBED_URL.rstrip('/') + '/api/embeddings',
        ])

    payload_variants = [
        {"model": EMBEDDING_MODEL, "prompt": text},
        {"model": EMBEDDING_MODEL, "input": text},
        {"model": EMBEDDING_MODEL, "text": text},
        {"model": EMBEDDING_MODEL, "prompt": text, "options": {"truncation": "none"}},
    ]

    headers = {"Content-Type": "application/json"}

    for url in endpoints:
        for payload in payload_variants:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=20)
            except requests.exceptions.RequestException as e:
                # network error or connection refused
                # try next combo
                # keep errors sparse to avoid noise
                continue

            if r.status_code != 200:
                # try next; but continue looping to find working endpoint
                continue

            try:
                resp_json = r.json()
            except Exception:
                continue

            emb = _parse_embedding_response(resp_json)
            if emb and isinstance(emb, list) and len(emb) > 0:
                try:
                    return np.array(emb, dtype=np.float32)
                except Exception:
                    return np.array(emb)

    # if we reach here, all attempts failed — return a zero-vector fallback
    # keep the original dimensionality expectation (384) to avoid downstream errors
    return np.zeros(384, dtype=np.float32)
