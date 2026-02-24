import os
import pickle
import shutil
import numpy as np
import json
from typing import List, Dict, Any
from backend.embeddings import get_embedding

# PDF reading
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Paths
DATA_DIR = os.path.join("data")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
DOCS_STORE_PATH = os.path.join(DATA_DIR, "docs_store.pkl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

# Ensure data structure exists
os.makedirs(DATA_DIR, exist_ok=True)

class Document:
    """Simple document class"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def load_pdf(file_path: str) -> str:
    """
    Loads text from a PDF file.
    """
    if not PdfReader:
        print("PyPDF2 not installed. Skipping PDF loading.")
        return ""
    
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return ""

def load_text(file_path: str) -> str:
    """
    Loads text from a TXT file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading TXT {file_path}: {e}")
        return ""

def load_documents(files: List[str]) -> List[Document]:
    """
    Loads documents from a list of file paths (PDF or TXT).
    """
    documents = []
    for file_path in files:
        try:
            content = ""
            if file_path.lower().endswith(".pdf"):
                content = load_pdf(file_path)
            elif file_path.lower().endswith(".txt"):
                content = load_text(file_path)
            
            if content:
                doc = Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path)}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    """
    Splits documents into overlapping chunks and extracts a simple header for each chunk.
    """
    chunks = []
    for doc in documents:
        text = doc.page_content

        def extract_header_for_offset(full_text, start_idx):
            # look backwards up to 1000 chars for a header-like line
            start = max(0, start_idx - 1000)
            window = full_text[start:start_idx]
            lines = window.splitlines()
            for line in reversed(lines):
                ln = line.strip()
                if not ln:
                    continue
                # markdown headers
                if ln.startswith('#'):
                    return ln.lstrip('#').strip()
                # ALL CAPS short line
                if len(ln) <= 120 and ln == ln.upper() and any(c.isalpha() for c in ln):
                    return ln
                # lines that end with ':' look like section headers
                if ln.endswith(':') and len(ln) < 120:
                    return ln.rstrip(':').strip()
            return ""

        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                header = extract_header_for_offset(text, i)
                metadata = doc.metadata.copy()
                metadata.update({"header": header})
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                chunks.append(chunk_doc)
    return chunks

def build_indices(chunks: List[Document]):
    """
    Builds Vector Index and Keyword Index (BM25).
    """
    from rank_bm25 import BM25Okapi
    
    # 1. Vector Index with Ollama embeddings
    print("Building vector embeddings...")
    embeddings = []
    all_texts = []
    doc_metadata = []
    
    # Load existing data if any
    if os.path.exists(DOCS_STORE_PATH):
        with open(DOCS_STORE_PATH, "rb") as f:
            existing_data = pickle.load(f)
            all_texts = existing_data.get("texts", [])
            doc_metadata = existing_data.get("metadata", [])
    
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
    
    # Get embeddings for new chunks in parallel to speed up processing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def embed_and_collect(idx_chunk):
        i, chunk = idx_chunk
        try:
            emb = get_embedding(chunk.page_content)
        except Exception as e:
            print(f"Embedding error for chunk {i}: {e}")
            emb = np.zeros(384, dtype=np.float32)
        return (i, emb, chunk.page_content, chunk.metadata)

    if chunks:
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = [exe.submit(embed_and_collect, (i, c)) for i, c in enumerate(chunks)]
            for fut in as_completed(futures):
                i, emb, text, meta = fut.result()
                embeddings.append(emb)
                all_texts.append(text)
                doc_metadata.append(meta)
    
    # Save embeddings and metadata
    vector_store = {
        "embeddings": embeddings,
        "texts": all_texts,
        "metadata": doc_metadata
    }
    
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    # 2. BM25 Index for keyword search
    print("Building BM25 index with header boosting...")
    # If a chunk has a header, boost header tokens by repeating them in the text used for BM25
    boosted_texts = []
    for i, text in enumerate(all_texts):
        header = ""
        try:
            header = doc_metadata[i].get("header", "")
        except Exception:
            header = ""
        if header:
            # repeat header words 3 times to increase their weight
            boosted = f"{header} {header} {header} {text}"
        else:
            boosted = text
        boosted_texts.append(boosted)

    tokenized_corpus = [t.lower().split() for t in boosted_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    
    # Save all data
    final_data = {
        "texts": all_texts,
        "metadata": doc_metadata
    }
    
    with open(DOCS_STORE_PATH, "wb") as f:
        pickle.dump(final_data, f)
    
    print(f"Indexed {len(chunks)} new chunks. Total corpus size: {len(all_texts)}")

def ingest_file(file_path: str):
    """
    End-to-end ingestion trigger.
    """
    print(f"Ingesting {file_path}...")
    docs = load_documents([file_path])
    chunks = chunk_documents(docs)
    build_indices(chunks)
    print("Ingestion complete.")
