import os
import shutil
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from backend.ingest import ingest_file, DATA_DIR
from backend.search import hybrid_search
from backend.llm_client import analyze_query, summarize_text
from backend.memory import add_memory, query_memory, clear_memory
from fastapi import Request
import pathlib

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

# Mount frontend
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
# For root access to frontend if needed, or we just serve index on /
# We'll creating a route for / later if we want to serve html directly or use index.html

class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None

@app.get("/")
async def root():
    """Redirect to frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/index.html")

class AnalyzeResponse(BaseModel):
    answer: str
    key_points: List[str]
    confidence: str
    sources: List[dict]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Check file extension
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only .pdf and .txt files are allowed")
        
        file_path = os.path.join(DATA_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            print(f"File saved: {file_path}")

            # Trigger Ingestion in background so upload returns quickly
            thread = threading.Thread(target=ingest_file, args=(file_path,), daemon=True)
            thread.start()

            return {"filename": file.filename, "status": "Indexing started"}
        except Exception as e:
            print(f"Error saving upload: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/query", response_model=AnalyzeResponse)
async def query_documents(request: QueryRequest):
    try:
        # 1. Hybrid Search (optional source filter)
        # retrieve memory entries relevant to the query
        mem_entries = query_memory(request.query, k=3) or []
        mem_context = [
            {"content": m.get("content", ""), "source": "memory", "score": float(m.get("score", 0.0))}
            for m in mem_entries
        ]

        # 1. Hybrid Search (optional source filter)
        doc_context = hybrid_search(request.query, k=5, source_filter=request.source)

        # combine memory + documents (memory first so LLM sees user-specific context)
        context = mem_context + (doc_context or [])

        if not context:
            return {
                "answer": "No relevant documents or memories found.",
                "key_points": [],
                "confidence": "Low",
                "sources": []
            }

        # 2. LLM Analysis
        analysis = analyze_query(request.query, context)

        return {
            "answer": analysis.get("answer", ""),
            "key_points": analysis.get("key_points", []),
            "confidence": analysis.get("confidence", "Low"),
            "sources": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf") or f.endswith(".txt")]
    return files


class MemoryEntry(BaseModel):
    content: str
    role: Optional[str] = "user"


@app.post("/memory/add")
async def memory_add(entry: MemoryEntry):
    try:
        res = add_memory(entry.content, role=entry.role or "user")
        return {"status": "ok", "entry": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/query")
async def memory_query_endpoint(q: QueryRequest):
    try:
        mems = query_memory(q.query, k=5)
        return {"results": mems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
async def memory_clear():
    try:
        clear_memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_chunk")
async def upload_chunk(request: Request):
    """Accepts chunked uploads (form-data): fields: filename, index, total, file (binary)"""
    form = await request.form()
    filename = form.get("filename")
    index = form.get("index")
    total = form.get("total")
    chunk_file = form.get("file")
    if not filename or index is None or total is None or chunk_file is None:
        raise HTTPException(status_code=400, detail="Missing chunk upload fields")

    try:
        index_i = int(index)
        total_i = int(total)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid index/total")

    temp_dir = os.path.join(DATA_DIR, "_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    part_path = os.path.join(temp_dir, f"{filename}.part{index_i}")

    try:
        with open(part_path, "wb") as f:
            shutil.copyfileobj(chunk_file.file, f)
    except Exception as e:
        print(f"Error saving chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving chunk: {e}")

    # If last chunk, assemble
    assembled = False
    if index_i == total_i - 1:
        # check all parts
        parts = [os.path.join(temp_dir, f"{filename}.part{i}") for i in range(total_i)]
        if all(os.path.exists(p) for p in parts):
            dest_path = os.path.join(DATA_DIR, filename)
            with open(dest_path, "wb") as dest:
                for p in parts:
                    with open(p, "rb") as pf:
                        shutil.copyfileobj(pf, dest)
            # cleanup
            for p in parts:
                os.remove(p)
            assembled = True
            # trigger ingestion
            thread = threading.Thread(target=ingest_file, args=(dest_path,), daemon=True)
            thread.start()

    return {"filename": filename, "part": index_i, "assembled": assembled}


@app.get("/status")
async def status(filename: str = None):
    """Return indexing status summary or specific file status"""
    docs_info = {}
    if os.path.exists(DOCS_STORE_PATH):
        try:
            with open(DOCS_STORE_PATH, "rb") as f:
                data = pickle.load(f)
                texts = data.get("texts", [])
                metadata = data.get("metadata", [])
                docs_info = {md.get("source"): True for md in metadata if md.get("source")}
        except Exception:
            docs_info = {}

    if filename:
        return {"filename": filename, "indexed": docs_info.get(filename, False)}
    else:
        return {"indexed_files": list(docs_info.keys())}
