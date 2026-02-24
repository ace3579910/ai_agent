import os
import sqlite3
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from backend.search import get_embedding, cosine_similarity

DATA_DIR = os.path.join("data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")

os.makedirs(DATA_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            embedding BLOB,
            ts TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def add_memory(content: str, role: str = "user") -> Dict[str, Any]:
    """Store a memory entry (computes and stores embedding).
    Returns the inserted record metadata.
    """
    init_db()
    emb = get_embedding(content)
    emb_blob = pickle.dumps(emb)
    ts = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memory (role, content, embedding, ts) VALUES (?, ?, ?, ?)",
        (role, content, sqlite3.Binary(emb_blob), ts),
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()

    return {"id": rowid, "role": role, "content": content, "ts": ts}


def _load_all() -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, role, content, embedding, ts FROM memory ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        try:
            emb = pickle.loads(r[3]) if r[3] is not None else np.zeros(0, dtype=np.float32)
        except Exception:
            emb = np.zeros(0, dtype=np.float32)
        out.append({
            "id": r[0],
            "role": r[1],
            "content": r[2],
            "embedding": emb,
            "ts": r[4],
        })
    return out


def query_memory(query: str, k: int = 5, roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return top-k relevant memory entries for the given query.
    Each entry contains `content`, `role`, `score`, and `ts`.
    """
    all_mem = _load_all()
    if not all_mem:
        return []

    role_filter = {str(r).strip().lower() for r in (roles or []) if str(r).strip()}
    if role_filter:
        all_mem = [m for m in all_mem if str(m.get("role", "")).strip().lower() in role_filter]
        if not all_mem:
            return []

    q_emb = get_embedding(query)
    if len(q_emb) == 0 or float(np.linalg.norm(q_emb)) <= 1e-8:
        return []

    scored = []
    for m in all_mem:
        emb = m.get("embedding")
        if emb is None or len(emb) == 0:
            continue
        if float(np.linalg.norm(emb)) <= 1e-8:
            continue
        try:
            score = float(cosine_similarity(q_emb, emb))
        except Exception:
            continue
        scored.append({"id": m["id"], "role": m["role"], "content": m["content"], "score": score, "ts": m["ts"]})

    if not scored:
        return []

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
    return scored_sorted


def clear_memory():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM memory")
    conn.commit()
    conn.close()


# initialize on import
init_db()
