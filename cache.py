import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
except Exception:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

DB_NAME = "harness_cache.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            embedding TEXT,
            response TEXT,
            reliability REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            initial_score REAL,
            final_score REAL,
            retries INTEGER,
            cached INTEGER,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates similarity between two number lists (0.0 = different, 1.0 = identical)."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def check_cache(prompt: str, threshold: float = 0.88):
    """Searches SQLite for any past question with >= 88% meaning similarity."""
    query_vec = embedder.encode(prompt)
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("SELECT prompt, embedding, response, reliability FROM responses").fetchall()
    conn.close()

    best_match = None
    highest_sim = 0.0

    for cached_prompt, emb_json, response, reliability in rows:
        cached_vec = np.array(json.loads(emb_json))
        sim = cosine_similarity(query_vec, cached_vec)
        if sim >= threshold and sim > highest_sim:
            highest_sim = sim
            best_match = {
                "response": response,
                "reliability": reliability,
                "similarity": round(sim, 3),
                "matched_prompt": cached_prompt
            }
    return best_match


def save_to_cache(prompt: str, response: str, reliability: float):
    """Selective Caching: Only saves answers that scored 0.80 or higher."""
    if reliability < 0.80:
        return
    vec_list = embedder.encode(prompt).tolist()
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO responses (prompt, embedding, response, reliability) VALUES (?, ?, ?, ?)",
        (prompt, json.dumps(vec_list), response, reliability)
    )
    conn.commit()
    conn.close()


def log_execution(prompt: str, initial_score: float, final_score: float, retries: int, cached: bool, latency_ms: float):
    """Records run metrics for the Streamlit analytics chart."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO logs (prompt, initial_score, final_score, retries, cached, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
        (prompt, initial_score, final_score, retries, int(cached), latency_ms)
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Testing semantic cache...")
    save_to_cache("What is Python?", "Python is a high-level programming language.", 0.95)
    match = check_cache("Tell me what Python is")
    print("Cache search result:", match)
