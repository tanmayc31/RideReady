"""
manual_data.py
RideReady knowledge base + retrieval.

Loads the pre-built index from manual_index/index.pkl (created by ingest.py)
and retrieves the closest chunks to a question by cosine similarity, scoped
to the active vehicle. Run `python ingest.py` first to build the index.
"""

import os
import pickle

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = os.path.join("manual_index", "index.pkl")

# ---------------------------------------------------------------------------
# Load the pre-built index once at import (chunks + their embedding vectors).
# ---------------------------------------------------------------------------
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(
        f"No index found at {INDEX_PATH}. Run `python ingest.py` first to "
        f"build it from the PDFs in the manuals/ folder."
    )

with open(INDEX_PATH, "rb") as f:
    _data = pickle.load(f)

CHUNKS = _data["chunks"]              # list of {year, make, model, section, page, text}
_CHUNK_VECTORS = _data["vectors"]     # np.array, one row per chunk


def _embed(text):
    """Return the embedding vector for a piece of text as a numpy array."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding)


def _cosine(a, b):
    """Cosine similarity between two vectors (1.0 = identical direction)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question, top_k=1, vehicle=None):
    """
    Embed the question, compare against every chunk, return the top_k closest
    chunks along with their similarity score.

    If `vehicle` is given (e.g. "2023 Toyota Camry"), only chunks whose
    year/make/model all appear in that string are scored. This scopes retrieval
    to the correct vehicle so a query about another car cannot match this car's
    chunks. If no chunk matches the vehicle, an empty list is returned, which
    the agent treats as "not in this manual" (a safe decline).
    """
    q_vec = _embed(question)
    scored = []
    for chunk, vec in zip(CHUNKS, _CHUNK_VECTORS):
        if vehicle is not None:
            v = vehicle.lower()
            if not (chunk["year"].lower() in v
                    and chunk["make"].lower() in v
                    and chunk["model"].lower() in v):
                continue
        scored.append((_cosine(q_vec, vec), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    tests = [
        "how do I set adaptive cruise control?",
        "what is the tire pressure light?",
        "there's a red brake light on my dash",
    ]
    for q in tests:
        results = retrieve(q, top_k=1, vehicle="2023 Toyota Camry")
        if results:
            score, chunk = results[0]
            print(f"\nQ: {q}")
            print(f"   -> {chunk['section']} (p.{chunk['page']})  [sim {score:.3f}]")
        else:
            print(f"\nQ: {q}\n   -> no match")