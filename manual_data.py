"""
manual_data.py
Step 2 of RideReady: the knowledge base + retrieval.

Contains real owner-manual chunks (2023 Toyota Camry, OM06259U) with metadata,
and a retrieve() function that embeds the question, embeds each chunk, and
returns the closest chunk by cosine similarity. This is the real RAG layer.
"""

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from the .env file
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"  # cheap, fast, good enough for this

# ---------------------------------------------------------------------------
# The knowledge base: real text copied from the 2023 Toyota Camry Owner's
# Manual (OM06259U). Each chunk carries metadata so retrieval stays scoped to
# the vehicle and the page citation is always available.
# ---------------------------------------------------------------------------
CHUNKS = [
    {
        "make": "Toyota",
        "model": "Camry",
        "year": "2023",
        "section": "Dynamic Radar Cruise Control - Setting the vehicle speed",
        "page": 312,
        "text": (
            "Setting the vehicle speed (vehicle-to-vehicle distance control mode). "
            "Press the cruise control main switch to activate the cruise control. "
            "The dynamic radar cruise control indicator will come on and a message "
            "will be displayed on the multi-information display. Accelerate or "
            "decelerate, with accelerator pedal operation, to the desired vehicle "
            "speed (at or above approximately 20 mph [30 km/h]) and press the "
            "\"SET\" switch to set the speed. The cruise control \"SET\" indicator "
            "will come on. The vehicle speed at the moment the switch is released "
            "becomes the set speed. To change the set speed, press the \"+ RES\" or "
            "\"- SET\" switch until the desired set speed is displayed. Pressing the "
            "switch changes the vehicle-to-vehicle distance: Long, Medium, or Short."
        ),
    },
    {
        "make": "Toyota",
        "model": "Camry",
        "year": "2023",
        "section": "Tire Pressure Warning Light",
        "page": 532,
        "text": (
            "Tire pressure warning light. Indicates the following: low tire pressure "
            "due to a flat tire; low tire pressure due to natural causes; or the tire "
            "pressure warning system is malfunctioning. Immediately stop the vehicle "
            "in a safe place."
        ),
    },
    {
        "make": "Toyota",
        "model": "Camry",
        "year": "2023",
        "section": "Brake System Warning Light",
        "page": 528,
        "text": (
            "Brake system warning light (red). Indicates that the brake fluid level "
            "is low; or the brake system is malfunctioning. Immediately stop the "
            "vehicle in a safe place and contact your Toyota dealer. Continuing to "
            "drive the vehicle may be dangerous."
        ),
    },
]


def _embed(text):
    """Return the embedding vector for a piece of text as a numpy array."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding)


def _cosine(a, b):
    """Cosine similarity between two vectors (1.0 = identical direction)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Embed all chunks once when the module loads, so we don't re-embed every query.
_CHUNK_VECTORS = [_embed(c["text"]) for c in CHUNKS]


def retrieve(question, top_k=1):
    """
    Embed the question, compare against every chunk, return the top_k closest
    chunks along with their similarity score.
    """
    q_vec = _embed(question)
    scored = []
    for chunk, vec in zip(CHUNKS, _CHUNK_VECTORS):
        scored.append((_cosine(q_vec, vec), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Run this file directly (python manual_data.py) to test retrieval in
# isolation, before wiring in the agent or the UI.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "how do I set adaptive cruise control?",
        "what is the tire pressure light?",
        "there's a red brake light on my dash",
    ]
    for q in tests:
        score, chunk = retrieve(q)[0]
        print(f"\nQ: {q}")
        print(f"   -> matched: {chunk['section']} (p.{chunk['page']})  "
              f"[similarity {score:.3f}]")