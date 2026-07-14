# 🚗 RideReady

An AI conversational assistant that helps drivers of unfamiliar vehicles
(rentals, fleet, newly bought) understand their car by answering questions in
plain language — grounded strictly in the vehicle's official owner manual via
Retrieval-Augmented Generation (RAG), with a mandatory page citation on every
answer.

Built for **INFO 7375: Building AI Applications** (Summer 2026, Northeastern
University). Group 5 — Anjana Sruthi Ranga & Tanmay Vilas Chandan.

---

## What it does

Ask a question about the vehicle (typed in a chat UI) and RideReady:

1. decides to call a **retrieval tool** to search the owner's manual (ReAct /
   tool-calling pattern),
2. retrieves the most relevant manual section(s) using OpenAI embeddings +
   cosine similarity,
3. answers in a fixed, glance-able format — **Answer / Steps / Severity /
   Source** — grounded only in the retrieved text,
4. remembers the conversation, so multi-turn follow-ups work (e.g. a vague
   "there's a warning light" → asks the color → identifies it on the next turn).

If the manual doesn't contain the answer, it declines rather than guessing.

The current demo vehicle is a **2023 Toyota Camry** (Owner's Manual OM06259U),
with three sections indexed: Dynamic Radar Cruise Control (p.312), Tire Pressure
Warning Light (p.532), and Brake System Warning Light (p.528).

---

## Tech stack

- **Python**
- **LangChain / LangGraph** — orchestration (the agent + tool loop + memory)
- **OpenAI** — chat model (`gpt-4o-mini`) + embeddings (`text-embedding-3-small`)
- **Streamlit** — the chat UI

Single-provider by design: chat, embeddings, and (planned) voice all run on
OpenAI.

---

## Project layout

| File | What it is |
| --- | --- |
| `manual_data.py` | The RAG layer: real manual chunks + embedding + cosine-similarity retrieval. |
| `agent.py` | The LangChain agent: V3 system prompt + the `retrieve_manual` tool + conversation memory. |
| `app.py` | The Streamlit chat UI (a face on top of the agent). |
| `.env` | Your OpenAI API key. **Not committed** (see below). |

---

## Setup

> **You need your own OpenAI API key.** The `.env` file is intentionally
> excluded from git (see `.gitignore`), so it will **not** download when you
> clone. Each person creates their own. Without it, the app cannot make any
> calls and will error on startup.

**1. Clone the repo**

```bash
git clone https://github.com/tanmayc31/RideReady.git
cd RideReady
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**3. Install the dependencies**

```bash
pip install openai langchain langchain-openai streamlit numpy python-dotenv
```

**4. Add your OpenAI API key**

Create a file named `.env` in the project root with one line:

```
OPENAI_API_KEY=sk-your-key-here
```

Get a key at [platform.openai.com](https://platform.openai.com) and make sure a
small credit balance is added under Billing (a few dollars is far more than
enough — usage is fractions of a cent per question).

---

## Running it

**The full app (chat UI):**

```bash
streamlit run app.py
```

This opens a browser tab (usually `http://localhost:8501`). Try a starter
question, or ask your own about the 2023 Camry.

**Test the pieces on their own (optional):**

```bash
python manual_data.py    # prints retrieval matches + similarity scores
python agent.py          # runs the three demo use cases in the terminal
```

---

## Demo use cases

1. **Cruise setup** — "How do I set the cruise control?" → numbered Steps + p.312.
2. **Warning light + severity** — "What does the tire pressure light mean?" →
   Severity + p.532.
3. **Two-turn memory** — "There's a warning light on my dash, what is it?" →
   asks the color → "it's red" → Brake System Warning, *Stop safely & check
   now*, p.528. (Answering "yellow" instead correctly resolves to the Tire
   Pressure light with *Caution* — the disambiguation reasons about the color.)

---

## Notes

- Voice input (Whisper) and audio output (TTS) are designed but not wired into
  this build — text only for now.
- Manual chunks were copied by hand from the official 2023 Toyota Camry Owner's
  Manual (OM06259U) with real page numbers, so every citation is verifiable.

---

*INFO 7375: Building AI Applications | Summer 2026 | Northeastern University |
College of Engineering*
