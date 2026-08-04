"""
app.py
RideReady: the Streamlit chat UI.

Flow:
  1. On first load, show a centered WELCOME screen that asks the user to pick
     their vehicle (big buttons).
  2. Once a vehicle is chosen, show the normal app: sidebar (vehicle + reset +
     voice toggle) and the chat interface, with grounded answers, read-aloud
     (TTS), and voice questions (Whisper mic).

Run with:  streamlit run app.py
"""

import base64
import uuid

import streamlit as st
from streamlit_mic_recorder import mic_recorder

import agent  # our step-3 LangChain agent

st.set_page_config(page_title="RideReady", page_icon="🚗")

VEHICLES = [
    "2023 Toyota Camry",
    "2022 Honda Accord",
    "2023 Ford Explorer",
    "2024 Hyundai Elantra",
    "2024 Tesla Model3",
]

# --- session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "vehicle" not in st.session_state:
    st.session_state.vehicle = None   # None = show the welcome screen
if "last_mic_id" not in st.session_state:
    st.session_state.last_mic_id = None

# =============================================================================
# WELCOME SCREEN — shown until a vehicle is chosen
# =============================================================================
if st.session_state.vehicle is None:
    st.markdown(
        "<div style='text-align:center; margin-top:8vh;'>"
        "<h1>🚗 Welcome to RideReady</h1>"
        "<h3 style='color:#3A5A7A; font-weight:400;'>Your AI car assistant</h3>"
        "<p style='color:#555; font-size:1.05em; max-width:34rem; margin:1rem auto;'>"
        "Ask plain-language questions about your vehicle and get answers grounded "
        "in the official owner's manual — with a page citation every time."
        "</p>"
        "<p style='font-weight:600; margin-top:1.5rem;'>To get started, choose your vehicle:</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        for v in VEHICLES:
            if st.button(v, key=f"pick_{v}", use_container_width=True):
                st.session_state.vehicle = v
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
    st.stop()   # don't render the chat until a vehicle is picked

# =============================================================================
# MAIN APP — shown once a vehicle is chosen
# =============================================================================

# --- sidebar: vehicle select + reset ----------------------------------------
with st.sidebar:
    st.header("Vehicle")
    selected = st.selectbox(
        "Select your vehicle",
        VEHICLES,
        index=VEHICLES.index(st.session_state.vehicle),
    )
    if selected != st.session_state.vehicle:
        st.session_state.vehicle = selected
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Answers scoped to the {selected} owner's manual.")
    st.divider()
    st.caption(
        "Memory is resettable by design. Resetting clears this "
        "conversation and starts a fresh session."
    )
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_mic_id = None
        st.rerun()
    if st.button("← Change vehicle"):
        st.session_state.vehicle = None
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.session_state.voice_on = st.toggle("🔊 Read answers aloud", value=True)

# --- header ------------------------------------------------------------------
st.title("🚗 RideReady")
st.caption(
    f"Ask about your **{st.session_state.vehicle}**. Answers are grounded in "
    "the owner's manual, with a page citation."
)

# --- example starter questions ----------------------------------------------
if not st.session_state.messages and "pending" not in st.session_state:
    st.write("Try one of these:")
    examples = [
        "How do I set the cruise control?",
        "What does the tire pressure light mean?",
        "There's a warning light on my dash, what is it?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex):
            st.session_state.pending = ex
            st.rerun()

# --- replay the conversation so far -----------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- get input: typed text, clicked example, or voice -----------------------
prompt = st.chat_input(f"Ask about your {st.session_state.vehicle}...")

if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

# --- handle a new message ----------------------------------------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the manual..."):
            answer = agent.ask(
                prompt,
                thread_id=st.session_state.thread_id,
                vehicle=st.session_state.vehicle,
            )
        st.markdown(answer)

        if st.session_state.get("voice_on", True):
            audio_bytes = agent.speak(answer)
            b64 = base64.b64encode(audio_bytes).decode()
            st.iframe(
                f"""
                <div style="display:flex; align-items:center; gap:10px;
                     font-family:sans-serif; color:#3A5A7A; font-size:14px;">
                  <button id="btn" style="border:1px solid #3A5A7A;
                     background:#fff; color:#3A5A7A; border-radius:6px;
                     padding:2px 10px; cursor:pointer; font-size:14px;">
                     ⏸ Pause</button>
                  <span id="ind" style="display:none; align-items:center; gap:8px;">
                    <span style="width:10px; height:10px; border-radius:50%;
                       background:#3A5A7A; display:inline-block;
                       animation:blink 1s infinite;"></span>
                    🔊 Reading aloud…
                  </span>
                </div>
                <style>@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}</style>
                <audio id="a" autoplay>
                  <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                <script>
                  const a = document.getElementById("a");
                  const ind = document.getElementById("ind");
                  const btn = document.getElementById("btn");
                  a.onplay  = () => {{ ind.style.display = "flex"; btn.innerHTML = "⏸ Pause"; }};
                  a.onpause = () => {{ ind.style.display = "none"; btn.innerHTML = "▶ Play"; }};
                  a.onended = () => {{ ind.style.display = "none"; btn.innerHTML = "▶ Play"; }};
                  btn.onclick = () => {{ a.paused ? a.play() : a.pause(); }};
                </script>
                """,
                height=40,
            )

    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- voice input: mic button, rendered LAST so it flows after the answer ----
st.caption("🎤 Or ask by voice:")
audio = mic_recorder(
    start_prompt="🎤 Speak",
    stop_prompt="🔴 Recording… (click to stop)",
    key="mic",
    format="wav",
)

if audio and audio.get("id") != st.session_state.get("last_mic_id"):
    st.session_state.last_mic_id = audio["id"]
    with st.spinner("Transcribing..."):
        text = agent.transcribe(audio["bytes"])
    st.session_state.pending = text
    st.rerun()