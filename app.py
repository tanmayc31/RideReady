"""
app.py
Step 4 of RideReady: the Streamlit chat UI.

This is just a face on top of agent.py. It calls ask() for every user
message, keeps the conversation on screen, and preserves memory across turns
in a session (so the two-turn brake-light demo works right in the browser).

Run with:  streamlit run app.py
"""

import uuid

import streamlit as st

import agent  # our step-3 LangChain agent

st.set_page_config(page_title="RideReady", page_icon="🚗")

# --- session state -----------------------------------------------------------
# messages: the visible chat history (list of {role, content})
# thread_id: a unique id so the agent's memory groups this conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "vehicle" not in st.session_state:
    st.session_state.vehicle = "2023 Toyota Camry"

# --- sidebar: vehicle select + reset ---------------------------------------
VEHICLES = ["2023 Toyota Camry", "2022 Honda Accord"]

with st.sidebar:
    st.header("Vehicle")
    selected = st.selectbox(
        "Select your vehicle",
        VEHICLES,
        index=VEHICLES.index(st.session_state.get("vehicle", VEHICLES[0])),
    )
    # If the user switches vehicles, reset the conversation so memory doesn't
    # carry over answers from the previous car.
    if selected != st.session_state.get("vehicle"):
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
        st.rerun()

    st.session_state.voice_on = st.toggle("🔊 Read answers aloud", value=True)



# --- header ------------------------------------------------------------------
st.title("🚗 RideReady")
st.caption(
    "Ask about your vehicle. Answers are grounded in the owner's "
    "manual, with a page citation."
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

# --- get input: either a clicked example or typed text ----------------------
prompt = st.chat_input(f"Ask about your {st.session_state.vehicle}...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

# --- handle a new message ----------------------------------------------------
if prompt:
    # show the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # get the agent's answer, reusing the session's memory thread
    with st.chat_message("assistant"):
        with st.spinner("Checking the manual..."):
            answer = agent.ask(
                prompt,
                thread_id=st.session_state.thread_id,
                vehicle=st.session_state.vehicle,
            )
        st.markdown(answer)

        # Speak the answer aloud with a synced "reading aloud" indicator.        
        if st.session_state.get("voice_on", True):
            import base64
            audio = agent.speak(answer)
            b64 = base64.b64encode(audio).decode()
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
