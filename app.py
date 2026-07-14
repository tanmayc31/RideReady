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

# --- sidebar: vehicle profile + reset ---------------------------------------
with st.sidebar:
    st.header("Vehicle")
    st.write("**2023 Toyota Camry**")
    st.caption("Owner's Manual OM06259U")
    st.divider()
    st.caption(
        "Memory is resettable by design. Resetting clears this "
        "conversation and starts a fresh session."
    )
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# --- header ------------------------------------------------------------------
st.title("🚗 RideReady")
st.caption("Ask about your vehicle. Answers are grounded in the owner's "
           "manual, with a page citation.")

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
prompt = st.chat_input("Ask about your 2023 Toyota Camry...")
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
            answer = agent.ask(prompt, thread_id=st.session_state.thread_id)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})