import time

import requests
import streamlit as st

# ---- CONFIGURATION ----
API_URL = "http://localhost:8000/api/v1/query"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(page_title="RAG Pipeline", page_icon="🤖", layout="centered")

# ---- SESSION STATE INITIALIZATION -----
if "backend_connected" not in st.session_state:
    st.session_state.backend_connected = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your Technical Assistant. I'm ready to answer questions.",
        }
    ]

# --- STARTUP HEALTH CHECK ---
# This block runs only if we haven't confirmed the connection yet.
if not st.session_state.backend_connected:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("### Connecting to Knowledge Engine...")
        progress = st.progress(0)

        # Try connection
        try:
            for i in range(1, 101, 20):
                time.sleep(0.05)
                progress.progress(i)

            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                st.session_state.backend_connected = True
                progress.progress(100)
                time.sleep(0.5)
                placeholder.empty()
                st.rerun()
            else:
                st.error("Backend is offline. Please wait until we resolve the error.")
                if st.button("Retry Connection"):
                    st.rerun()
                st.stop()

        except requests.exceptions.ConnectionError:
            st.error("could not reach the server.")
            if st.button("Retry Connection"):
                st.rerun()
            st.stop()

# --- CHAT INTERFACE ----

st.title("RAG Pipeline")
st.caption("Context-Aware Technical Q&A System")

with st.sidebar:
    st.success("System Online")
    st.markdown("---")
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat Cleared! How can I help you?",
            }
        ]
        st.rerun()

# Displaying chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User Input
if prompt := st.chat_input("Ask about vectors, attention, or RAG..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("🧠 Analyzing... "):
            try:
                payload = {
                    "query": prompt,
                    "k": 5,
                    "use_hybrid_search": True,
                }

                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("Sources", [])

                    full_response = answer
                    message_placeholder.markdown(full_response)

                    if sources:
                        with st.expander(" Verified Sources"):
                            for doc in sources:
                                score = doc.get("score", 0)
                                source_name = doc.get("source", "Unknown")
                                preview = doc.get("content_preview", "")

                                st.markdown(f"**{source_name}**")
                                st.caption(f"Score: {score:.2f}")
                                st.text(preview)
                else:
                    st.error(f"Server Error: {response.status_code}")
                    full_response = "I encountered an error processing your request."

            except Exception:
                st.error("Connection drop.")
                full_response = "Network Error."

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
