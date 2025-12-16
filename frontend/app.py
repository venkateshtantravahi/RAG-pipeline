import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/query"
st.set_page_config(page_title="RAG Pipeline", page_icon="🤖")

st.title("RAG Pipeline")
st.caption("Ask questions about your Technical Knowledge Base (PDFs).")

with st.sidebar:
    st.header("Settings")
    k_value = st.slider("Retrival Depth (k)", min_value=1, max_value=10, value=3)
    use_hybrid = st.toggle("Use Hybrid Search", value=True)

    if st.button("Check Backend Health"):
        try:
            res = requests.get("http://localhost:8000/health")
            if res.status_code == 200:
                st.success("Backend is Online")
            else:
                st.error("Backend returned error.")
        except Exception:
            st.error("Cannot connect to Backend. Is it running?")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- HANDLING USER INPUT -----
if prompt := st.chat_input("Ask a question about LLMs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        payload = {
            "query": prompt,
            "k": k_value,
            "use_hybrid_search": use_hybrid,
        }

        try:
            with st.spinner("Thinking..."):
                response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                full_response = answer
                message_placeholder.markdown(full_response)

                if sources:
                    with st.expander("Sources & Citations"):
                        for doc in sources:
                            st.markdown(
                                f"**Source:** `{doc['source']}` (Score: {doc['score']:.2f})"
                            )
                            st.caption(doc["content_preview"])
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                full_response = "Sorry, I encountered an error."

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend. Please ensure is running and try again later."
            )
            full_response = "connection error."

        st.session_state.messages.append({"role": "assistant", "content": full_response})
