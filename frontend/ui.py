import streamlit as st
from config import DEFAULT_TOP_K
from api_client import APIClient
from helpers import save_uploaded_files

# Ingestion UI
def render_ingest():
    st.header("📥 Upload & Ingest PDFs")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Ingest Documents"):
        with st.spinner("Uploading and ingesting..."):
            try:
                paths = save_uploaded_files(uploaded)
                data = APIClient.upload_pdfs(paths)
                st.success(f"Ingested {data['pages']} pages, {data['chunks']} chunks.")
                st.session_state.ingested = True
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# Chat UI
def render_chat():
    try:
        available = APIClient.has_vectors()
    except Exception as e:
        st.error(f"Could not check vector store status: {e}")
        return

    if not available:
        st.warning("Please ingest PDFs to start chatting.")
        return
    if "history" not in st.session_state:
        st.session_state.history = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Your question: ")
        top_k    = st.slider(
            "Number of contexts to retrieve:",
            min_value=1, 
            max_value=10,         
            key="top_k"
        )
        submitted = st.form_submit_button("Send")

        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = APIClient.query(question, top_k)
                        st.session_state.history.append({
                            "user": question,
                            "answer": result["answer"],
                            "contexts": result["contexts"]
                        })
                       
                    except Exception as e:
                        st.error(f"Query failed: {e}")

    for entry in st.session_state.history:
        st.markdown(f"**You: {entry['user']}")
        st.markdown(f"**Bot: {entry['answer']}")
        with st.expander("Show Sources"):
            for ctx in entry["contexts"]:
                txt = ctx["text"].replace("\n", " ")
                st.markdown(
                    f"- *{ctx['source']}* : page {ctx['page_number']}  \
"
                    f"  {txt[:200]}{'…' if len(txt) > 200 else ''}  \
"
                    f"  *(score: {ctx['score']:.3f})*"
                )