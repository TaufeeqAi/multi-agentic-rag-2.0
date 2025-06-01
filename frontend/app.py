# frontend/app.py
import streamlit as st
from helpers import clear_temp_dir
from ui import render_ingest, render_chat


def main():
    st.set_page_config(page_title="PDF Q&A Chat", layout="wide")
    st.title("🗂️ PDF RAG Q&A")
    st.write("Upload PDF files, ingest them, then ask questions in a chat interface.")

    # Reset temp folder on cold start
    if 'initialized' not in st.session_state:
        clear_temp_dir()
        st.session_state.initialized = True
        st.session_state.ingested = False
        st.session_state.history = []

    render_ingest()
    st.markdown("---")
    render_chat()

if __name__ == "__main__":
    main()