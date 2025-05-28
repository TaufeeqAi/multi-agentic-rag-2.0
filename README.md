# Multi-Agent Q&A RAG System

This repository will house a multi-agent Retrieval-Augmented Generation (RAG) system for answering questions over uploaded PDFs. It includes:
- PDF ingestion and page-level text extraction
- Embeddings via SentenceTransformers → Qdrant vector database
- A Groq-hosted LLM forquestion answering (RAG pipeline)
- FastAPI backend with database persistence
- Streamlit frontend for user interaction
