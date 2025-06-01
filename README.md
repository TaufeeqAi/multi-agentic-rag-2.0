## PDF RAG Q\&A: Multi-Agent Retrieval‐Augmented Generation System

This repository implements a multi-agent, PDF‐based Retrieval‐Augmented Generation (RAG) system. Users can upload PDF documents, index their contents into a vector database (Qdrant), and interactively ask questions that are answered by an LLM (via Groq) using retrieved document chunks as context. The system consists of:

* **FastAPI Backend** – Handles PDF ingestion, vector embedding (via SentenceTransformers → Qdrant), retrieval, and LLM inference.
* **Streamlit Frontend** – Provides a simple UI for uploading PDFs, triggering ingestion, and chatting with the RAG assistant.
* **Dockerfiles** – Containerize both backend (API) and frontend (Streamlit) for easy deployment.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)

   * [1. Clone Repository](#1-clone-repository)
   * [2. Set Up Qdrant](#2-set-up-qdrant)
   * [3. Backend (FastAPI) Setup](#3-backend-fastapi-setup)
   * [4. Frontend (Streamlit) Setup](#4-frontend-streamlit-setup)
5. [Environment Variables & Configuration](#environment-variables--configuration)
6. [Running Locally (Without Docker)](#running-locally-without-docker)
7. [Running with Docker](#running-with-docker)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

---

## Features

* **PDF Ingestion**

  * Extract page‐level text from one or more uploaded PDF files.
  * Automatically chunk long pages into smaller segments for better embeddings.

* **Embedding & Vector Database**

  * Compute embeddings for each text chunk using a SentenceTransformer model.
  * Store embeddings and associated metadata (source file, page number, chunk index) in a Qdrant collection.

* **Retrieval Agent**

  * Given a user query, encode it into the same embedding space and retrieve the top‐K most similar document chunks from Qdrant.

* **LLM Agent (RAG)**

  * Combine retrieved chunks as “contexts” with the user’s question to generate a coherent, contextually grounded answer via a Groq‐hosted Llama‐4 Instruct model.

* **RESTful API (FastAPI)**

  * **`POST /upload`**: Upload and ingest PDFs.
  * **`GET /query`**: Ask a question, get answer + context chunks.
  * **`GET /status`**: Check vector count in Qdrant.

* **Streamlit Frontend**

  * Upload PDFs, trigger ingestion, and hold a chat‐style conversation with the RAG assistant.
  * “Show Sources” expander reveals the retrieved chunks (source filename, page number, snippet, and relevance score).

* **Modular, Agent‐based Pipeline**

  * **IngestionAgent**
  * **VectorEmbeddingAgent**
  * **RetrievalAgent**
  * **LLMAgent**
  * **ContextManager** orchestrates these steps seamlessly.

* **Docker Support**

  * Dockerfiles for both API (backend) and Streamlit (frontend) – build and deploy the entire system as containers.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             Streamlit Frontend                            │
│  ┌────────────────────┐   ┌────────────────────┐                          │
│  │ File Uploader      │   │ Chat Input/Output  │                          │
│  └─────────┬──────────┘   └─────────┬──────────┘                          │ 
│            │                         │                                    │
│            ▼                         ▼                                    │
│   ┌───────────────────┐      ┌───────────────────┐                        │
│   │ save_uploaded     │      │ APIClient.query() │                        │
│   │ files → TMP_DIR   │      └─────────┬─────────┘                        │
│   └─────────┬─────────┘                │                                  │
│             │                          │                                  │
│             ▼                          ▼                                  │
│   ┌──────────────────┐         ┌────────────────────┐                     │
│   │ POST /upload     │◀────────┤ GET /query         │                     │       
│   │ (Ingest PDFs)    │         │ (Ask question)     │                     │
│   └─────────┬────────┘         └─────────┬──────────┘                     │
│             │                            │                                │
│             ▼                            ▼                                │
│ ┌────────────────────┐         ┌───────────────────────────┐              │
│ │ ContextManager.ingest() │    │ ContextManager.query()                   │
│ │  - IngestionAgent.run() │    │  - RetrievalAgent.run()                  │
│ │  - VectorEmbeddingAgent │    │  - LLMAgent.run()                        │
│ └─────────┬──────────────┘    └────────┬──────────────────┘               │
│           │                            │                                  │
│           ▼                            ▼                                  │
│ ┌───────────────────┐        ┌─────────────────────────────┐              │
│ │ IngestionAgent   │        │ RetrievalAgent               │              │
│ │  (pdfplumber →   │        │  (query embedding → search)  │              │
│ │   docs per page) │        └─────────────────────────────┘               │
│ └───────────────────┘                                                     │
│           │                                                               │
│           ▼                                                               │
│ ┌────────────────────┐                                                    │
│ │ VectorEmbeddingAgent│                                                   │
│ │  (SentenceTransformer  → Qdrant) │                                      │
│ └────────────────────┘                                                    │
│                                                                           │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │                         Qdrant Vector DB                           │    │
│ │ Collection “pdf_chunks”: { vector, metadata: {source, page, chunk}}│    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │                   LLMAgent (Groq-hosted Llama‑4)                   │    │
│ │  - Build prompt: question + retrieved chunks as context            │   `│
│ │  - Return “answer” text                                            │   `│
│ └────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Python 3.9+**
2. **Qdrant** (v1.2.4 or later recommended)
3. **Docker & Docker Compose** (optional, but highly recommended for containerized deployment)
4. **Environment variables** for:

   * `QDRANT_HOST` (e.g., `localhost`)
   * `QDRANT_PORT` (default: `6334`)
   * `API_BASE` (used by Streamlit; default: `http://localhost:8000`)

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/TaufeeqAi/multi-agentic-rag-2.0.git
cd pdf-rag-qa
```

### 2. Set Up Qdrant

* **Option A: Local Dockerized Qdrant**

  ```bash
  docker run -d --name qdrant -p 6334:6334 qdrant/qdrant:v1.2.4
  ```

  By default, this exposes Qdrant’s HTTP API on `http://localhost:6334`.

* **Option B: Self‑hosted / managed Qdrant**
  Ensure you have a running Qdrant instance and note its host/port. Set environment variables accordingly.

### 3. Backend (FastAPI) Setup

1. **Navigate to backend folder**

   ```bash
   cd mas/backend
   ```

2. **Create & activate a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   # venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file (in `Project_root`) with:

   ```env
   QDRANT_HOST=localhost
   QDRANT_PORT=6334
   ```

5. **Run the API**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   * The API will be available at `http://localhost:8000`.
   * **Endpoints:**

     * `POST /upload` – ingest PDF(s)
     * `GET /query?q=<your_question>&top_k=<int>`
     * `GET /status`
     * `GET /health` (optional)

### 4. Frontend (Streamlit) Setup

1. **Navigate to frontend folder**

   ```bash
   cd frontend
   ```

2. **(Optionally) Reuse the same virtual environment**
   If you want a separate venv, create/activate it similarly:

   ```bash
   python3 -m venv venv-streamlit
   source venv-streamlit/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   In `frontend`, create a `.env` (or export directly) with:

   ```env
   # If your backend is running on a different host/port, update accordingly
   API_BASE=http://localhost:8000
   ```

5. **Run the Streamlit App**

   ```bash
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

   * Open `http://localhost:8501` in your browser.
   * Upload PDFs, click “Ingest Documents,” and start chatting.

---

## Environment Variables & Configuration

| Variable           | Description                                                               | Default                  |
| ------------------ | ------------------------------------------------------------------------- | ------------------------ |
| **QDRANT\_HOST**   | Hostname or IP of Qdrant instance                                         | `localhost`              |
| **QDRANT\_PORT**   | Port on which Qdrant listens                                              | `6334`                   |
| **API\_BASE**      | Base URL of FastAPI backend (used by Streamlit frontend)                  | `http://localhost:8000`  |
| **API\_KEY\_GROQ** | (Optional) API key or token for authenticating with Groq/Llama‑4 endpoint | N/A (must be configured) |

> **Note**: If your Groq LLM requires an API key or any other credentials, you can modify `agents/rag_agent.py` to read from environment variables or a config file. By default, it assumes unauthenticated access to a local Groq endpoint.

---

## Running Locally (Without Docker)

1. **Start Qdrant** (Docker or local binary).
2. **Launch Backend**

   ```bash
   cd backend
   source ../venv/bin/activate     # if not already activated
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. **Launch Frontend**

   ```bash
   cd frontend
   source ../venv-streamlit/bin/activate  # if using a separate venv
   streamlit run app.py --server.port=8501
   ```
4. **Browse** `http://localhost:8501`.

---

## Running with Docker

You can quickly spin up both backend and frontend using Docker. Adjust ports and environment variables as needed.

### 1. Build Backend Image

```bash
cd docker
docker build -f Dockerfile.api -t pdf-rag-api:latest .
```

### 2. Build Frontend Image

```bash
docker build -f Dockerfile.streamlit -t pdf-rag-streamlit:latest .
```

### 3. Run Qdrant, API, and Streamlit Containers

Optionally, create a `docker-compose.yml` in the project root:

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.2.4
    container_name: qdrant
    ports:
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

  api:
    image: pdf-rag-api:latest
    container_name: pdf-rag-api
    depends_on:
      - qdrant
    environment:
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6334
    ports:
      - "8000:8000"

  streamlit:
    image: pdf-rag-streamlit:latest
    container_name: pdf-rag-streamlit
    depends_on:
      - api
    environment:
      API_BASE: http://api:8000
    ports:
      - "8501:8501"

volumes:
  qdrant_storage:
```

Start everything with:

```bash
docker-compose up -d
```

* **Streamlit UI**: `http://localhost:8501`
* **FastAPI Swagger Docs**: `http://localhost:8000/docs`
* **Qdrant HTTP**: `http://localhost:6334`

---

## Usage

1. **Upload & Ingest PDFs**

   * In the Streamlit UI, click “Browse files” and select one or more PDF documents.
   * Click **Ingest Documents**.
   * The backend will:

     1. Extract text from each page (via `IngestionAgent`).
     2. Chunk large pages if needed and compute embeddings (via `VectorEmbeddingAgent`).
     3. Store all embeddings into Qdrant under collection `pdf_chunks`.
   * Streamlit will display:

     ```
     Ingested 50 pages, 120 chunks
     ```

2. **Ask a Question**

   * Once ingestion is complete, a text input appears.
   * Type your question (e.g., “What is self-attention in a Transformer?”) and click **Send**.
   * Internally, the backend:

     1. Encodes the query (via `RetrievalAgent`), retrieves top‐K chunks.
     2. Passes query + chunks to `LLMAgent` (Groq Llama‑4) to generate an answer.
   * The UI will show:

     ```
     You: What is self-attention in a Transformer?
     Bot: Self‐attention is…
     ```

3. **“Show Sources”**

   * Click the expander beneath the bot’s reply to reveal each retrieved chunk, including:

     * Source filename
     * Page number
     * Relevance score
     * A \~200‐character preview of that chunk

---

## Project Structure

```plaintext
project_root/
├── agents/
│   ├── ingestion_agent.py        # Extract PDF pages → text docs
│   ├── vector_embedding_agent.py # Chunk text & upsert embeddings into Qdrant
│   ├── retrieval_agent.py        # Query embedding & top‐K vector search
│   ├── rag_agent.py              # Compose prompt & call Groq LLM for final answer
│   └── __init__.py
│
├── common/
│   ├── exception.py              # Defines AppException (custom error)
│   ├── logging.py                # Configures rotating logger + console output
│   └── __init__.py
│
├── context/
│   ├── context_manager.py        # Orchestrates ingestion → embedding → retrieval → LLM
│   └── __init__.py
│
├── backend/
│   ├── main.py                   # FastAPI app: /upload, /query, /status endpoints
│   ├── schemas.py                # Pydantic models for request/response payloads
│   ├── requirements.txt          # Backend dependencies
│   └── __init__.py
│
├── frontend/
│   ├── config.py                 # Frontend settings (API_BASE, URLs, TMP_DIR, DEFAULT_TOP_K)
│   ├── api_client.py             # Simple wrappers for POST /upload, GET /query, GET /status
│   ├── helpers.py                # Save & clear temporary PDFs
│   ├── ui.py                     # Streamlit components: render_ingest() & render_chat()
│   ├── app.py                    # Top‐level Streamlit script
│   ├── requirements.txt          # Frontend dependencies
│   └── __init__.py
│
└── docker/
    ├── Dockerfile.api            # Build image for FastAPI backend
    └── Dockerfile.streamlit      # Build image for Streamlit frontend
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** this repository.
2. **Create a new branch** for your feature or bug fix:

   ```bash
   git checkout -b feature/your‐feature‐name
   ```
3. **Make changes** and ensure all existing tests (if any) pass.
4. **Commit & Push** your changes to your fork:

   ```bash
   git commit -m "Add <feature/bugfix description>"
   git push origin feature/your‐feature‐name
   ```
5. **Open a Pull Request** against the `main` branch of this repo.
6. We will review and provide feedback; once approved, your PR will be merged.

Please adhere to the existing code style (PEP8 for Python) and add docstrings for any new modules/classes/functions.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute as permitted.

---

## Acknowledgements

* **Qgpt** – Qdrant vector database powering retrieval.
* **SentenceTransformers** – Embeddings for document chunks.
* **Groq Llama‑4 Instruct** – Hosted LLM for generating answers.
* **FastAPI** – High‐performance Python web framework for building APIs.
* **Streamlit** – Rapid MVP for frontend UI.
* **Agno** – Agent orchestration framework (used for consistency in agent design).
* **pdfplumber** – PDF text extraction library.
* **LangChain** – Utility classes (e.g., `RecursiveCharacterTextSplitter`) to chunk long texts.

---

> **Note**: If you find any issues or have ideas for improvement, please open an issue or submit a pull request. We hope this RAG system proves helpful for your PDF Q\&A needs!
