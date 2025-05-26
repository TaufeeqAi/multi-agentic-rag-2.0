import os

project_structure = {
    "agents": {
        "agno": ["__init__.py", "agno_kb.py"],
        "pdf": ["pdf_upload_agent.py", "vector_embedding_agent.py", "retrieval_agent.py"],
        "qa": ["summarization_agent.py", "rag_agent.py"],
        "__init__.py": None,
    },
    "api": {
        "__init__.py": None,
        "main.py": None,
        "crud.py": None,
        "database.py": None,
        "models.py": None,
        "schemas.py": None,
        "routers": ["qa_router.py", "pdf_router.py", "chat_router.py"],
    },
    "context": {
        "__init__.py": None,
        "context_manager.py": None,
    },
    "common": {
        "__init__.py": None,
        "exception.py": None,
        "logging.py": None,
    },
    "frontend": {
        "__init__.py": None,
        "main.py": None,
    },
    "docker": {
        "Dockerfile.api": None,
        "Dockerfile.streamlit": None,
        "docker-compose.yml": None,
    },
    "alembic": {
        "env.py": None,
        "versions": {},  # Can be empty for now
    },
    "requirements.txt": None,
    ".env": None,
    "README.md": None,
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                file_path = os.path.join(path, file)
                with open(file_path, "w") as f:
                    f.write(f"# {file}\n")
        elif content is None:
            if "." in name:  # File
                with open(path, "w") as f:
                    if name.endswith(".py"):
                        f.write(f"# {name}\n")
                    elif name == "README.md":
                        f.write("# Project Title\n\nMulti Agentic RAG.\n")
                    elif name == ".env":
                        f.write("# Environment variables\nQDRANT_API_KEY=\nDB_URL=\nGROQ_API_KEY=\n")
                    elif name == "requirements.txt":
                        f.write("fastapi\nuvicorn\nsqlalchemy\npydantic\nstreamlit\ngenai-sdk\npdfplumber\nqdrant-client\n")
            else:  # Directory
                os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    base_dir = os.path.abspath("./")
    os.makedirs(base_dir, exist_ok=True)
    create_structure(base_dir, project_structure)
    print(f"✅ Project structure created at: {base_dir}")
