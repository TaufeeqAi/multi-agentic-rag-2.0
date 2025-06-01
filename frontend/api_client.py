import requests
from config import UPLOAD_URL, QUERY_URL, STATUS_URL
from pathlib import Path

class APIClient:
    @staticmethod
    def upload_pdfs(file_paths: list[Path]) -> dict:
        files = [("files", (p.name, open(p, 'rb'), "application/pdf")) for p in file_paths]
        resp = requests.post(UPLOAD_URL, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def query(q: str, top_k: int) -> dict:
        params = {"q": q, "top_k": top_k}
        resp = requests.get(QUERY_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    

    @staticmethod
    def has_vectors() -> bool:
        """
        Check if the vector store has any embeddings.
        """
        resp = requests.get(STATUS_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("count", 0) > 0