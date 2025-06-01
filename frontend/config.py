import os
from pathlib import Path

# API endpoints
API_BASE   = os.getenv("API_BASE", "http://localhost:8000")
UPLOAD_URL = f"{API_BASE}/upload"
QUERY_URL  = f"{API_BASE}/query"
STATUS_URL   = f"{API_BASE}/status" 
# Local storage
TMP_DIR = Path(os.getenv("TMP_DIR", "./data/temp_pdfs"))

# Defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))