import shutil
from pathlib import Path
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config import TMP_DIR

TMP_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_files(uploaded_files: list[UploadedFile]) -> list[Path]:
    paths = []
    for uf in uploaded_files:
        dest = TMP_DIR / uf.name
        with open(dest, 'wb') as f:
            f.write(uf.getbuffer())
        paths.append(dest)
    return paths

def clear_temp_dir():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir()