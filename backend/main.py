from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from backend.schemas import IngestResponse, QueryResponse
from context.context_manager import ContextManager
import os, shutil
from common.exception import AppException
from fastapi.middleware.cors import CORSMiddleware
from common.logging import logger
from qdrant_client import QdrantClient
from contextlib import asynccontextmanager


qdrant = QdrantClient(url="localhost:6334", prefer_grpc=True)
manager = ContextManager(qdrant_client=qdrant)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.context_manager = manager
    logger.info("Application startup complete; ContextManager ready.")
    yield
    #cleanup on shutdown
    logger.info("Application shutdown: cleanup complete.")

app = FastAPI(title="Multi-Agentic RAG",lifespan=lifespan)

@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    logger.info("Handling AppException: %s (status code: %s)", exc.message, exc.status_code)
    code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    return JSONResponse(
        status_code=code,
        content={"detail": exc.message}
    )

#catch-all 500 handler for anything else
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest one or more PDF files"
)
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Save uploaded PDFs to disk, then invoke the ContextManager.ingest() pipeline.
    Returns total pages and chunks indexed.
    """
    upload_dir = "data/temp_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    saved_paths = []

    # 1. Save files locally
    for file in files:
        dest_path = os.path.join(upload_dir, file.filename)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
        saved_paths.append(dest_path)

    # 2. Ingest via ContextManager
    try:
        result = manager.ingest(saved_paths)
        return IngestResponse(**result)
    except AppException as ae:
        # Handled by @app.exception_handler
        raise ae
    except Exception as e:
        # Unexpected
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected ingestion error."
        )

@app.get(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against ingested documents"
)
async def query(
    q: str = Query(..., description="Natural language question about your PDFs"),
    top_k: int = Query(None, alias="top_k", ge=1, le=10, description="How many contexts to retrieve (max 10)")
):
    """
    Given a user question, retrieve top-K chunks and generate an answer via LLM.
    """
    try:
        result = manager.query(q, top_k)
        return QueryResponse(**result)
    except AppException as ae:
        logger.warning("AppException in /query: %s", ae.message, exc_info=ae.error_detail)
        raise ae
    except Exception as e:
        logger.exception("Unexpected error in /query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected query error."
        )