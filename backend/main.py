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
from qdrant_client.http.exceptions import ResponseHandlingException


qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6334")

qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=True)
manager = ContextManager(qdrant_client=qdrant)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.context_manager = manager
    logger.info("Application startup complete; ContextManager ready.")
    global vector_count
    try:
        # Attempt to fetch collection info (replace 'pdf_chunks' with your actual collection name)
        collection_info = qdrant.get_collection("pdf_chunks")

        # CollectionInfo.points_count is the number of points currently stored
        vector_count = collection_info.points_count
        logger.info(f"Startup: loaded initial vector_count = {vector_count}")

    except Exception as e:
        # If Qdrant is not reachable (or the collection doesn’t exist yet), default to 0
        vector_count = 0
        logger.warning(f"Startup: failed to load vector_count from Qdrant ({e}), defaulting to 0.")

    yield
    #cleanup on shutdown
    logger.info("Application shutdown: cleanup complete.")

app = FastAPI(title="Multi-Agentic RAG",lifespan=lifespan)

@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    logger.info("Handling AppException: %s (status code: %s)", exc.message, exc.status_code)
    code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    return JSONResponse(
        status_code=exc.status_code,
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
        updated_info = qdrant.get_collection("pdf_chunks")
        global vector_count
        vector_count = updated_info.points_count
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
    

@app.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Return how many vectors are currently indexed"
)
async def get_status():
    try:
        collection_info = qdrant.get_collection("pdf_chunks")
        count = collection_info.points_count
        return {"count": count}
    except ResponseHandlingException as e:
        # Qdrant raised NOT_FOUND because the collection doesn't exist → treat as zero
        if "Collection `pdf_chunks` doesn't exist" in str(e):
            return {"count": 0}
        # Other Qdrant errors should still surface as HTTP 500
        logger.error("Unexpected Qdrant error in /status: %s", e)
        raise AppException("Error fetching Qdrant status", status_code=500)
    except Exception as e:
        text = str(e)
        if "Collection `pdf_chunks` doesn't exist" in text:
            return {"count": 0}
        logger.error("Error fetching Qdrant status: %s", e)
        raise AppException("Error fetching Qdrant status", status_code=500)