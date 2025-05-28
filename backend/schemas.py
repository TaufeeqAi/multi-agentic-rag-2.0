from pydantic import BaseModel, Field
from typing import List

class IngestResponse(BaseModel):
    status: str = Field(..., example="ingested")
    pages: int = Field(..., example=12)
    chunks: int = Field(..., example=48)

class SourceContext(BaseModel):
    text: str
    score: float
    page_number: int
    source: str
    chunk_id: str
    chunk_index: int
    doc_id: str

class QueryResponse(BaseModel):
    answer: str
    contexts: List[SourceContext]
