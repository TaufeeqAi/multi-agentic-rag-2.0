import logging
import uuid
from typing import List, Dict
from agno.agent import Agent
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from common.exception import AppException
from common.logging import logger


class VectorEmbeddingAgent(Agent):

    def __init__(self, collection_name: str = "pdf_chunks"):
        self.collection_name = collection_name

        super().__init__(
            name="Vector Embedding Agent",
            role="Chunk text and embed with SentenceTransformer, then store in Qdrant.",
            instructions=[
                "Use RecursiveCharacterTextSplitter to split each page.text into overlapping chunks.",
                "Embed each chunk via SentenceTransformer.",
                "Prepare Qdrant PointStructs with vector and payload {text, page_number, source, chunk_id}.",
                "Recreate the Qdrant collection to ensure idempotency."
            ]
        )


        # 1. Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
        except Exception as e:
            raise AppException("VectorEmbeddingAgent: Failed to connect to Qdrant", error_detail=e)

        # 2. Initialize embedding model (384-dimensional vectors)
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise AppException("VectorEmbeddingAgent: Failed to load SentenceTransformer model", error_detail=e)

        # 3. Ensure the collection exists (create if not)
        try:
            exists = self.qdrant_client.collection_exists(self.collection_name)

            if not exists:
                # Create with 384 dimensions and cosine distance
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance="Cosine"),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' with 384 dim, Cosine.")
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            raise AppException(
                f"VectorEmbeddingAgent: Error ensuring Qdrant collection '{collection_name}'", error_detail=e
            )

    def run(self, pages_data: List[Dict], batch_size: int = 64) -> Dict:
        if not isinstance(pages_data, list) or len(pages_data) == 0:
            raise AppException("VectorEmbeddingAgent.run: pages_data must be a non-empty list")

        all_points = []
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        except Exception as e:
            raise AppException("VectorEmbeddingAgent: Failed to initialize text splitter", error_detail=e)

        total_chunks = 0

        # 4. Iterate over each page's text
        for page_dict in pages_data:
            page_num = page_dict.get("page")
            page_text = page_dict.get("text", "")
            doc_name = page_dict.get("source")
            doc_id = page_dict.get("doc_id")

            # Skip pages with empty text
            if not page_text.strip():
                logger.debug(f"Page {page_num} is blank. Skipping.")
                continue

            # 5. Split page_text into documents/chunks
            try:
                docs = text_splitter.create_documents([page_text])
            except Exception as e:
                raise AppException(f"VectorEmbeddingAgent: Text splitting failed on page {page_num}", error_detail=e)

            # 6. For each chunk, generate an embedding and prepare a PointStruct
            for idx, doc in enumerate(docs):
                chunk_text = doc.page_content
                try:
                    vector = self.embedding_model.encode(chunk_text).tolist()
                except Exception as e:
                    raise AppException("VectorEmbeddingAgent: Embedding computation failed", error_detail=e)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk_text,
                        "page_number": page_num,
                        "source": doc_name,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_name}_p{page_num}_c{idx}",
                        "chunk_index": idx
                    },
                )
                all_points.append(point)
                total_chunks += 1

        # 7. Upsert points in batches into Qdrant
        inserted = 0
        try:
            for i in range(0, len(all_points), batch_size):
                batch = all_points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                inserted += len(batch)
                logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} points).")
        except Exception as e:
            raise AppException("VectorEmbeddingAgent: Qdrant upsert failed", error_detail=e)

        logger.info(f"VectorEmbeddingAgent: Total chunks inserted: {inserted}")
        return {"status": "success", "points_inserted": inserted}


if __name__ == "__main__":
    # Optional CLI for manual test:
    # Usage: python vector_embedding_agent.py path/to/sample.pdf

    import sys
    import os
    from agents.ingestion_agent import IngestionAgent
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    doc_dir=os.path.join(BASE_DIR,"data")

    path = [ 
        os.path.join(doc_dir, "sample.pdf"),
        os.path.join(doc_dir, "generalist.pdf"),
            
    ]
    
    pdf_agent = IngestionAgent()
    try:
        resp = pdf_agent.run(path)
        docs= resp['documents']
    except AppException as e:
        print(f"Error extracting PDF pages: {e}")
        sys.exit(1)

    embed_agent = VectorEmbeddingAgent()
    try:
        res = embed_agent.run(docs, batch_size=32)
        result = res['points_inserted']
        print(f"Upsert result: {result}")
    except AppException as e:
        print(f"Error during embedding/upsert: {e}")
        if e.error_detail:
            print(f"Detail: {e.error_detail}")
        sys.exit(1)
