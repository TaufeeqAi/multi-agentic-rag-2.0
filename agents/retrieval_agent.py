import logging
from typing import List, Dict
from agno.agent import Agent
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from common.exception import AppException
from common.logging import logger

class RetrievalAgent(Agent):
    def __init__(self, collection_name: str = "pdf_chunks"):
        self.collection_name = collection_name

        super().__init__(
            name="Semantic Retrieval Agent",
            role="Encode user query and fetch top-K similar text chunks from Qdrant.",
            instructions=[
                "Encode the free‑form query via the same SentenceTransformer model used for embedding.",
                "Call QdrantClient.search with the query vector and limit=top_k.",
                "Extract `payload` and `score` from each hit, returning structured results."
            ]
        )

        # 1. Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url="localhost:6334", prefer_grpc=True)
        except Exception as e:
            raise AppException("RetrievalAgent: Unable to connect to Qdrant", error_detail=e)

        # 2. Initialize embedding model (same as VectorEmbeddingAgent)
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise AppException("RetrievalAgent: Unable to load SentenceTransformer model", error_detail=e)

    def run(self, query: str, top_k: int = 5) -> Dict:
        if not isinstance(query, str) or not query.strip():
            raise AppException("RetrievalAgent.run: Query must be a non-empty string")

        # 1. Compute the query embedding
        try:
            query_vector = self.embedding_model.encode(query).tolist()
        except Exception as e:
            raise AppException("RetrievalAgent: Embedding computation failed", error_detail=e)

        # 2. Perform search in Qdrant
        try:
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                
            )
        except Exception as e:
            raise AppException("RetrievalAgent: Qdrant search failed", error_detail=e)

        # 3. Parse hits into list of dicts
        results = []
        for hit in hits:
            payload = hit.payload or {}
            score= float(hit.score)
            results.append({
                "text": payload.get("text", ""),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "score": score,
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "source":payload.get("source"),
            })

        logger.info(f"RetrievalAgent: Retrieved {len(results)} results for query '{query}'")
        return {"results":results}


if __name__ == "__main__":
  
    query_text = "What is generalist agent ?"
    agent = RetrievalAgent()
    try:
        resp = agent.run(query_text, top_k=5)
        hits = resp["results"]
        print(f"Top {len(hits)} results for query: '{query_text}'\n")
        for idx, r in enumerate(hits, start=1):
            print(
                f"{idx}. doc_name: {r['source']}, Page: {r['page_number']} "
                f"(chunk_index: {r['chunk_index']}, score: {r['score']:.4f}), chunk_id: {r['chunk_id']}"
            )
            txt = r['text'].replace("\n", " ")
            print(f"   {txt[:200]}…")
            print("-" * 80)
    except AppException as e:
        print(f"Error: {e}")
        if getattr(e, 'error_detail', None):
            print(f"Detail: {e.error_detail}")
        