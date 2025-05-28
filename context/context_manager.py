from agents.ingestion_agent import IngestionAgent
from agents.rag_agent import LLMAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vector_embedding_agent import VectorEmbeddingAgent
from common.exception import AppException
from typing import List,Dict
from qdrant_client import QdrantClient
from common.logging import logger

class ContextManager:
    def __init__(self, qdrant_client: QdrantClient):
        self.ingestor = IngestionAgent()
        self.embedder = VectorEmbeddingAgent()
        self.retriever = RetrievalAgent()
        self.llm_agent = LLMAgent()
        self.indexed = False
        self.qdrant = qdrant_client
        self.collection_name ="pdf_chunks"

        try:
            count_resp = self.qdrant.count(collection_name=self.collection_name)
            self.indexed = count_resp.count > 0
            logger.info(
                "ContextManager initialized: %d vectors found in '%s'",
                count_resp.count,
                self.collection_name
            )
        except Exception as e:
            logger.exception("Failed to initialize ContextManager from Qdrant")
            raise AppException(
                message="Error accessing vector store.",
                status_code=500,
                error_detail=e
            )


    def ingest(self, file_paths: List[str]) -> Dict:
        """
        Ingest multiple PDFs, returning a list of page-level documents.
        """
        try:
            #1. Extract pages
            doc_res = self.ingestor.run(file_paths)
            pages = doc_res["documents"]
            #2. chunk and embed
            embed_res= self.embedder.run(pages)
            print(len(pages))
            print(embed_res['points_inserted'])
            self.indexed= True
            logger.info("Ingested %d documents into '%s'", len(pages), self.collection_name)
            return {
                "status": "Ingested",
                "pages": len(pages),
                "chunks": embed_res['points_inserted']
            }
             
        except Exception as e:
            # Catch any unexpected error
            raise AppException("Unexpected ingestion error",e)
        


    def query(self, question: str, top_k: int = None) -> Dict:
        if not self.indexed:
            try:
                count_resp = self.qdrant.count(collection_name=self.collection_name)
                if count_resp.count == 0:
                    raise AppException(
                        message="No documents available to query. Please ingest at least one document first.",
                        status_code=400
                    )
                # switch flag if embeddings exist
                self.indexed = True
                logger.info(
                    "Detected %d existing vectors in '%s', enabling queries.",
                    count_resp.count,
                    self.collection_name
                )
            except AppException:
                raise
            except Exception as e:
                logger.exception("Error checking vector store before query")
                raise AppException(
                    message="Error accessing vector store.",
                    status_code=500,
                    error_detail=e
                )

        # 1. Retrieve top-K contexts
        hits = self.retriever.run(question, top_k)["results"]
        # 2. Generate answer via LLM
        answer = self.llm_agent.run(question, hits)["answer"]
        return {"answer": answer, "contexts": hits}

if __name__ == "__main__":
    import os
    manager= ContextManager()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_dir=os.path.join(BASE_DIR,"data")
    print(doc_dir)

    path = [ 
        os.path.join(doc_dir, "sample.pdf"),
        os.path.join(doc_dir, "generalist.pdf"),
            
    ]
    manager.ingest(path)
    answer=manager.query(question="What is self attention in transformer ?", top_k=3)
    print(answer)