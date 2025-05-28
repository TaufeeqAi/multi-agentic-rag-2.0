from typing import List, Dict
from agno.agent import Agent
from agno.models.groq import Groq 
from agents.retrieval_agent import RetrievalAgent
from common.exception import AppException
import os
from dotenv import load_dotenv
from common.logging import logger

class LLMAgent(Agent):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,
        
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base_dir, ".env")
        print(base_dir)
        load_dotenv(env_path) 
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")

        print(groq_api_key)

        super().__init__(
            name="LLM Answer Agent",
            model= Groq(id=model_name, temperature=temperature,api_key=groq_api_key),
            role="Generate a concise answer to the user query based on provided contexts.",
            markdown=True,
            instructions=[
                "Format each context with source and page number annotations.",
                "Compose a system+user prompt that includes contexts and the question.",
                "If contexts don’t cover the question, respond “I don't know.”",
                "Return only the answer text."
            ]
        )

    def run(self, query: str, contexts: List[Dict]) -> Dict:
        if not query.strip():
            raise AppException("Query must be a non-empty string.")
        
        if not contexts:
            raise AppException("No context provided for LLM generation", status_code=400)

        context_blocks = []
        for c in contexts:
            source = c.get('source', 'Unknown Source')
            page_number = c.get('page_number', 'Unknown Page')
            text = c.get('text', '')
            header = f"[Source: {source} | Page: {page_number}]"
            context_blocks.append(f"{header}\n{text}")

        prompt = (
            "You are a helpful assistant. Use the following extracted passages to answer the user's question."
            "If the answer is not contained within the passages, reply with “I don't know.”\n\n"
            "Passages:\n"
            + "\n\n---\n\n".join(context_blocks)
            + f"\n\nQuestion: {query}\nAnswer:"
        )

        # 2. Call LLM
        try:
            response = super().run(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error("Failed to generate a response", e)
            raise AppException("Failed to generate answer", e)

        return {"answer": answer}


if __name__ == "__main__":
    query_text = "What are neural turing machines"

    retriever_agent=RetrievalAgent()
    contexts=retriever_agent.run(query_text, top_k=5)['results']
    print(f"Retrieved context: {contexts}")

    agent = LLMAgent()
    try:
        result = agent.run(query_text, contexts)
        print("Answer:", result["answer"])
    except AppException as e:
        print(f"Error: {e}")
        if getattr(e, 'error_detail', None):
            print(f"Detail: {e.error_detail}")
