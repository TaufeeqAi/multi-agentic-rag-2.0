from agno.agent import Agent
import pdfplumber
import os
from typing import List, Dict
import uuid
from common.exception import AppException
from common.logging import logger

class IngestionAgent(Agent):

    def __init__(self):
        super().__init__(
            name="PDF Ingestion Agent",
            role="Given a list of PDF file paths, extract text from each page "
                 "and record page-number and source filename.",
            instructions=[
                "For each valid PDF path, open with pdfplumber.",
                "Iterate pages in order, extract text, and append a dict:",
                "  {text: str, page_number: int, source: str}.",
                "Skip pages with no text.",
                "On any file-level exception, wrap in AppException."
            ]
        )

    def run(self, file_paths: List[str]) -> Dict:
        # 1. Check existence & readability
        
        pages_data = []
        for pdf_path in file_paths:
            filename = os.path.basename(pdf_path)
            print(filename)
            if not os.path.exists(pdf_path):
                msg = f"File not found: {pdf_path}"
                raise AppException(msg)
            if not pdf_path.lower().endswith(".pdf"):
                raise AppException(f"IngestionAgent.run: Invalid file extension, expected .pdf: '{pdf_path}'")
            
            doc_id=str(uuid.uuid4())


            try:
                # 2. Open the PDF with pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    # 3. Iterate over each page (pdfplumber pages are 0-indexed under the hood)
                    for i, page in enumerate(pdf.pages, start=1):
                        # Extract text for this page (returns None if blank)
                        txt = page.extract_text() or ""
                        if not txt.strip():
                            continue
                        pages_data.append({
                            "page": i, 
                            "text": txt,
                            "source":filename,
                            "doc_id":doc_id
                        })

                
            except Exception as e:
                logger.error("failed to parse the file")
                raise AppException(f"IngestionAgent.run: Error parsing '{pdf_path}'", error_detail=e)
        
        return {"documents": pages_data}


if __name__ == "__main__":
    # CLI interface: python agents/pdf/pdf_upload_agent.py path/to/file.pdf

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_dir=os.path.join(BASE_DIR,"data")
    print(doc_dir)

    path = [ 
        os.path.join(doc_dir, "sample.pdf"),
        os.path.join(doc_dir, "generalist.pdf"),
            
    ]
    
    agent = IngestionAgent()
    try:
        resp = agent.run(path)
        docs= resp['documents']
        print(f"Extracted {len(docs)} pages from '{path}'\n")
        for doc_info in docs:
            doc_id = doc_info["doc_id"]
            doc_name = doc_info["source"]
            page_num = doc_info["page"]
            page_txt = doc_info["text"].strip().replace("\n", " ")[:200]  # first 200 chars
            print(f" doc_name: {doc_name}, doc_id: {doc_id}, Page {page_num}: {page_txt!r}")
            print("=" * 80)
    except AppException as ae:
        print(f"Error: {ae}")
        if ae.error_detail:
            print(f"Detail: {ae.error_detail}")
    
