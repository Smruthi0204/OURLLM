import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

# Lightweight models for free tier
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
QA_MODEL = "google/flan-t5-small"  # ~300MB

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    response_time_ms: float

def validate_token(authorization: str = Header(...)):
    expected_token = "Bearer 1fb4d5e907d39a522b475e6d3e60c57a14fbade2cbcbe1c476ece8091ba1c0bd"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

@app.post("/hackrx/run")
async def run_query(
    request: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    # Validate token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing")
    validate_token(authorization)
    
    start_time = time.time()
    try:
        # Load and process PDF
        loader = PyPDFLoader(request.documents)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Smaller chunks
        texts = text_splitter.split_documents(pages)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(texts, embeddings)
        
        # Load lightweight QA model
        qa_model = pipeline(
            "text2text-generation",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device="cpu"
        )
        
        # Generate answers
        answers = []
        for q in request.questions:
            docs = db.similarity_search(q, k=2)  # Fewer docs to reduce memory
            context = " ".join([d.page_content for d in docs][:500])  # Limit context
            answer = qa_model(f"question: {q} context: {context}")[0]["generated_text"]
            answers.append(answer)
            
        response_time = (time.time() - start_time) * 1000
        return QueryResponse(answers=answers, response_time_ms=round(response_time, 2))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
