from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import time

app = FastAPI()

# Authentication
API_TOKEN = "1fb4d5e907d39a522b475e6d3e60c57a14fbade2cbcbe1c476ece8091ba1c0bd"

async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest):
    start_time = time.time()
    try:
        # Load and process PDF
        loader = PyPDFLoader(request.documents)
        pages = loader.load()
        
        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)
        
        # Initialize QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device="cpu"
        )
        
        # Answer questions
        answers = []
        for question in request.questions:
            docs = db.similarity_search(question, k=3)
            context = " ".join([d.page_content for d in docs])
            result = qa_pipeline(question=question, context=context)
            answers.append(result['answer'])
            
        return QueryResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ready"}
