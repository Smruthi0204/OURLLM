from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

app = FastAPI()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "declare-lab/flan-alpaca-large"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    response_time_ms: float

@app.post("/hackrx/run")
async def run_query(request: QueryRequest):
    start_time = time.time()
    try:
        loader = PyPDFLoader(request.documents)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(texts, embeddings)
        qa_pipeline = pipeline("text2text-generation", model=QA_MODEL, device="cpu")
        answers = []
        for q in request.questions:
            docs = db.similarity_search(q, k=3)
            context = " ".join([d.page_content for d in docs])
            answer = qa_pipeline(f"question: {q} context: {context}")[0]["generated_text"]
            answers.append(answer)
        response_time = (time.time() - start_time) * 1000
        return QueryResponse(answers=answers, response_time_ms=round(response_time, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)