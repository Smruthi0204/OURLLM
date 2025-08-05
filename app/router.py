# app/router.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
from .parser import parse_document
from .embedder import chunk_text, embed_chunks
from .retriever import retrieve_clauses
from .llm import generate_answer

router = APIRouter()

class RunRequest(BaseModel):
    documents: str  # single URL for now
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=RunResponse)
async def run_submission(payload: RunRequest, request: Request):
    try:
        # Step 1: Parse document
        raw_text = parse_document(payload.documents)

        # Step 2: Chunk + Embed document
        chunks = chunk_text(raw_text)
        embed_chunks(chunks)  # Recreates index every time (could cache)

        # Step 3: Retrieve clauses per question
        matches = retrieve_clauses(payload.questions)

        # Step 4: Generate answer for each question
        answers = []
        for q in payload.questions:
            context = matches.get(q, [])
            answer = generate_answer(q, context)
            answers.append(answer)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))