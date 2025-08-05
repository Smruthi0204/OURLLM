# app/main.py
from fastapi import FastAPI
from .router import router

app = FastAPI(
    title="HackRX Query-Retrieval System",
    description="LLM-powered clause retrieval and explanation API",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")