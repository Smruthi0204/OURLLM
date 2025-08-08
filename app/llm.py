# app/llm.py
import os
import requests
from typing import List

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "mixtral-8x7b-32768"  # Fastest open-weight model

def generate_answer(question: str, context_chunks: List[str]) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    context = " ".join(context_chunks)[:2000]
    
    payload = {
        "messages": [
            {"role": "system", "content": "Answer using the provided context"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ],
        "model": MODEL
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "Answer unavailable"
