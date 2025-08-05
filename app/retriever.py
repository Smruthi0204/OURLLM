# app/retriever.py
from typing import List, Dict
from .embedder import search

def retrieve_clauses(questions: List[str], top_k: int = 5) -> Dict[str, List[str]]:
    """
    Given a list of questions, return top-K context matches for each.
    """
    result = {}
    for q in questions:
        top_chunks = search(q, top_k=top_k)
        result[q] = top_chunks
    return result