from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

# Use tiny model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
EMBED_CACHE = {}

def embed_chunks(chunks: List[str]):
    """Batch process chunks with caching"""
    global EMBED_CACHE
    new_chunks = [c for c in chunks if c not in EMBED_CACHE]
    
    if new_chunks:
        embeddings = model.encode(new_chunks, show_progress_bar=False)
        EMBED_CACHE.update(zip(new_chunks, embeddings))
    
    return np.array([EMBED_CACHE[c] for c in chunks])

def search(query: str, top_k: int = 3) -> List[str]:
    """Simplified semantic search"""
    query_embed = model.encode([query])[0]
    similarities = [
        (text, np.dot(query_embed, embed))
        for text, embed in EMBED_CACHE.items()
    ]
    return [text for text, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]]
