# app/embedder.py
import os
import faiss
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

CHUNK_SIZE = 500  # characters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index/index.faiss"
METADATA_FILE = "faiss_index/metadata.pkl"

model = SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text: str) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks, buffer = [], ""
    for para in paragraphs:
        if len(buffer) + len(para) < CHUNK_SIZE:
            buffer += para + "\n\n"
        else:
            chunks.append(buffer.strip())
            buffer = para + "\n\n"
    if buffer:
        chunks.append(buffer.strip())
    return chunks

def embed_chunks(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    vectors = model.encode(chunks, convert_to_numpy=True)
    vectors = normalize(vectors)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save index + metadata
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(chunks, f)

    return index, chunks

def load_index() -> Tuple[faiss.IndexFlatL2, List[str]]:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query: str, top_k: int = 5) -> List[str]:
    index, chunks = load_index()
    q_vector = model.encode([query], convert_to_numpy=True)
    q_vector = normalize(q_vector)

    D, I = index.search(q_vector, top_k)
    return [chunks[i] for i in I[0]]