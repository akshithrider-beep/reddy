from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from utils.schema import Question
from services.question_svc import load_all_questions
import os

# Load a lightweight local embedding model
# all-MiniLM-L6-v2 is fast and good for semantic search
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to load sentence-transformers model. RAG may not work: {e}")
    model = None

# In-memory FAISS index
dimension = 384 # Output dimension of all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
question_registry = [] # To map index back to question objects

def build_index():
    global index, question_registry
    if model is None:
        return

    questions = load_all_questions()
    if not questions:
        print("No questions to index.")
        return
        
    texts = [f"{q.subject} {q.topic} {q.question}" for q in questions]
    
    print(f"Encoding {len(texts)} questions for FAISS index...")
    embeddings = model.encode(texts)
    
    # FAISS expects float32
    embeddings = np.array(embeddings).astype('float32')
    
    # Reset index and rebuild
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    question_registry = questions
    print("FAISS index built successfully.")

def search_similar_questions(query: str, k: int = 5) -> List[Question]:
    if model is None or index.ntotal == 0:
        # Fallback to simple text search if model failed or index empty
        from services.question_svc import search_questions as fallback_search
        return fallback_search(query)[:k]
        
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(question_registry) and idx >= 0:
            results.append(question_registry[idx])
            
    return results

# Build index on module import (runs once when server starts)
# In production, you'd save/load the .faiss file instead of rebuilding on every boot.
build_index()
