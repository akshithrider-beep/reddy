from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from utils.schema import Question
from services.question_svc import load_all_questions

# Lazy load model
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
question_registry = []
INDEX_READY = False


def build_index():
    global index, question_registry, INDEX_READY

    if INDEX_READY:
        return

    model = get_model()

    questions = load_all_questions()

    if not questions:
        print("No questions to index.")
        INDEX_READY = True
        return

    texts = [
        f"{q.subject} {q.topic} {q.question}"
        for q in questions
    ]

    print(f"Encoding {len(texts)} questions...")

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    question_registry = questions

    INDEX_READY = True

    print("FAISS index built successfully.")


def search_similar_questions(
    query: str,
    k: int = 5
) -> List[Question]:

    global model

    if not INDEX_READY:
        build_index()

    if model is None or index.ntotal == 0:
        from services.question_svc import search_questions
        return search_questions(query)[:k]

    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(
        query_vector,
        k
    )

    results = []

    for idx in indices[0]:
        if 0 <= idx < len(question_registry):
            results.append(question_registry[idx])

    return results
