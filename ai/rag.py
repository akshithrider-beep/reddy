from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from utils.schema import Question
from services.question_svc import load_all_questions

model = None
index = None
question_registry = []

dimension = 384
INDEX_READY = False


def get_model():
    global model

    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    return model


def build_index():
    global index
    global question_registry
    global INDEX_READY

    if INDEX_READY:
        return

    model = get_model()

    questions = load_all_questions()

    if not questions:
        print("No questions found")
        return

    texts = [
        f"{q.subject} {q.topic} {q.question}"
        for q in questions
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    question_registry = questions

    INDEX_READY = True

    print("Index ready")


def search_similar_questions(
    query: str,
    k: int = 5
) -> List[Question]:

    build_index()

    query_vector = (
        get_model()
        .encode([query])
        .astype("float32")
    )

    distances, indices = index.search(
        query_vector,
        k
    )

    results = []

    for idx in indices[0]:
        if idx >= 0:
            results.append(
                question_registry[idx]
            )

    return results
