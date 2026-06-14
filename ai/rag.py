```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from utils.schema import Question
from services.question_svc import load_all_questions
from services.question_svc import search_questions

# -------------------------
# Global Variables
# -------------------------

model = None
INDEX_READY = False

dimension = 384
index = faiss.IndexFlatL2(dimension)
question_registry = []


# -------------------------
# Load Embedding Model
# -------------------------

def get_model():
    global model

    if model is None:
        print("Loading embedding model...")

        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    return model


# -------------------------
# Build FAISS Index
# -------------------------

def build_index():

    global index
    global question_registry

    try:

        questions = load_all_questions()

        if not questions:
            print("No questions found.")
            return

        texts = [
            f"{q.subject} {q.topic} {q.question}"
            for q in questions
        ]

        print(f"Encoding {len(texts)} questions...")

        embeddings = (
            get_model()
            .encode(
                texts,
                convert_to_numpy=True
            )
            .astype("float32")
        )

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        question_registry = questions

        print("FAISS index ready.")

    except Exception as e:
        print(f"Index build error: {e}")


# -------------------------
# Search Function
# -------------------------

def search_similar_questions(
    query: str,
    k: int = 5
) -> List[Question]:

    global INDEX_READY

    try:

        if not INDEX_READY:

            print("Building index...")

            build_index()

            INDEX_READY = True

        if model is None or index.ntotal == 0:

            print("Using fallback search")

            return search_questions(query)[:k]

        query_vector = (
            get_model()
            .encode(
                [query],
                convert_to_numpy=True
            )
            .astype("float32")
        )

        distances, indices = index.search(
            query_vector,
            k
        )

        results = []

        for idx in indices[0]:

            if (
                idx >= 0
                and idx < len(question_registry)
            ):

                results.append(
                    question_registry[idx]
                )

        return results

    except Exception as e:

        print(
            f"Search failed: {e}"
        )

        return search_questions(query)[:k]
```
