import json
import os
from typing import List, Optional
from utils.schema import Question

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

def load_all_questions() -> List[Question]:
    questions = []
    if not os.path.exists(DATASETS_DIR):
        return questions
        
    for filename in os.listdir(DATASETS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATASETS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for item in data:
                        questions.append(Question(**item))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return questions

def get_all_questions(exam: Optional[str] = None, subject: Optional[str] = None, limit: int = 20) -> List[Question]:
    all_q = load_all_questions()
    filtered = all_q
    
    if exam:
        filtered = [q for q in filtered if q.exam.lower() == exam.lower()]
    if subject:
        filtered = [q for q in filtered if q.subject.lower() == subject.lower()]
        
    return filtered[:limit]

def search_questions(query: str) -> List[Question]:
    # Placeholder for semantic search (RAG phase)
    # Currently just does a naive text search
    all_q = load_all_questions()
    query_lower = query.lower()
    
    results = [q for q in all_q if query_lower in q.question.lower() or query_lower in q.topic.lower()]
    return results[:10]
