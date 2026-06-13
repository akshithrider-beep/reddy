from fastapi import APIRouter, HTTPException
from typing import List, Optional
from utils.schema import Question, TutorRequest
from services.question_svc import get_all_questions, load_all_questions
from ai.rag import search_similar_questions
from ai.llm_client import ask_tutor

router = APIRouter()

@router.get("/questions", response_model=List[Question])
def fetch_questions(exam: Optional[str] = None, subject: Optional[str] = None, limit: int = 20):
    return get_all_questions(exam=exam, subject=subject, limit=limit)

from ai.rag import search_similar_questions

@router.get("/questions/search", response_model=List[Question])
def search_questions_api(q: str):
    return search_similar_questions(q)

@router.post("/tutor/ask")
def ask_tutor_api(request: TutorRequest):
    question_context = None
    if request.question_id:
        # Find the question context
        all_q = load_all_questions()
        for q in all_q:
            if q.id == request.question_id:
                question_context = q.dict()
                break
                
    answer = ask_tutor(request.query, question_context)
    return {"answer": answer}
