from pydantic import BaseModel
from typing import List, Optional

class Question(BaseModel):
    id: str
    exam: str
    year: Optional[int] = None
    subject: str
    topic: str
    question: str
    options: List[str]
    correct_option_index: int
    explanation: str

class TestGenerationRequest(BaseModel):
    exam: str
    subjects: List[str]
    count: int

class TutorRequest(BaseModel):
    question_id: Optional[str] = None
    query: str
    history: Optional[List[dict]] = None
