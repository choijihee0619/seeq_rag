"""
퀴즈 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.chains.quiz_chain import QuizChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class QuizRequest(BaseModel):
    """퀴즈 요청 모델"""
    topic: Optional[str] = None
    folder_id: Optional[str] = None
    difficulty: str = "medium"
    count: int = 5
    quiz_type: str = "multiple_choice"

class QuizItem(BaseModel):
    """퀴즈 항목 모델"""
    question: str
    quiz_type: Optional[str] = "multiple_choice"  # multiple_choice, true_false, short_answer, fill_in_blank
    options: Optional[List[str]] = None  # 객관식/참거짓용
    correct_option: Optional[int] = None  # 객관식/참거짓용 정답 인덱스
    correct_answer: Optional[str] = None  # 단답형/빈칸채우기용 정답
    difficulty: str
    explanation: Optional[str] = None

class QuizResponse(BaseModel):
    """퀴즈 응답 모델"""
    quizzes: List[QuizItem]
    topic: Optional[str]
    total_count: int

@router.post("/", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    """퀴즈 생성 엔드포인트"""
    try:
        db = await get_database()
        quiz_chain = QuizChain(db)
        
        # 퀴즈 생성
        result = await quiz_chain.process(
            topic=request.topic,
            folder_id=request.folder_id,
            difficulty=request.difficulty,
            count=request.count,
            quiz_type=request.quiz_type
        )
        
        # 퀴즈 항목 변환
        quiz_items = []
        for quiz in result["quizzes"]:
            quiz_items.append(QuizItem(
                question=quiz["question"],
                quiz_type=quiz.get("quiz_type", "multiple_choice"),
                options=quiz.get("options"),
                correct_option=quiz.get("correct_option"),
                correct_answer=quiz.get("correct_answer"),
                difficulty=quiz["difficulty"],
                explanation=quiz.get("explanation")
            ))
        
        return QuizResponse(
            quizzes=quiz_items,
            topic=request.topic,
            total_count=len(quiz_items)
        )
        
    except Exception as e:
        logger.error(f"퀴즈 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
