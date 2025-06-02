"""
퀴즈 체인
퀴즈 생성 파이프라인
"""
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from ai_processing.qa_generator import QAGenerator
from retrieval.hybrid_search import HybridSearch
from utils.logger import get_logger

logger = get_logger(__name__)

class QuizChain:
    """퀴즈 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.qa_generator = QAGenerator()
        self.hybrid_search = HybridSearch(db)
        self.qapairs = db.qapairs
    
    async def process(
        self,
        topic: Optional[str] = None,
        folder_id: Optional[str] = None,
        difficulty: str = "medium",
        count: int = 5,
        quiz_type: str = "multiple_choice"
    ) -> Dict:
        """퀴즈 처리"""
        try:
            quizzes = []
            
            # 기존 퀴즈 조회
            filter_dict = {"difficulty": difficulty}
            if folder_id:
                filter_dict["folder_id"] = folder_id
            
            existing_quizzes = await self.qapairs.find(
                filter_dict
            ).limit(count).to_list(None)
            
            # 기존 퀴즈가 충분하면 반환
            if len(existing_quizzes) >= count:
                for quiz in existing_quizzes[:count]:
                    quizzes.append({
                        "question": quiz["question"],
                        "options": quiz.get("quiz_options"),
                        "correct_option": quiz.get("correct_option"),
                        "difficulty": quiz["difficulty"],
                        "explanation": quiz.get("answer")
                    })
            else:
                # 새로운 퀴즈 생성 필요
                if topic:
                    # 주제 관련 문서 검색
                    search_results = await self.hybrid_search.search(
                        query=topic,
                        k=3,
                        folder_id=folder_id
                    )
                    
                    # 각 문서에서 퀴즈 생성
                    for result in search_results:
                        text = result["document"]["text"]
                        quiz = await self.qa_generator.generate_quiz(
                            text=text,
                            quiz_type=quiz_type
                        )
                        
                        if quiz:
                            quiz["difficulty"] = difficulty
                            quizzes.append(quiz)
                            
                            if len(quizzes) >= count:
                                break
            
            return {
                "quizzes": quizzes[:count]
            }
            
        except Exception as e:
            logger.error(f"퀴즈 처리 실패: {e}")
            raise
