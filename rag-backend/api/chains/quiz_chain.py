"""
퀴즈 체인
퀴즈 생성 파이프라인
"""
from typing import Dict, Optional, List
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
        self.documents = db.documents
        self.chunks = db.chunks
    
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
                texts_for_quiz = []
                
                if topic:
                    # 주제 관련 청크 검색
                    search_results = await self.hybrid_search.search(
                        query=topic,
                        k=10,  # 더 많은 청크에서 텍스트 수집
                        folder_id=folder_id
                    )
                    
                    # 검색된 청크들의 텍스트 수집
                    for result in search_results:
                        chunk_text = result.get("chunk", {}).get("text", "")
                        if chunk_text and len(chunk_text) > 100:  # 충분한 길이의 텍스트만
                            texts_for_quiz.append(chunk_text)
                    
                else:
                    # topic이 없으면 폴더 또는 전체에서 텍스트 수집
                    filter_dict = {}
                    if folder_id:
                        # folder_id가 제공된 경우
                        docs = await self.documents.find({"folder_id": folder_id}).to_list(10)
                        if docs:
                            file_ids = [doc["file_id"] for doc in docs]
                            chunks = await self.chunks.find({"file_id": {"$in": file_ids}}).limit(10).to_list(None)
                        else:
                            chunks = []
                    else:
                        # 전체에서 샘플링
                        chunks = await self.chunks.find({}).limit(10).to_list(None)
                    
                    for chunk in chunks:
                        chunk_text = chunk.get("text", "")
                        if chunk_text and len(chunk_text) > 100:
                            texts_for_quiz.append(chunk_text)
                
                # 수집된 텍스트로 퀴즈 생성
                logger.info(f"퀴즈 생성용 텍스트 {len(texts_for_quiz)}개 수집")
                
                for i, text in enumerate(texts_for_quiz):
                    if len(quizzes) >= count:
                        break
                        
                    try:
                        quiz = await self.qa_generator.generate_quiz(
                            text=text,
                            quiz_type=quiz_type,
                            difficulty=difficulty
                        )
                        
                        if quiz:
                            quizzes.append(quiz)
                            logger.info(f"퀴즈 {len(quizzes)}/{count} 생성 완료")
                            
                    except Exception as e:
                        logger.warning(f"텍스트 {i+1}에서 퀴즈 생성 실패: {e}")
                        continue
                
                # 생성된 퀴즈가 없는 경우 기본 퀴즈 제공
                if not quizzes:
                    quizzes = self._generate_fallback_quizzes(topic or "일반", difficulty, count)
            
            return {
                "quizzes": quizzes[:count]
            }
            
        except Exception as e:
            logger.error(f"퀴즈 처리 실패: {e}")
            raise
    
    def _generate_fallback_quizzes(self, topic: str, difficulty: str, count: int) -> List[Dict]:
        """기본 퀴즈 생성 (텍스트가 없을 때)"""
        fallback_quizzes = [
            {
                "question": f"{topic}에 대한 질문입니다. 현재 시스템에 관련 문서가 부족합니다.",
                "options": ["데이터가 필요합니다", "문서를 업로드해주세요", "퀴즈 생성 불가", "다시 시도해주세요"],
                "correct_option": 1,
                "difficulty": difficulty,
                "explanation": f"{topic}에 대한 퀴즈를 생성하려면 관련 문서가 필요합니다."
            }
        ]
        
        return fallback_quizzes * min(count, 1)  # 최대 1개만 반환
