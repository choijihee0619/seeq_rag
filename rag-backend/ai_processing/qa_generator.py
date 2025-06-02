"""
QA 생성 모듈
LLM을 사용한 질문-답변 쌍 및 퀴즈 생성
MODIFIED 2024-12-19: models 의존성 제거하여 단순화
"""
from typing import List, Dict
import json
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class QAGenerator:
    """QA 생성 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    async def generate_qa_pairs(
        self,
        text: str,
        num_pairs: int = 3
    ) -> List[Dict]:
        """텍스트에서 QA 쌍 생성"""
        prompt = f"""다음 텍스트를 읽고 {num_pairs}개의 질문-답변 쌍을 생성해주세요.

텍스트:
{text}

다음 JSON 배열 형식으로 응답해주세요:
[
    {{
        "question": "질문 내용",
        "answer": "답변 내용",
        "question_type": "factoid|reasoning|summary 중 하나",
        "difficulty": "easy|medium|hard 중 하나"
    }}
]
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.5,
                max_tokens=800
            )
            
            # JSON 파싱
            qa_pairs = json.loads(response)
            return qa_pairs
        except Exception as e:
            logger.error(f"QA 생성 실패: {e}")
            return []
    
    async def generate_quiz(
        self,
        text: str,
        quiz_type: str = "multiple_choice"
    ) -> Dict:
        """퀴즈 생성"""
        prompt = f"""다음 텍스트를 읽고 객관식 퀴즈를 생성해주세요.

텍스트:
{text}

다음 JSON 형식으로 응답해주세요:
{{
    "question": "퀴즈 질문",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
    "correct_option": 0,
    "explanation": "정답 설명",
    "difficulty": "easy|medium|hard 중 하나"
}}
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.5,
                max_tokens=400
            )
            
            # JSON 파싱
            quiz = json.loads(response)
            return quiz
        except Exception as e:
            logger.error(f"퀴즈 생성 실패: {e}")
            return {}
