"""
라벨링 모듈
LLM을 사용한 자동 라벨링 및 키워드 추출
"""
from typing import List, Dict
import json
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class AutoLabeler:
    """자동 라벨링 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    async def extract_labels(self, text: str) -> Dict:
        """텍스트에서 라벨 추출"""
        prompt = f"""다음 텍스트를 분석하여 주요 정보를 추출해주세요.

텍스트:
{text}

다음 JSON 형식으로 응답해주세요:
{{
    "main_topic": "주요 주제 (한 문장)",
    "tags": ["태그1", "태그2", "태그3"],
    "category": "카테고리명",
    "summary": "한 문장 요약"
}}
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # JSON 파싱
            labels = json.loads(response)
            labels["confidence"] = 0.9  # 기본 신뢰도
            
            return labels
        except Exception as e:
            logger.error(f"라벨 추출 실패: {e}")
            # 기본값 반환
            return {
                "main_topic": "알 수 없음",
                "tags": [],
                "category": "기타",
                "summary": text[:100] + "...",
                "confidence": 0.5
            }
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출"""
        prompt = f"""다음 텍스트에서 가장 중요한 키워드를 {max_keywords}개 추출해주세요.

텍스트:
{text}

키워드를 쉼표로 구분하여 나열해주세요:"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            # 키워드 파싱
            keywords = [k.strip() for k in response.split(',')]
            return keywords[:max_keywords]
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
