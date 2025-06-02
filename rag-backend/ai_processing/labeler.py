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
        """텍스트에서 라벨 추출 - 환각 방지 강화"""
        prompt = f"""
        **환각 방지 지침: 텍스트 내용에만 기반한 분석**
        
        다음 텍스트를 분석하여 **실제 내용에만 기반한** 정보를 추출해주세요.
        
        **절대 금지사항:**
        1. 텍스트에 없는 정보 추가 금지
        2. 추측이나 가정에 기반한 분석 금지
        3. 일반적 지식으로 빈 공간 채우기 금지
        4. 불확실한 내용은 "알 수 없음"으로 표시
        
        **텍스트:**
        {text[:2000]}
        
        **요구사항:** 텍스트에서 실제로 확인 가능한 내용만 포함하여 JSON으로 응답
        
        {{
            "main_topic": "텍스트에서 명확히 드러나는 주요 주제",
            "tags": ["실제로 언급된 키워드들만"],
            "category": "텍스트 내용에서 판단되는 카테고리",
            "summary": "텍스트 내용만으로 작성한 한 문장 요약"
        }}
        """
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.1,  # 창의성 최소화
                max_tokens=200
            )
            
            # JSON 파싱
            labels = json.loads(response)
            labels["confidence"] = 0.8  # 환각 방지로 신뢰도 조정
            
            logger.info(f"라벨 추출 완료: {labels.get('main_topic', 'N/A')}")
            return labels
            
        except Exception as e:
            logger.error(f"라벨 추출 실패: {e}")
            # 기본값 반환 (환각 없는 안전한 기본값)
            return {
                "main_topic": "분석 불가",
                "tags": [],
                "category": "미분류",
                "summary": text[:100] + "..." if len(text) > 100 else text,
                "confidence": 0.3
            }
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출 - 환각 방지 강화"""
        prompt = f"""
        **환각 방지 중요 지침: 제공된 텍스트에만 기반하여 키워드 추출**
        
        다음 텍스트에서 **실제로 언급된** 가장 중요한 키워드를 {max_keywords}개 추출해주세요.
        
        **반드시 준수할 규칙:**
        1. 텍스트에 명시적으로 나타난 단어/개념만 사용하세요
        2. 추측하거나 연관성만으로 키워드를 생성하지 마세요  
        3. 텍스트에 없는 내용은 절대 추가하지 마세요
        4. 불확실하면 키워드 수를 줄여서라도 정확한 것만 선택하세요
        5. 일반적인 단어(예: "있다", "하다", "것", "수")는 제외하세요
        
        **텍스트:**
        {text[:3000]}  
        
        **응답 형식:** 키워드를 쉼표로 구분하여 나열 (예: 키워드1, 키워드2, 키워드3)
        **주의:** 텍스트에서 실제로 찾을 수 있는 키워드만 포함하세요.
        """
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.1,  # 창의성 최소화
                max_tokens=100
            )
            
            # 키워드 파싱 및 정제
            raw_keywords = [k.strip() for k in response.split(',')]
            
            # 빈 키워드나 너무 짧은 키워드 제거
            keywords = []
            for keyword in raw_keywords[:max_keywords]:
                keyword = keyword.strip()
                if len(keyword) >= 2 and keyword.lower() not in ['있다', '하다', '것', '수', '등', '및', '또는']:
                    keywords.append(keyword)
            
            logger.info(f"키워드 추출 완료: {len(keywords)}개 - {keywords}")
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
