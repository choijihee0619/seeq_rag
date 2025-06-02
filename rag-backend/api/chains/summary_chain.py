"""
요약 체인
문서 요약 생성
"""
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class SummaryChain:
    """요약 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.llm_client = LLMClient()
        self.documents = db.documents
    
    async def process(
        self,
        document_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        summary_type: str = "brief"
    ) -> Dict:
        """요약 처리"""
        try:
            # 문서 조회
            filter_dict = {}
            if document_ids:
                filter_dict["_id"] = {"$in": document_ids}
            elif folder_id:
                filter_dict["folder_id"] = folder_id
            else:
                raise ValueError("document_ids 또는 folder_id 필요")
            
            documents = await self.documents.find(filter_dict).to_list(None)
            
            if not documents:
                return {
                    "summary": "요약할 문서가 없습니다.",
                    "document_count": 0
                }
            
            # 텍스트 결합
            combined_text = "\n\n".join([doc["text"] for doc in documents])
            
            # 프롬프트 선택
            if summary_type == "brief":
                prompt = f"다음 텍스트를 1-2문장으로 간단히 요약해주세요:\n\n{combined_text}"
            elif summary_type == "detailed":
                prompt = f"다음 텍스트를 상세하게 요약해주세요:\n\n{combined_text}"
            else:  # bullets
                prompt = f"다음 텍스트의 핵심 내용을 불릿 포인트로 정리해주세요:\n\n{combined_text}"
            
            # 요약 생성
            summary = await self.llm_client.generate(prompt, max_tokens=500)
            
            return {
                "summary": summary,
                "document_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"요약 처리 실패: {e}")
            raise
