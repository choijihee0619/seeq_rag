"""
요약 체인
문서 요약 생성
"""
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger
from bson import ObjectId

logger = get_logger(__name__)

class SummaryChain:
    """요약 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.llm_client = LLMClient()
        self.documents = db.documents
        self.chunks = db.chunks
    
    async def process(
        self,
        document_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        summary_type: str = "brief"
    ) -> Dict:
        """요약 처리"""
        try:
            # document_ids 정리 - "string", "null" 같은 기본값 제거
            clean_document_ids = None
            if document_ids:
                clean_document_ids = [
                    doc_id.strip() for doc_id in document_ids 
                    if doc_id and doc_id.strip() and doc_id.strip() not in ["string", "null"]
                ]
                if not clean_document_ids:
                    clean_document_ids = None
            
            # folder_id 정리
            clean_folder_id = None
            if folder_id and folder_id.strip() and folder_id.strip() not in ["string", "null"]:
                clean_folder_id = folder_id.strip()
            
            logger.info(f"요약 처리 시작 - document_ids: {clean_document_ids}, folder_id: '{clean_folder_id}'")
            
            # 문서 조회 조건 설정
            if clean_document_ids:
                # clean_document_ids가 제공된 경우, 해당 file_id들을 가진 문서들 찾기
                filter_dict = {"file_id": {"$in": clean_document_ids}}
                documents = await self.documents.find(filter_dict).to_list(None)
                
                if not documents:
                    return {
                        "summary": "요약할 문서가 없습니다.",
                        "document_count": 0
                    }
                
                # 해당 문서들의 텍스트 사용 (processed_text 또는 raw_text)
                texts = []
                for doc in documents:
                    text = doc.get("processed_text", doc.get("raw_text", ""))
                    if text:
                        texts.append(text)
                
                if not texts:
                    return {
                        "summary": "요약할 텍스트가 없습니다.",
                        "document_count": len(documents)
                    }
                
                combined_text = "\n\n".join(texts)
                document_count = len(documents)
                
            elif clean_folder_id:
                # clean_folder_id가 제공된 경우, 해당 폴더의 모든 청크들 찾기
                logger.info(f"폴더별 요약 처리: {clean_folder_id}")
                
                # 먼저 해당 폴더의 문서들 찾기
                folder_docs = await self.documents.find({"folder_id": clean_folder_id}).to_list(None)
                if not folder_docs:
                    return {
                        "summary": "요약할 문서가 없습니다.",
                        "document_count": 0
                    }
                
                logger.info(f"폴더 '{clean_folder_id}'에서 {len(folder_docs)}개 문서 발견")
                
                # 해당 문서들의 file_id 목록 만들기
                file_ids = [doc["file_id"] for doc in folder_docs]
                
                # 해당 file_id들의 청크들 찾기
                chunks = await self.chunks.find({"file_id": {"$in": file_ids}}).sort("sequence", 1).to_list(None)
                
                if not chunks:
                    return {
                        "summary": "요약할 청크가 없습니다.",
                        "document_count": len(folder_docs)
                    }
                
                logger.info(f"총 {len(chunks)}개 청크에서 텍스트 추출")
                
                # 청크들의 텍스트 결합
                combined_text = "\n\n".join([chunk["text"] for chunk in chunks])
                document_count = len(folder_docs)
                
            else:
                raise ValueError("document_ids 또는 folder_id가 필요합니다.")
            
            # 텍스트가 너무 긴 경우 제한
            if len(combined_text) > 8000:  # 토큰 제한 고려
                combined_text = combined_text[:8000] + "..."
                logger.warning("텍스트가 너무 길어 8000자로 제한했습니다.")
            
            logger.info(f"요약할 텍스트 길이: {len(combined_text)} 문자")
            
            # 프롬프트 선택
            if summary_type == "brief":
                prompt = f"다음 텍스트를 1-2문장으로 간단히 요약해주세요:\n\n{combined_text}"
            elif summary_type == "detailed":
                prompt = f"다음 텍스트를 상세하게 요약해주세요:\n\n{combined_text}"
            else:  # bullets
                prompt = f"다음 텍스트의 핵심 내용을 불릿 포인트로 정리해주세요:\n\n{combined_text}"
            
            # 요약 생성
            summary = await self.llm_client.generate(prompt, max_tokens=500)
            
            logger.info(f"요약 완료 - 문서 수: {document_count}, 요약 길이: {len(summary)}")
            
            return {
                "summary": summary,
                "document_count": document_count
            }
            
        except Exception as e:
            logger.error(f"요약 처리 실패: {e}")
            raise
