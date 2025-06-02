"""
요약 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.chains.summary_chain import SummaryChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class SummaryRequest(BaseModel):
    """요약 요청 모델"""
    document_ids: Optional[List[str]] = None
    folder_id: Optional[str] = None
    summary_type: str = "brief"  # brief, detailed, bullets

class SummaryResponse(BaseModel):
    """요약 응답 모델"""
    summary: str
    document_count: int
    summary_type: str

@router.post("/", response_model=SummaryResponse)
async def create_summary(request: SummaryRequest):
    """요약 생성 엔드포인트"""
    try:
        db = await get_database()
        summary_chain = SummaryChain(db)
        
        # 요약 생성
        result = await summary_chain.process(
            document_ids=request.document_ids,
            folder_id=request.folder_id,
            summary_type=request.summary_type
        )
        
        return SummaryResponse(
            summary=result["summary"],
            document_count=result["document_count"],
            summary_type=request.summary_type
        )
        
    except Exception as e:
        logger.error(f"요약 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
