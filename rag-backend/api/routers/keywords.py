"""
키워드 추출 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ai_processing.labeler import AutoLabeler
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class KeywordsRequest(BaseModel):
    """키워드 추출 요청 모델"""
    text: str
    max_keywords: int = 10

class KeywordsResponse(BaseModel):
    """키워드 추출 응답 모델"""
    keywords: List[str]
    count: int

@router.post("/", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest):
    """키워드 추출 엔드포인트"""
    try:
        labeler = AutoLabeler()
        
        # 키워드 추출
        keywords = await labeler.extract_keywords(
            text=request.text,
            max_keywords=request.max_keywords
        )
        
        return KeywordsResponse(
            keywords=keywords,
            count=len(keywords)
        )
        
    except Exception as e:
        logger.error(f"키워드 추출 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
