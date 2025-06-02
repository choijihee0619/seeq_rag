"""
추천 API 라우터
MODIFIED 2024-01-20: YouTube 연동 옵션 추가 및 응답 모델 확장
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from api.chains.recommend_chain import RecommendChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class RecommendRequest(BaseModel):
    """추천 요청 모델"""
    keywords: List[str]
    content_types: List[str] = ["book", "movie", "video", "youtube_video"]
    max_items: int = 10
    include_youtube: bool = True  # YouTube 검색 포함 여부
    youtube_max_per_keyword: int = 3  # 키워드당 YouTube 결과 수

class RecommendItem(BaseModel):
    """추천 항목 모델"""
    title: str
    content_type: str
    description: Optional[str]
    source: str
    metadata: Dict
    keyword: Optional[str] = None  # 어떤 키워드로 검색된 항목인지
    recommendation_source: Optional[str] = None  # 추천 소스 (database, youtube_realtime, fallback)

class RecommendResponse(BaseModel):
    """추천 응답 모델"""
    recommendations: List[RecommendItem]
    total_count: int
    youtube_included: bool  # YouTube 결과 포함 여부
    sources_summary: Dict  # 소스별 개수 요약

class YouTubeTrendingRequest(BaseModel):
    """YouTube 인기 동영상 요청 모델"""
    category_id: str = "0"  # 카테고리 ID (0: 전체)
    max_results: int = 10

@router.post("/", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """콘텐츠 추천 엔드포인트"""
    try:
        db = await get_database()
        recommend_chain = RecommendChain(db)
        
        # 추천 생성
        result = await recommend_chain.process(
            keywords=request.keywords,
            content_types=request.content_types,
            max_items=request.max_items,
            include_youtube=request.include_youtube,
            youtube_max_per_keyword=request.youtube_max_per_keyword
        )
        
        # 추천 항목 변환
        recommendations = []
        sources_count = {}
        youtube_count = 0
        
        for item in result["recommendations"]:
            recommendation = RecommendItem(
                title=item["title"],
                content_type=item["content_type"],
                description=item.get("description"),
                source=item["source"],
                metadata=item.get("metadata", {}),
                keyword=item.get("keyword"),
                recommendation_source=item.get("recommendation_source")
            )
            recommendations.append(recommendation)
            
            # 소스별 개수 집계
            source = item.get("recommendation_source", "unknown")
            sources_count[source] = sources_count.get(source, 0) + 1
            
            if item["content_type"] == "youtube_video":
                youtube_count += 1
        
        return RecommendResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            youtube_included=youtube_count > 0,
            sources_summary=sources_count
        )
        
    except Exception as e:
        logger.error(f"추천 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/youtube/trending", response_model=RecommendResponse)
async def get_youtube_trending(request: YouTubeTrendingRequest):
    """YouTube 인기 동영상 추천 엔드포인트"""
    try:
        db = await get_database()
        recommend_chain = RecommendChain(db)
        
        # YouTube 인기 동영상 가져오기
        trending_videos = await recommend_chain.get_youtube_trending(
            category_id=request.category_id,
            max_results=request.max_results
        )
        
        # 응답 형식으로 변환
        recommendations = []
        for item in trending_videos:
            recommendation = RecommendItem(
                title=item["title"],
                content_type=item["content_type"],
                description=item.get("description"),
                source=item["source"],
                metadata=item.get("metadata", {}),
                keyword=item.get("keyword"),
                recommendation_source=item.get("recommendation_source")
            )
            recommendations.append(recommendation)
        
        return RecommendResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            youtube_included=True,
            sources_summary={"youtube_trending": len(recommendations)}
        )
        
    except Exception as e:
        logger.error(f"YouTube 인기 동영상 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/youtube/categories")
async def get_youtube_categories():
    """YouTube 카테고리 목록 조회"""
    try:
        from utils.youtube_api import youtube_api
        categories = await youtube_api.get_video_categories(region_code='KR')
        
        return {
            "categories": categories,
            "total_count": len(categories)
        }
        
    except Exception as e:
        logger.error(f"YouTube 카테고리 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/youtube/save")
async def save_youtube_recommendation(
    keyword: str,
    video_id: str
):
    """
    YouTube 동영상을 추천으로 저장
    관리자가 수동으로 좋은 동영상을 추천 DB에 저장할 때 사용
    """
    try:
        from utils.youtube_api import youtube_api
        db = await get_database()
        recommend_chain = RecommendChain(db)
        
        # 동영상 정보 가져오기
        videos = await youtube_api.search_videos(
            query=f"video_id:{video_id}",
            max_results=1
        )
        
        if not videos:
            raise HTTPException(status_code=404, detail="동영상을 찾을 수 없습니다")
        
        video_info = videos[0]
        
        # DB에 저장
        success = await recommend_chain.save_youtube_recommendation(
            keyword=keyword,
            video_info=video_info
        )
        
        if success:
            return {
                "message": "YouTube 추천이 성공적으로 저장되었습니다",
                "keyword": keyword,
                "video_title": video_info["title"],
                "video_id": video_id
            }
        else:
            return {
                "message": "이미 저장된 추천입니다",
                "keyword": keyword,
                "video_id": video_id
            }
        
    except Exception as e:
        logger.error(f"YouTube 추천 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
