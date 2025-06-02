"""
추천 체인
콘텐츠 추천 생성
MODIFIED 2024-01-20: YouTube API 연동 추가 - 실시간 YouTube 동영상 추천 기능 통합
"""
from typing import Dict, List
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger
from utils.youtube_api import youtube_api

logger = get_logger(__name__)

class RecommendChain:
    """추천 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.recommendations = db.recommendations
    
    async def process(
        self,
        keywords: List[str],
        content_types: List[str] = ["book", "movie", "video"],
        max_items: int = 10,
        include_youtube: bool = True,
        youtube_max_per_keyword: int = 3
    ) -> Dict:
        """
        추천 처리
        
        Args:
            keywords: 검색 키워드 리스트
            content_types: 콘텐츠 타입 리스트
            max_items: 최대 추천 항목 수
            include_youtube: YouTube 검색 포함 여부
            youtube_max_per_keyword: 키워드당 YouTube 결과 최대 수
        """
        try:
            recommendations = []
            
            # 1. 기존 DB에서 저장된 추천 검색
            db_recommendations = await self._search_db_recommendations(
                keywords, content_types, max_items
            )
            recommendations.extend(db_recommendations)
            
            # 2. YouTube 실시간 검색 (video 타입이 포함된 경우)
            if include_youtube and ("video" in content_types or "youtube_video" in content_types):
                youtube_recommendations = await self._search_youtube_recommendations(
                    keywords, youtube_max_per_keyword
                )
                recommendations.extend(youtube_recommendations)
            
            # 3. 결과 정렬 및 제한
            # 다양성을 위해 키워드별로 균등하게 분배
            final_recommendations = self._balance_recommendations(
                recommendations, keywords, max_items
            )
            
            # 추천이 없는 경우 더미 데이터 생성
            if not final_recommendations:
                final_recommendations = self._generate_fallback_recommendations(keywords)
            
            return {
                "recommendations": final_recommendations[:max_items]
            }
            
        except Exception as e:
            logger.error(f"추천 처리 실패: {e}")
            raise

    async def _search_db_recommendations(
        self,
        keywords: List[str],
        content_types: List[str],
        max_items: int
    ) -> List[Dict]:
        """DB에서 저장된 추천 검색"""
        recommendations = []
        
        for keyword in keywords:
            filter_dict = {
                "keyword": keyword,
                "content_type": {"$in": content_types}
            }
            
            items = await self.recommendations.find(
                filter_dict
            ).limit(max_items // len(keywords)).to_list(None)
            
            for item in items:
                recommendations.append({
                    "title": item["title"],
                    "content_type": item["content_type"],
                    "description": item.get("description"),
                    "source": item["source"],
                    "metadata": item.get("metadata", {}),
                    "keyword": keyword,
                    "recommendation_source": "database"
                })
        
        return recommendations

    async def _search_youtube_recommendations(
        self,
        keywords: List[str],
        max_per_keyword: int
    ) -> List[Dict]:
        """YouTube에서 실시간 동영상 추천 검색"""
        youtube_recommendations = []
        
        for keyword in keywords:
            try:
                # YouTube 검색 실행
                youtube_videos = await youtube_api.search_videos(
                    query=keyword,
                    max_results=max_per_keyword,
                    order="relevance",
                    video_duration="medium"  # 중간 길이 동영상 우선
                )
                
                # 결과를 표준 추천 형식으로 변환
                for video in youtube_videos:
                    youtube_recommendations.append({
                        "title": video["title"],
                        "content_type": "youtube_video",
                        "description": video["description"],
                        "source": "youtube",
                        "metadata": {
                            "video_id": video["video_id"],
                            "video_url": video["video_url"],
                            "channel_title": video["channel_title"],
                            "channel_id": video["channel_id"],
                            "thumbnail_url": video["thumbnail_url"],
                            "view_count": video["view_count"],
                            "like_count": video["like_count"],
                            "duration": video["duration"],
                            "duration_seconds": video["duration_seconds"],
                            "published_at": video["published_at"],
                            "tags": video["tags"]
                        },
                        "keyword": keyword,
                        "recommendation_source": "youtube_realtime"
                    })
                
                logger.info(f"YouTube 검색 완료: '{keyword}' - {len(youtube_videos)}개 결과")
                
            except Exception as e:
                logger.error(f"YouTube 검색 실패 (키워드: {keyword}): {e}")
                continue
        
        return youtube_recommendations

    def _balance_recommendations(
        self,
        recommendations: List[Dict],
        keywords: List[str],
        max_items: int
    ) -> List[Dict]:
        """키워드별로 추천을 균등하게 분배"""
        if not recommendations:
            return []
        
        # 키워드별로 그룹화
        keyword_groups = {}
        for rec in recommendations:
            keyword = rec.get("keyword", "unknown")
            if keyword not in keyword_groups:
                keyword_groups[keyword] = []
            keyword_groups[keyword].append(rec)
        
        # 라운드 로빈 방식으로 균등 분배
        balanced = []
        max_per_keyword = max(1, max_items // len(keywords)) if keywords else max_items
        
        # 각 키워드에서 순서대로 선택
        for i in range(max_per_keyword):
            for keyword in keywords:
                if keyword in keyword_groups and i < len(keyword_groups[keyword]):
                    balanced.append(keyword_groups[keyword][i])
                    if len(balanced) >= max_items:
                        return balanced
        
        # 남은 자리가 있으면 추가로 채움
        for keyword in keywords:
            if keyword in keyword_groups:
                for rec in keyword_groups[keyword][max_per_keyword:]:
                    balanced.append(rec)
                    if len(balanced) >= max_items:
                        break
            if len(balanced) >= max_items:
                break
        
        return balanced

    def _generate_fallback_recommendations(self, keywords: List[str]) -> List[Dict]:
        """추천이 없는 경우 기본 추천 생성"""
        fallback_recommendations = []
        
        for keyword in keywords[:3]:  # 최대 3개 키워드까지
            fallback_recommendations.extend([
                {
                    "title": f"{keyword} 관련 도서 추천",
                    "content_type": "book",
                    "description": f"{keyword}에 관한 유용한 도서를 곧 추천해드릴 예정입니다.",
                    "source": "internal",
                    "metadata": {},
                    "keyword": keyword,
                    "recommendation_source": "fallback"
                },
                {
                    "title": f"{keyword} 관련 동영상 추천",
                    "content_type": "video",
                    "description": f"{keyword}에 관한 교육 동영상을 곧 추천해드릴 예정입니다.",
                    "source": "internal", 
                    "metadata": {},
                    "keyword": keyword,
                    "recommendation_source": "fallback"
                }
            ])
        
        return fallback_recommendations

    async def save_youtube_recommendation(
        self,
        keyword: str,
        video_info: Dict
    ) -> bool:
        """
        YouTube 추천을 DB에 저장 (캐싱 목적)
        
        Args:
            keyword: 검색 키워드
            video_info: YouTube 동영상 정보
        """
        try:
            recommendation_doc = {
                "keyword": keyword,
                "content_type": "youtube_video",
                "content_id": video_info["video_id"],
                "title": video_info["title"],
                "description": video_info["description"],
                "source": "youtube",
                "metadata": video_info
            }
            
            # 중복 체크 (같은 키워드 + 동영상 ID)
            existing = await self.recommendations.find_one({
                "keyword": keyword,
                "content_id": video_info["video_id"],
                "source": "youtube"
            })
            
            if not existing:
                await self.recommendations.insert_one(recommendation_doc)
                logger.info(f"YouTube 추천 저장: {keyword} - {video_info['title']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"YouTube 추천 저장 실패: {e}")
            return False

    async def get_youtube_trending(
        self,
        category_id: str = "0",  # 전체 카테고리
        max_results: int = 10
    ) -> List[Dict]:
        """
        YouTube 인기 동영상 가져오기
        
        Args:
            category_id: 카테고리 ID (0: 전체)
            max_results: 최대 결과 수
        """
        try:
            # YouTube API로 인기 동영상 검색
            # 실제로는 trending API를 사용해야 하지만, 여기서는 인기도 기반 검색으로 대체
            trending_videos = await youtube_api.search_videos(
                query="인기",  # 한국어 인기 키워드
                max_results=max_results,
                order="viewCount"  # 조회수 순 정렬
            )
            
            recommendations = []
            for video in trending_videos:
                recommendations.append({
                    "title": video["title"],
                    "content_type": "youtube_video",
                    "description": video["description"],
                    "source": "youtube_trending",
                    "metadata": video,
                    "keyword": "trending",
                    "recommendation_source": "youtube_trending"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"YouTube 인기 동영상 조회 실패: {e}")
            return []
