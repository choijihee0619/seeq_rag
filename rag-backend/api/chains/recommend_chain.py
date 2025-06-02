"""
추천 체인
콘텐츠 추천 생성
MODIFIED 2024-01-20: YouTube API 연동 추가 - 실시간 YouTube 동영상 추천 기능 통합
ENHANCED 2024-01-21: 파일 기반 키워드 자동 추출 기능 추가
CLEANED 2024-01-21: 불필요한 YouTube 개별 API 제거, 핵심 추천 기능만 유지
REFACTORED 2024-01-21: 키워드 추출 통합 및 TextCollector 적용
"""
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger
from utils.youtube_api import youtube_api
from utils.text_collector import TextCollector
from ai_processing.labeler import AutoLabeler
from utils.web_recommendation import web_recommendation_engine

logger = get_logger(__name__)

class RecommendChain:
    """추천 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.recommendations = db.recommendations
        self.documents = db.documents  # documents 컬렉션 추가
        self.chunks = db.chunks        # chunks 컬렉션 추가
    
    async def process(
        self,
        keywords: List[str],
        content_types: List[str] = ["book", "movie", "youtube_video"],
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
            
            # 2. YouTube 실시간 검색 (include_youtube가 True이고 video 관련 타입이 포함된 경우)
            if include_youtube and ("video" in content_types or "youtube_video" in content_types):
                youtube_recommendations = await self._search_youtube_recommendations(
                    keywords, youtube_max_per_keyword
                )
                recommendations.extend(youtube_recommendations)
            
            # 3. 웹 검색 기반 실시간 추천 (book, movie, video 타입)
            web_recommendations = await self._search_web_recommendations(
                keywords, content_types, max_items
            )
            recommendations.extend(web_recommendations)
            
            # 4. 결과 정렬 및 제한
            # 다양성을 위해 키워드별로 균등하게 분배
            final_recommendations = self._balance_recommendations(
                recommendations, keywords, max_items
            )
            
            # 5. 추천이 부족한 경우에만 fallback 데이터 추가
            if len(final_recommendations) < max_items:
                fallback_recommendations = self._generate_fallback_recommendations(keywords)
                final_recommendations.extend(fallback_recommendations)
            
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
        """저장된 추천에서 검색"""
        try:
            # MongoDB 텍스트 검색 사용
            search_query = " ".join(keywords)
            
            cursor = self.recommendations.find({
                "$text": {"$search": search_query},
                "content_type": {"$in": content_types}
            }).limit(max_items)
            
            recommendations = []
            async for doc in cursor:
                recommendations.append({
                    "title": doc["title"],
                    "content_type": doc["content_type"],
                    "description": doc.get("description"),
                    "source": doc.get("source", "database"),
                    "metadata": doc.get("metadata", {}),
                    "keyword": keywords[0] if keywords else "",
                    "recommendation_source": "database"
                })
            
            logger.info(f"DB에서 {len(recommendations)}개 추천 검색")
            return recommendations
            
        except Exception as e:
            logger.warning(f"DB 추천 검색 실패: {e}")
            return []

    async def _search_youtube_recommendations(
        self,
        keywords: List[str],
        max_per_keyword: int
    ) -> List[Dict]:
        """YouTube에서 실시간 추천 검색"""
        try:
            logger.info(f"YouTube API 상태 확인 중...")
            logger.info(f"YouTube API 사용 가능: {youtube_api.is_available()}")
            logger.info(f"YouTube API 키 존재: {youtube_api.api_key is not None}")
            logger.info(f"YouTube 객체 존재: {youtube_api.youtube is not None}")
            
            if not youtube_api.is_available():
                logger.warning("YouTube API를 사용할 수 없습니다.")
                return []
            
            recommendations = []
            
            for keyword in keywords:
                try:
                    logger.info(f"YouTube에서 '{keyword}' 검색 시작...")
                    videos = await youtube_api.search_videos(
                        query=keyword,
                        max_results=max_per_keyword,
                        order="relevance"
                    )
                    
                    for video in videos:
                        recommendations.append({
                            "title": video["title"],
                            "content_type": "youtube_video",
                            "description": video.get("description", "")[:200] + "...",
                            "source": video["video_url"],
                            "metadata": {
                                "channel": video.get("channel_title"),
                                "duration": video.get("duration"),
                                "view_count": video.get("view_count", 0),
                                "thumbnail": video.get("thumbnail_url")
                            },
                            "keyword": keyword,
                            "recommendation_source": "youtube_realtime"
                        })
                    
                    logger.info(f"YouTube에서 '{keyword}' 키워드로 {len(videos)}개 동영상 검색")
                    
                except Exception as e:
                    logger.warning(f"YouTube 검색 실패 (키워드: {keyword}): {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"YouTube 추천 검색 실패: {e}")
            return []

    async def _search_web_recommendations(
        self,
        keywords: List[str],
        content_types: List[str],
        max_items: int
    ) -> List[Dict]:
        """웹 검색 기반 실시간 추천"""
        try:
            recommendations = []
            max_per_type = max(1, max_items // len(content_types))
            
            for keyword in keywords[:3]:  # 최대 3개 키워드만 처리
                # 도서 추천
                if "book" in content_types:
                    books = await web_recommendation_engine.search_books(
                        keyword, max_results=max_per_type
                    )
                    recommendations.extend(books)
                
                # 영화 추천  
                if "movie" in content_types:
                    movies = await web_recommendation_engine.search_movies(
                        keyword, max_results=max_per_type
                    )
                    recommendations.extend(movies)
            
            logger.info(f"웹 검색에서 {len(recommendations)}개 추천 생성")
            return recommendations
            
        except Exception as e:
            logger.error(f"웹 검색 추천 실패: {e}")
            return []

    def _balance_recommendations(
        self,
        recommendations: List[Dict],
        keywords: List[str],
        max_items: int
    ) -> List[Dict]:
        """콘텐츠 타입별 및 키워드별로 추천을 균등하게 분배"""
        try:
            if not recommendations:
                return []
            
            # 1. 콘텐츠 타입별로 그룹화
            content_type_groups = {}
            for rec in recommendations:
                content_type = rec.get("content_type", "기타")
                if content_type not in content_type_groups:
                    content_type_groups[content_type] = []
                content_type_groups[content_type].append(rec)
            
            # 2. 각 콘텐츠 타입에서 최대 허용 개수 계산
            num_content_types = len(content_type_groups)
            max_per_content_type = max(2, max_items // num_content_types)  # 최소 2개씩 보장
            
            balanced = []
            
            # 3. 각 콘텐츠 타입에서 균등하게 선택
            for content_type, group in content_type_groups.items():
                # 해당 콘텐츠 타입 내에서 키워드별로 균등분배
                keyword_groups = {}
                for rec in group:
                    keyword = rec.get("keyword", "기타")
                    if keyword not in keyword_groups:
                        keyword_groups[keyword] = []
                    keyword_groups[keyword].append(rec)
                
                # 키워드별로 균등하게 선택
                content_type_items = []
                items_per_keyword = max(1, max_per_content_type // len(keyword_groups))
                
                for keyword, keyword_group in keyword_groups.items():
                    selected = keyword_group[:items_per_keyword]
                    content_type_items.extend(selected)
                
                # 콘텐츠 타입별 최대 개수로 제한
                content_type_items = content_type_items[:max_per_content_type]
                balanced.extend(content_type_items)
                
                logger.info(f"콘텐츠 타입 '{content_type}': {len(content_type_items)}개 선택")
            
            # 4. 남은 자리가 있으면 추가 선택 (다양성 우선)
            remaining = max_items - len(balanced)
            if remaining > 0:
                remaining_items = [rec for rec in recommendations if rec not in balanced]
                # 콘텐츠 타입별로 순서대로 추가 (다양성 보장)
                type_rotation = list(content_type_groups.keys())
                type_index = 0
                
                for item in remaining_items:
                    if len(balanced) >= max_items:
                        break
                    
                    # 현재 타입이 이미 많이 포함되었는지 확인
                    current_type = item.get("content_type")
                    current_type_count = sum(1 for b in balanced if b.get("content_type") == current_type)
                    
                    # 타입별 최대 개수를 넘지 않도록 제한
                    if current_type_count < max_per_content_type + 1:  # +1 여유
                        balanced.append(item)
            
            logger.info(f"균형 조정 완료: 총 {len(balanced)}개 (타입별 균등분배)")
            
            # 5. 콘텐츠 타입별 분포 로깅
            final_distribution = {}
            for item in balanced:
                content_type = item.get("content_type", "기타")
                final_distribution[content_type] = final_distribution.get(content_type, 0) + 1
            
            logger.info(f"최종 콘텐츠 타입 분포: {final_distribution}")
            
            return balanced[:max_items]
            
        except Exception as e:
            logger.error(f"추천 균등 분배 실패: {e}")
            return recommendations[:max_items]

    def _generate_fallback_recommendations(self, keywords: List[str]) -> List[Dict]:
        """폴백 추천 생성 (검색 결과가 없을 때)"""
        fallback_recommendations = []
        
        for keyword in keywords[:5]:  # 최대 5개 키워드만
            fallback_recommendations.extend([
                {
                    "title": f"{keyword} 관련 도서 추천",
                    "content_type": "book",
                    "description": f"{keyword}에 대해 더 자세히 알 수 있는 도서를 찾아보세요.",
                    "source": "추천 시스템",
                    "metadata": {},
                    "keyword": keyword,
                    "recommendation_source": "fallback"
                }
            ])
        
        return fallback_recommendations

    async def extract_keywords_from_file(
        self,
        file_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        max_keywords: int = 5
    ) -> List[str]:
        """
        업로드된 파일이나 폴더에서 키워드를 자동 추출
        
        Args:
            file_id: 특정 파일 ID
            folder_id: 폴더 ID (해당 폴더의 모든 파일에서 키워드 추출)
            max_keywords: 추출할 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        try:
            # TextCollector를 사용하여 텍스트 수집
            combined_text = ""
            if file_id:
                combined_text = await TextCollector.get_text_from_file(
                    self.db, file_id, use_chunks=True
                )
            elif folder_id:
                combined_text = await TextCollector.get_text_from_folder(
                    self.db, folder_id, use_chunks=True
                )
            
            if not combined_text.strip():
                logger.warning(f"텍스트를 찾을 수 없음: file_id={file_id}, folder_id={folder_id}")
                return []
            
            # 텍스트가 너무 긴 경우 제한
            if len(combined_text) > 5000:
                combined_text = combined_text[:5000] + "..."
                logger.info("텍스트가 너무 길어 5000자로 제한했습니다.")
            
            # AutoLabeler를 사용하여 키워드 추출
            labeler = AutoLabeler()
            keywords = await labeler.extract_keywords(
                text=combined_text,
                max_keywords=max_keywords
            )
            
            logger.info(f"키워드 추출 완료: {len(keywords)}개 - {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
