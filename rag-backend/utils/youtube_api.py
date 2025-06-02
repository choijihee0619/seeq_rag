"""
YouTube API 연동 모듈
MODIFIED 2024-01-20: YouTube 동영상 검색 및 메타데이터 수집 기능 추가
"""
import os
import asyncio
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)

class YouTubeAPI:
    """YouTube API 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        YouTube API 초기화
        
        Args:
            api_key: YouTube Data API v3 키
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            logger.warning("YouTube API 키가 설정되지 않았습니다. 환경변수 YOUTUBE_API_KEY를 확인하세요.")
        
        self.youtube = None
        if self.api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            except Exception as e:
                logger.error(f"YouTube API 초기화 실패: {e}")

    async def search_videos(
        self,
        query: str,
        max_results: int = 10,
        order: str = "relevance",
        video_duration: str = "any",
        video_category_id: Optional[str] = None
    ) -> List[Dict]:
        """
        키워드로 YouTube 동영상 검색
        
        Args:
            query: 검색 키워드
            max_results: 최대 결과 수 (1-50)
            order: 정렬 방식 (relevance, date, rating, viewCount, title)
            video_duration: 동영상 길이 (any, short, medium, long)
            video_category_id: 카테고리 ID (선택)
            
        Returns:
            검색 결과 리스트
        """
        if not self.youtube:
            logger.error("YouTube API가 초기화되지 않았습니다.")
            return []
        
        try:
            # 동영상 검색 실행
            search_request = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=min(max_results, 50),  # API 제한
                order=order,
                type='video',
                videoDuration=video_duration,
                regionCode='KR',  # 한국 지역 설정
                relevanceLanguage='ko'  # 한국어 우선
            )
            
            # 카테고리 필터 추가
            if video_category_id:
                search_request = search_request.execute()
                # 상세 정보에서 카테고리 필터링 필요
            else:
                search_request = search_request.execute()
            
            videos = []
            video_ids = []
            
            # 비디오 ID 수집
            for item in search_request.get('items', []):
                video_ids.append(item['id']['videoId'])
            
            # 상세 정보 가져오기 (통계, 지속시간 등)
            if video_ids:
                videos_details = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids)
                ).execute()
                
                # 결과 정리
                for item in videos_details.get('items', []):
                    video_info = await self._parse_video_info(item)
                    if video_info:
                        videos.append(video_info)
            
            logger.info(f"YouTube 검색 완료: '{query}' - {len(videos)}개 결과")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API 요청 실패: {e}")
            return []
        except Exception as e:
            logger.error(f"YouTube 검색 중 오류: {e}")
            return []

    async def _parse_video_info(self, item: Dict) -> Optional[Dict]:
        """YouTube API 응답을 파싱하여 정리된 동영상 정보 반환"""
        try:
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            content_details = item.get('contentDetails', {})
            
            # 썸네일 URL 선택 (고화질 우선)
            thumbnails = snippet.get('thumbnails', {})
            thumbnail_url = (
                thumbnails.get('maxres', {}).get('url') or
                thumbnails.get('high', {}).get('url') or
                thumbnails.get('medium', {}).get('url') or
                thumbnails.get('default', {}).get('url')
            )
            
            # 조회수 숫자 변환
            view_count = statistics.get('viewCount', '0')
            try:
                view_count = int(view_count)
            except (ValueError, TypeError):
                view_count = 0
            
            # 좋아요 수 변환
            like_count = statistics.get('likeCount', '0')
            try:
                like_count = int(like_count)
            except (ValueError, TypeError):
                like_count = 0
            
            # 동영상 길이 파싱 (ISO 8601 duration)
            duration = self._parse_duration(content_details.get('duration', ''))
            
            return {
                'video_id': item['id'],
                'title': snippet.get('title', '제목 없음'),
                'description': snippet.get('description', '')[:500],  # 설명 500자 제한
                'channel_title': snippet.get('channelTitle', ''),
                'channel_id': snippet.get('channelId', ''),
                'published_at': snippet.get('publishedAt', ''),
                'thumbnail_url': thumbnail_url,
                'view_count': view_count,
                'like_count': like_count,
                'duration': duration,
                'duration_seconds': self._duration_to_seconds(duration),
                'video_url': f"https://www.youtube.com/watch?v={item['id']}",
                'tags': snippet.get('tags', [])[:10],  # 태그 10개 제한
                'category_id': snippet.get('categoryId', ''),
                'default_language': snippet.get('defaultLanguage', ''),
                'content_type': 'youtube_video'
            }
            
        except Exception as e:
            logger.error(f"동영상 정보 파싱 실패: {e}")
            return None

    def _parse_duration(self, duration: str) -> str:
        """ISO 8601 duration을 읽기 쉬운 형식으로 변환"""
        if not duration or not duration.startswith('PT'):
            return "0:00"
        
        import re
        # PT15M33S 형식 파싱
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        
        if not match:
            return "0:00"
        
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def _duration_to_seconds(self, duration: str) -> int:
        """duration을 총 초 수로 변환"""
        if not duration or duration == "0:00":
            return 0
        
        parts = duration.split(':')
        try:
            if len(parts) == 3:  # H:M:S
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:  # M:S
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return 0
        except (ValueError, IndexError):
            return 0

    async def get_video_categories(self, region_code: str = 'KR') -> Dict[str, str]:
        """
        YouTube 동영상 카테고리 목록 가져오기
        
        Args:
            region_code: 지역 코드 (기본: KR)
            
        Returns:
            카테고리 ID와 이름 매핑
        """
        if not self.youtube:
            return {}
        
        try:
            response = self.youtube.videoCategories().list(
                part='snippet',
                regionCode=region_code
            ).execute()
            
            categories = {}
            for item in response.get('items', []):
                categories[item['id']] = item['snippet']['title']
            
            return categories
            
        except Exception as e:
            logger.error(f"카테고리 조회 실패: {e}")
            return {}

    async def search_by_channel(
        self,
        channel_id: str,
        max_results: int = 10,
        order: str = "date"
    ) -> List[Dict]:
        """
        특정 채널의 동영상 검색
        
        Args:
            channel_id: YouTube 채널 ID
            max_results: 최대 결과 수
            order: 정렬 방식
            
        Returns:
            채널 동영상 리스트
        """
        if not self.youtube:
            return []
        
        try:
            search_request = self.youtube.search().list(
                channelId=channel_id,
                part='id,snippet',
                maxResults=min(max_results, 50),
                order=order,
                type='video'
            ).execute()
            
            videos = []
            for item in search_request.get('items', []):
                video_info = await self._parse_video_info(item)
                if video_info:
                    videos.append(video_info)
            
            return videos
            
        except Exception as e:
            logger.error(f"채널 검색 실패: {e}")
            return []

# 싱글톤 인스턴스
youtube_api = YouTubeAPI() 