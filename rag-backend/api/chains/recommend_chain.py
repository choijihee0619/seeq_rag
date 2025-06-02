"""
추천 체인
콘텐츠 추천 생성
MODIFIED 2024-01-20: YouTube API 연동 추가 - 실시간 YouTube 동영상 추천 기능 통합
ENHANCED 2024-01-21: 파일 기반 키워드 자동 추출 기능 추가
CLEANED 2024-01-21: 불필요한 YouTube 개별 API 제거, 핵심 추천 기능만 유지
"""
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger
from utils.youtube_api import youtube_api

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
            # 1. 텍스트 수집
            texts = []
            
            if file_id:
                # 특정 파일의 텍스트 수집
                file_texts = await self._get_file_texts(file_id)
                texts.extend(file_texts)
            elif folder_id:
                # 폴더 내 모든 파일의 텍스트 수집
                folder_texts = await self._get_folder_texts(folder_id)
                texts.extend(folder_texts)
            
            if not texts:
                logger.warning(f"텍스트를 찾을 수 없음: file_id={file_id}, folder_id={folder_id}")
                return []
            
            # 2. 텍스트 결합
            combined_text = " ".join(texts)
            
            # 3. OpenAI를 사용한 키워드 추출
            keywords = await self._extract_keywords_with_ai(combined_text, max_keywords)
            
            logger.info(f"키워드 추출 완료: {len(keywords)}개 - {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []

    async def _get_file_texts(self, file_id: str) -> List[str]:
        """특정 파일의 모든 텍스트 가져오기"""
        texts = []
        
        # chunks 컬렉션에서 해당 파일의 청크들 조회
        chunks = await self.chunks.find({"file_id": file_id}).to_list(None)
        
        if chunks:
            # 청크가 있으면 청크 텍스트 사용
            for chunk in chunks:
                if chunk.get("text"):
                    texts.append(chunk["text"])
        else:
            # 청크가 없으면 documents 컬렉션에서 원본 텍스트 조회
            docs = await self.documents.find({"file_id": file_id}).to_list(None)
            for doc in docs:
                if doc.get("raw_text"):
                    texts.append(doc["raw_text"])
                elif doc.get("processed_text"):
                    texts.append(doc["processed_text"])
        
        return texts

    async def _get_folder_texts(self, folder_id: str) -> List[str]:
        """폴더 내 모든 파일의 텍스트 가져오기"""
        texts = []
        
        # chunks 컬렉션에서 해당 폴더의 모든 청크 조회
        chunks = await self.chunks.find({"metadata.folder_id": folder_id}).to_list(None)
        
        if chunks:
            # 청크가 있으면 청크 텍스트 사용
            for chunk in chunks:
                if chunk.get("text"):
                    texts.append(chunk["text"])
        else:
            # 청크가 없으면 documents 컬렉션에서 조회
            docs = await self.documents.find({"folder_id": folder_id}).to_list(None)
            for doc in docs:
                if doc.get("raw_text"):
                    texts.append(doc["raw_text"])
                elif doc.get("processed_text"):
                    texts.append(doc["processed_text"])
        
        return texts

    async def _extract_keywords_with_ai(self, text: str, max_keywords: int) -> List[str]:
        """
        OpenAI를 사용하여 텍스트에서 키워드 추출
        
        Args:
            text: 키워드를 추출할 텍스트
            max_keywords: 추출할 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        try:
            import openai
            from config.settings import get_settings
            
            settings = get_settings()
            
            # 텍스트가 너무 길면 앞부분만 사용 (토큰 제한)
            if len(text) > 3000:
                text = text[:3000] + "..."
            
            # OpenAI에 키워드 추출 요청
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""당신은 텍스트에서 핵심 키워드를 추출하는 전문가입니다.
주어진 텍스트를 분석하여 가장 중요하고 의미있는 키워드 {max_keywords}개를 추출해주세요.

규칙:
1. 단일 단어나 간단한 구문으로 추출
2. 너무 일반적인 단어는 제외 (예: 것, 하다, 있다)
3. 고유명사, 전문용어, 주제어 우선
4. 한국어로 응답
5. 키워드만 쉼표로 구분해서 나열

예시: 인공지능, 머신러닝, 데이터 분석, 딥러닝, 자연어처리"""
                    },
                    {
                        "role": "user",
                        "content": f"다음 텍스트에서 핵심 키워드 {max_keywords}개를 추출해주세요:\n\n{text}"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            
            # 쉼표로 분리하고 정리
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            keywords = keywords[:max_keywords]  # 최대 개수 제한
            
            if not keywords:
                # AI 추출 실패시 간단한 방법으로 폴백
                logger.warning("AI 키워드 추출 실패, 간단한 방법으로 폴백")
                keywords = self._extract_keywords_simple(text, max_keywords)
            
            return keywords
            
        except Exception as e:
            logger.error(f"AI 키워드 추출 실패: {e}")
            # 폴백: 간단한 키워드 추출
            return self._extract_keywords_simple(text, max_keywords)

    def _extract_keywords_simple(self, text: str, max_keywords: int) -> List[str]:
        """
        간단한 키워드 추출 (AI 실패시 폴백)
        
        Args:
            text: 키워드를 추출할 텍스트
            max_keywords: 추출할 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        try:
            import re
            from collections import Counter
            
            # 한국어 단어 추출 (2글자 이상)
            korean_words = re.findall(r'[가-힣]{2,}', text)
            
            # 불용어 제거
            stopwords = {'것이', '하는', '있는', '되는', '같은', '통해', '위해', '대한', '관련', '경우', '때문', '따라'}
            filtered_words = [word for word in korean_words if word not in stopwords]
            
            # 빈도 계산
            word_counts = Counter(filtered_words)
            
            # 상위 키워드 선택
            keywords = [word for word, count in word_counts.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            logger.error(f"간단한 키워드 추출도 실패: {e}")
            return []
