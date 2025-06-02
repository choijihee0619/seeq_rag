"""
하이브리드 검색 모듈
벡터 검색과 키워드 검색을 결합
"""
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from retrieval.vector_search import VectorSearch
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridSearch:
    """하이브리드 검색 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.vector_search = VectorSearch(db)
        self.documents = db.documents
        self.labels = db.labels
    
    async def search(
        self,
        query: str,
        k: int = 5,
        folder_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """하이브리드 검색 실행"""
        # 필터 조건 생성
        filter_dict = {}
        if folder_id:
            filter_dict["folder_id"] = folder_id
        
        # 벡터 검색
        vector_results = await self.vector_search.search_similar(
            query, k=k*2, filter_dict=filter_dict
        )
        
        # 라벨 기반 필터링
        if categories or tags:
            filtered_results = []
            for result in vector_results:
                doc_id = str(result["document"]["_id"])
                
                # 라벨 조회
                label = await self.labels.find_one({"document_id": doc_id})
                
                if label:
                    # 카테고리 필터
                    if categories and label.get("category") not in categories:
                        continue
                    
                    # 태그 필터
                    if tags:
                        doc_tags = label.get("tags", [])
                        if not any(tag in doc_tags for tag in tags):
                            continue
                
                filtered_results.append(result)
            
            vector_results = filtered_results
        
        # 최종 결과 반환
        return vector_results[:k]
    
    async def search_by_keyword(
        self,
        keyword: str,
        k: int = 5
    ) -> List[Dict]:
        """키워드 기반 검색"""
        # 텍스트 검색
        text_filter = {"text": {"$regex": keyword, "$options": "i"}}
        
        documents = []
        cursor = self.documents.find(text_filter).limit(k)
        
        async for doc in cursor:
            documents.append({
                "document": doc,
                "score": 1.0  # 키워드 매칭은 동일한 점수
            })
        
        return documents
