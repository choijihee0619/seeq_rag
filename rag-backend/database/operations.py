"""
데이터베이스 작업 모듈
공통 CRUD 작업
"""
from typing import Dict, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseOperations:
    """데이터베이스 작업 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    async def insert_one(
        self,
        collection_name: str,
        document: Dict
    ) -> str:
        """단일 문서 삽입"""
        try:
            # 타임스탬프 추가
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.insert_one(document)
            
            logger.info(f"{collection_name}에 문서 삽입: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"문서 삽입 실패: {e}")
            raise
    
    async def insert_many(
        self,
        collection_name: str,
        documents: List[Dict]
    ) -> List[str]:
        """다중 문서 삽입"""
        try:
            # 타임스탬프 추가
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.insert_many(documents)
            
            logger.info(f"{collection_name}에 {len(result.inserted_ids)}개 문서 삽입")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"다중 문서 삽입 실패: {e}")
            raise
    
    async def find_one(
        self,
        collection_name: str,
        filter_dict: Dict
    ) -> Optional[Dict]:
        """단일 문서 조회"""
        try:
            collection = self.db[collection_name]
            document = await collection.find_one(filter_dict)
            return document
            
        except Exception as e:
            logger.error(f"문서 조회 실패: {e}")
            raise
    
    async def find_many(
        self,
        collection_name: str,
        filter_dict: Dict,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict]:
        """다중 문서 조회"""
        try:
            collection = self.db[collection_name]
            cursor = collection.find(filter_dict)
            
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(None)
            return documents
            
        except Exception as e:
            logger.error(f"다중 문서 조회 실패: {e}")
            raise
    
    async def update_one(
        self,
        collection_name: str,
        filter_dict: Dict,
        update_dict: Dict
    ) -> bool:
        """단일 문서 업데이트"""
        try:
            # 업데이트 타임스탬프 추가
            update_dict["$set"] = update_dict.get("$set", {})
            update_dict["$set"]["updated_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.update_one(filter_dict, update_dict)
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"문서 업데이트 실패: {e}")
            raise
    
    async def delete_one(
        self,
        collection_name: str,
        filter_dict: Dict
    ) -> bool:
        """단일 문서 삭제"""
        try:
            collection = self.db[collection_name]
            result = await collection.delete_one(filter_dict)
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            raise
