"""
데이터베이스 연결 모듈
MongoDB 연결 관리
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseConnection:
    """데이터베이스 연결 클래스"""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db: AsyncIOMotorDatabase = None
    
    async def connect(self):
        """데이터베이스 연결"""
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URI)
            self.db = self.client[settings.MONGODB_DB_NAME]
            
            # 연결 테스트
            await self.client.server_info()
            logger.info("MongoDB 연결 성공")
            
            # 인덱스 생성
            await self.create_indexes()
            
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            raise
    
    async def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB 연결 해제")
    
    async def create_indexes(self):
        """필요한 인덱스 생성"""
        try:
            # documents 컬렉션 인덱스
            await self.db.documents.create_index("folder_id")
            await self.db.documents.create_index("chunk_id")
            await self.db.documents.create_index([("text", "text")])
            
            # labels 컬렉션 인덱스
            await self.db.labels.create_index("document_id")
            await self.db.labels.create_index("folder_id")
            await self.db.labels.create_index("tags")
            await self.db.labels.create_index("category")
            
            # qapairs 컬렉션 인덱스
            await self.db.qapairs.create_index("document_id")
            await self.db.qapairs.create_index("folder_id")
            await self.db.qapairs.create_index("difficulty")
            
            # recommendations 컬렉션 인덱스
            await self.db.recommendations.create_index("keyword")
            await self.db.recommendations.create_index("content_type")
            
            logger.info("인덱스 생성 완료")
            
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")

# 싱글톤 인스턴스
db_connection = DatabaseConnection()

async def init_db():
    """데이터베이스 초기화"""
    await db_connection.connect()

async def close_db():
    """데이터베이스 종료"""
    await db_connection.disconnect()

async def get_database() -> AsyncIOMotorDatabase:
    """데이터베이스 인스턴스 반환"""
    if db_connection.db is None:
        await init_db()
    return db_connection.db
