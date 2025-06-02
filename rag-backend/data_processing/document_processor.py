"""
문서 처리 파이프라인
텍스트 추출 → 청킹 → 벡터화 → DB 저장
MODIFIED 2024-12-19: 새로운 통합 문서 처리 파이프라인 구현
"""
from typing import Dict, List
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime

from .loader import DocumentLoader
from .chunker import TextChunker
from .embedder import TextEmbedder
from .preprocessor import TextPreprocessor
from database.operations import DatabaseOperations
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """문서 처리 파이프라인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.db_ops = DatabaseOperations(db)
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()
        self.preprocessor = TextPreprocessor()
    
    async def process_and_store(self, file_path: Path, file_metadata: Dict) -> Dict:
        """전체 문서 처리 및 저장 파이프라인"""
        try:
            logger.info(f"문서 처리 시작: {file_path}")
            
            # 1. 문서 로드 및 텍스트 추출
            document_data = await self.loader.load_document(file_path)
            raw_text = document_data["text"]
            
            logger.info(f"텍스트 추출 완료: {len(raw_text)} 문자")
            
            # 2. 텍스트 전처리
            processed_text = await self.preprocessor.preprocess(raw_text)
            
            # 3. 문서 메타데이터 저장
            document_record = {
                **file_metadata,
                "raw_text": raw_text,
                "processed_text": processed_text,
                "text_length": len(processed_text),
                "processing_status": "processing",
                "processing_time": datetime.utcnow()
            }
            
            document_id = await self.db_ops.insert_one("documents", document_record)
            logger.info(f"문서 메타데이터 저장 완료: {document_id}")
            
            # 4. 텍스트 청킹
            chunk_metadata = {
                "file_id": file_metadata["file_id"],
                "document_id": document_id,
                "source": str(file_path),
                "file_type": file_metadata["file_type"]
            }
            
            chunks = self.chunker.chunk_text(processed_text, chunk_metadata)
            logger.info(f"청킹 완료: {len(chunks)}개 청크")
            
            # 5. 청크 임베딩 생성
            embedded_chunks = await self.embedder.embed_documents(chunks)
            
            # 6. 청크 저장 준비
            chunk_records = []
            for i, chunk in enumerate(embedded_chunks):
                chunk_record = {
                    "file_id": file_metadata["file_id"],
                    "document_id": document_id,
                    "chunk_id": f"{file_metadata['file_id']}_chunk_{i}",
                    "sequence": chunk["sequence"],
                    "text": chunk["text"],
                    "text_embedding": chunk["text_embedding"],
                    "metadata": chunk["metadata"],
                    "created_at": datetime.utcnow()
                }
                chunk_records.append(chunk_record)
            
            # 7. 청크 배치 저장
            chunk_ids = await self.db_ops.insert_many("chunks", chunk_records)
            logger.info(f"청크 저장 완료: {len(chunk_ids)}개")
            
            # 8. 문서 처리 상태 업데이트
            await self.db_ops.update_one(
                "documents",
                {"_id": document_id},
                {
                    "$set": {
                        "processing_status": "completed",
                        "chunks_count": len(chunks),
                        "completed_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"문서 처리 완료: {file_metadata['file_id']}")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_count": len(chunks),
                "text_length": len(processed_text),
                "processing_time": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"문서 처리 실패: {e}")
            
            # 실패 시 상태 업데이트
            if 'document_id' in locals():
                await self.db_ops.update_one(
                    "documents",
                    {"_id": document_id},
                    {
                        "$set": {
                            "processing_status": "failed",
                            "error_message": str(e),
                            "failed_at": datetime.utcnow()
                        }
                    }
                )
            
            raise
    
    async def reprocess_document(self, file_id: str) -> Dict:
        """문서 재처리"""
        try:
            # 기존 문서 정보 조회
            document = await self.db_ops.find_one("documents", {"file_id": file_id})
            if not document:
                raise ValueError(f"문서를 찾을 수 없습니다: {file_id}")
            
            # 기존 청크들 삭제
            await self.db.chunks.delete_many({"file_id": file_id})
            
            # 재처리를 위한 메타데이터 준비
            file_metadata = {
                "file_id": file_id,
                "original_filename": document["original_filename"],
                "file_type": document["file_type"],
                "file_size": document["file_size"],
                "folder_id": document.get("folder_id"),
                "description": document.get("description")
            }
            
            # 기존 텍스트로 재처리 (파일 다시 로드하지 않음)
            processed_text = document.get("processed_text", document.get("raw_text", ""))
            
            # 청킹부터 재시작
            chunk_metadata = {
                "file_id": file_id,
                "document_id": str(document["_id"]),
                "source": document.get("source", "reprocessed"),
                "file_type": document["file_type"]
            }
            
            chunks = self.chunker.chunk_text(processed_text, chunk_metadata)
            embedded_chunks = await self.embedder.embed_documents(chunks)
            
            # 청크 저장
            chunk_records = []
            for i, chunk in enumerate(embedded_chunks):
                chunk_record = {
                    "file_id": file_id,
                    "document_id": str(document["_id"]),
                    "chunk_id": f"{file_id}_chunk_{i}",
                    "sequence": chunk["sequence"],
                    "text": chunk["text"],
                    "text_embedding": chunk["text_embedding"],
                    "metadata": chunk["metadata"],
                    "created_at": datetime.utcnow()
                }
                chunk_records.append(chunk_record)
            
            chunk_ids = await self.db_ops.insert_many("chunks", chunk_records)
            
            # 문서 상태 업데이트
            await self.db_ops.update_one(
                "documents",
                {"file_id": file_id},
                {
                    "$set": {
                        "processing_status": "completed",
                        "chunks_count": len(chunks),
                        "reprocessed_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"문서 재처리 완료: {file_id}")
            
            return {
                "success": True,
                "file_id": file_id,
                "chunks_count": len(chunks),
                "reprocessed_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"문서 재처리 실패: {e}")
            raise
    
    async def get_document_stats(self, file_id: str) -> Dict:
        """문서 처리 통계 조회"""
        try:
            # 문서 정보
            document = await self.db_ops.find_one("documents", {"file_id": file_id})
            if not document:
                raise ValueError(f"문서를 찾을 수 없습니다: {file_id}")
            
            # 청크 통계
            chunks_count = await self.db.chunks.count_documents({"file_id": file_id})
            
            # 평균 청크 크기 계산
            pipeline = [
                {"$match": {"file_id": file_id}},
                {"$project": {"text_length": {"$strLenCP": "$text"}}},
                {"$group": {
                    "_id": None,
                    "avg_length": {"$avg": "$text_length"},
                    "min_length": {"$min": "$text_length"},
                    "max_length": {"$max": "$text_length"}
                }}
            ]
            
            chunk_stats = await self.db.chunks.aggregate(pipeline).to_list(1)
            
            stats = {
                "file_id": file_id,
                "original_filename": document["original_filename"],
                "file_type": document["file_type"],
                "file_size": document["file_size"],
                "text_length": document.get("text_length", 0),
                "processing_status": document.get("processing_status", "unknown"),
                "chunks_count": chunks_count,
                "upload_time": document.get("upload_time"),
                "processing_time": document.get("processing_time"),
                "completed_at": document.get("completed_at")
            }
            
            if chunk_stats:
                stats.update({
                    "avg_chunk_length": chunk_stats[0]["avg_length"],
                    "min_chunk_length": chunk_stats[0]["min_length"],
                    "max_chunk_length": chunk_stats[0]["max_length"]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"문서 통계 조회 실패: {e}")
            raise 