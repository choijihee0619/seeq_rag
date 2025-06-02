"""
파일 업로드 API 라우터
PDF, DOCX, TXT, HTML 파일 업로드 및 처리
"""
import os
import uuid
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import aiofiles
from datetime import datetime

from data_processing.document_processor import DocumentProcessor
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 업로드 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

class UploadResponse(BaseModel):
    """업로드 응답 모델"""
    success: bool
    message: str
    file_id: str
    original_filename: str
    processed_chunks: int
    storage_path: Optional[str] = None

class FileStatus(BaseModel):
    """파일 상태 모델"""
    file_id: str
    original_filename: str
    file_type: str
    file_size: int
    status: str  # 'uploading', 'processing', 'completed', 'failed'
    processed_chunks: int
    upload_time: datetime
    folder_id: Optional[str] = None

@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """파일 업로드 및 처리"""
    try:
        # 파일 검증
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다.")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # 파일 크기 검증
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기가 너무 큽니다. (최대 50MB)")
        
        # 고유 파일 ID 생성
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        # 파일 임시 저장
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"파일 저장 완료: {file_path}")
        
        # 문서 처리
        db = await get_database()
        processor = DocumentProcessor(db)
        
        # 파일 정보 메타데이터
        file_metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "file_type": file_ext[1:],  # 확장자에서 . 제거
            "file_size": len(content),
            "folder_id": folder_id,
            "description": description,
            "upload_time": datetime.utcnow()
        }
        
        # 문서 처리 및 DB 저장
        result = await processor.process_and_store(file_path, file_metadata)
        
        # 임시 파일 삭제
        if file_path.exists():
            os.remove(file_path)
        
        return UploadResponse(
            success=True,
            message="파일 업로드 및 처리가 완료되었습니다.",
            file_id=file_id,
            original_filename=file.filename,
            processed_chunks=result["chunks_count"],
            storage_path=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 실패: {e}")
        # 실패 시 임시 파일 정리
        if 'file_path' in locals() and file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/status/{file_id}", response_model=FileStatus)
async def get_file_status(file_id: str):
    """파일 처리 상태 조회"""
    try:
        db = await get_database()
        
        # documents 컬렉션에서 파일 정보 조회
        document = await db.documents.find_one({"file_id": file_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # chunks 컬렉션에서 청크 개수 조회
        chunks_count = await db.chunks.count_documents({"file_id": file_id})
        
        return FileStatus(
            file_id=file_id,
            original_filename=document["original_filename"],
            file_type=document["file_type"],
            file_size=document["file_size"],
            status="completed",  # 현재는 단순화
            processed_chunks=chunks_count,
            upload_time=document["upload_time"],
            folder_id=document.get("folder_id")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """업로드된 파일 및 관련 데이터 삭제"""
    try:
        db = await get_database()
        
        # 문서 정보 확인
        document = await db.documents.find_one({"file_id": file_id})
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 관련 청크들 삭제
        chunks_result = await db.chunks.delete_many({"file_id": file_id})
        
        # 문서 정보 삭제
        doc_result = await db.documents.delete_one({"file_id": file_id})
        
        logger.info(f"파일 삭제 완료: {file_id}, 청크 {chunks_result.deleted_count}개, 문서 {doc_result.deleted_count}개")
        
        return {
            "success": True,
            "message": "파일이 삭제되었습니다.",
            "deleted_chunks": chunks_result.deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_files(folder_id: Optional[str] = None, limit: int = 50, skip: int = 0):
    """업로드된 파일 목록 조회"""
    try:
        db = await get_database()
        
        # 필터 조건
        filter_dict = {}
        if folder_id:
            filter_dict["folder_id"] = folder_id
        
        # 문서 목록 조회
        cursor = db.documents.find(filter_dict).sort("upload_time", -1).skip(skip).limit(limit)
        documents = await cursor.to_list(None)
        
        # 각 문서의 청크 개수 조회
        result = []
        for doc in documents:
            chunks_count = await db.chunks.count_documents({"file_id": doc["file_id"]})
            
            result.append({
                "file_id": doc["file_id"],
                "original_filename": doc["original_filename"],
                "file_type": doc["file_type"],
                "file_size": doc["file_size"],
                "processed_chunks": chunks_count,
                "upload_time": doc["upload_time"],
                "folder_id": doc.get("folder_id"),
                "description": doc.get("description")
            })
        
        return {
            "files": result,
            "total": len(result),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"파일 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 