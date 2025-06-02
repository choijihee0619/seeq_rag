"""
파일 업로드 API 라우터
MODIFIED 2024-01-20: 파일 raw text 조회 및 편집 기능 추가
ENHANCED 2024-01-21: 파일 미리보기 기능 추가
REFACTORED 2024-01-21: 중복 검색 API 제거 및 코드 정리
FIXED 2024-01-21: import 경로 수정 (file_processing -> data_processing)
FIXED 2025-06-03: DocumentProcessor 초기화 및 메서드 호출 오류 수정
"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database.connection import get_database
from data_processing.document_processor import DocumentProcessor
from retrieval.vector_search import VectorSearch
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

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

class FileSearchRequest(BaseModel):
    """파일 검색 요청 모델"""
    query: str
    search_type: str = "both"  # filename, content, both
    folder_id: Optional[str] = None
    limit: int = 20
    skip: int = 0

class FileSearchResult(BaseModel):
    """파일 검색 결과 모델"""
    file_id: str
    original_filename: str
    file_type: str
    file_size: int
    processed_chunks: int
    upload_time: datetime
    folder_id: Optional[str] = None
    description: Optional[str] = None
    match_type: str  # filename, content, both
    relevance_score: float
    matched_content: Optional[str] = None  # 검색어와 매칭된 내용 미리보기

class FileSearchResponse(BaseModel):
    """파일 검색 응답 모델"""
    files: List[FileSearchResult]
    total_found: int
    query: str
    search_type: str
    execution_time: float

class FileUpdateRequest(BaseModel):
    """파일 정보 업데이트 요청 모델"""
    filename: Optional[str] = None
    description: Optional[str] = None
    folder_id: Optional[str] = None

class FilePreviewResponse(BaseModel):
    """파일 미리보기 응답 모델"""
    file_id: str
    original_filename: str
    file_type: str
    preview_text: str
    preview_length: int
    total_length: int
    has_more: bool
    preview_type: str  # "text", "pdf_extract", "document_extract"

@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """파일 업로드 및 처리"""
    try:
        # 디버깅: 받은 폼 데이터 로그 출력
        logger.info(f"업로드 폼 데이터 - 파일명: {file.filename}, folder_id: '{folder_id}', description: '{description}'")
        
        # 파일 정보 검증
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다.")
        
        # 파일 타입 확인
        allowed_types = ['.txt', '.pdf', '.docx', '.doc', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_types)}"
            )
        
        # 파일 크기 확인 (10MB 제한)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")
        
        # 파일을 다시 처음으로 되돌리기
        await file.seek(0)
        
        # 데이터베이스 연결
        db = await get_database()
        
        # 파일 ID 생성
        file_id = str(uuid.uuid4())
        
        # 임시 파일로 저장
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        temp_filename = f"{file_id}_{file.filename}"
        temp_file_path = upload_dir / temp_filename
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Form 데이터 정리 - 빈 값이나 기본값 처리
        clean_folder_id = None
        clean_description = None
        
        if folder_id and folder_id.strip() and folder_id not in ["string", "null"]:
            clean_folder_id = folder_id.strip()
        
        if description and description.strip() and description not in ["string", "null"]:
            clean_description = description.strip()
        
        logger.info(f"정리된 데이터 - clean_folder_id: '{clean_folder_id}', clean_description: '{clean_description}'")
        
        # 파일 메타데이터 준비
        file_metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "file_type": file_ext[1:],  # 점 제거
            "file_size": len(file_content),
            "upload_time": datetime.utcnow(),
            "folder_id": clean_folder_id,
            "description": clean_description
        }
        
        # 문서 처리기로 파일 처리
        processor = DocumentProcessor(db)
        result = await processor.process_and_store(
            file_path=temp_file_path,
            file_metadata=file_metadata
        )
        
        # 임시 파일 삭제
        try:
            temp_file_path.unlink()
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")
        
        logger.info(f"파일 업로드 완료: {file.filename} -> {file_id}")
        
        return UploadResponse(
            success=True,
            message="파일 업로드가 완료되었습니다.",
            file_id=file_id,
            original_filename=file.filename,
            processed_chunks=result["chunks_count"],
            storage_path=str(temp_file_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

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

@router.post("/search", response_model=FileSearchResponse)
async def search_files(request: FileSearchRequest):
    """📍 자연어 파일 검색 - 파일명과 내용으로 검색 가능"""
    try:
        start_time = time.time()
        db = await get_database()
        
        # 1. 기본 필터 조건 설정 (folder_id가 실제 값일 때만 적용)
        base_filter = {}
        if request.folder_id and request.folder_id.strip() and request.folder_id != "string":
            base_filter["folder_id"] = request.folder_id
        
        found_files = []
        
        # 2. 파일명 검색 (filename 또는 both)
        if request.search_type in ["filename", "both"]:
            filename_filter = base_filter.copy()
            filename_filter["original_filename"] = {"$regex": request.query, "$options": "i"}
            
            filename_docs = await db.documents.find(filename_filter).to_list(None)
            
            for doc in filename_docs:
                chunks_count = await db.chunks.count_documents({"file_id": doc["file_id"]})
                
                found_files.append({
                    "file_id": doc["file_id"],
                    "original_filename": doc["original_filename"],
                    "file_type": doc["file_type"],
                    "file_size": doc["file_size"],
                    "processed_chunks": chunks_count,
                    "upload_time": doc["upload_time"],
                    "folder_id": doc.get("folder_id"),
                    "description": doc.get("description"),
                    "match_type": "filename",
                    "relevance_score": 1.0,  # 파일명 매치는 높은 점수
                    "matched_content": f"파일명 매치: {doc['original_filename']}"
                })
        
        # 3. 내용 검색 (content 또는 both)
        if request.search_type in ["content", "both"]:
            # folder_id 필터링을 위해 documents와 조인
            pipeline = [
                {
                    "$lookup": {
                        "from": "documents",
                        "localField": "file_id",
                        "foreignField": "file_id",
                        "as": "document_info"
                    }
                },
                {"$unwind": "$document_info"}
            ]
            
            # folder_id 필터 추가 (실제 값일 때만)
            if request.folder_id and request.folder_id.strip() and request.folder_id != "string":
                pipeline.append({
                    "$match": {"document_info.folder_id": request.folder_id}
                })
            
            # 텍스트 검색 추가
            pipeline.extend([
                {"$match": {"text": {"$regex": request.query, "$options": "i"}}},
                {"$limit": request.limit * 2}  # 더 많이 찾아서 나중에 파일별로 그룹화
            ])
            
            content_chunks = await db.chunks.aggregate(pipeline).to_list(None)
            
            # 파일별로 그룹화하고 최고 매칭 청크만 남기기
            file_matches = {}
            for chunk in content_chunks:
                file_id = chunk["file_id"]
                doc_info = chunk["document_info"]
                
                if file_id not in file_matches:
                    chunks_count = await db.chunks.count_documents({"file_id": file_id})
                    
                    # 매칭된 텍스트 추출 (간단한 하이라이트)
                    text = chunk["text"]
                    query_lower = request.query.lower()
                    text_lower = text.lower()
                    
                    # 검색어 주변 텍스트 추출
                    match_index = text_lower.find(query_lower)
                    if match_index != -1:
                        start = max(0, match_index - 50)
                        end = min(len(text), match_index + len(request.query) + 50)
                        matched_content = "..." + text[start:end] + "..."
                    else:
                        matched_content = text[:100] + "..."
                    
                    file_matches[file_id] = {
                        "file_id": file_id,
                        "original_filename": doc_info["original_filename"],
                        "file_type": doc_info["file_type"],
                        "file_size": doc_info["file_size"],
                        "processed_chunks": chunks_count,
                        "upload_time": doc_info["upload_time"],
                        "folder_id": doc_info.get("folder_id"),
                        "description": doc_info.get("description"),
                        "match_type": "content",
                        "relevance_score": 0.8,  # 내용 매치는 중간 점수
                        "matched_content": matched_content
                    }
            
            found_files.extend(file_matches.values())
        
        # 4. 중복 제거 및 정렬 (file_id 기준)
        unique_files = {}
        for file_data in found_files:
            file_id = file_data["file_id"]
            if file_id not in unique_files or file_data["relevance_score"] > unique_files[file_id]["relevance_score"]:
                unique_files[file_id] = file_data
        
        # 관련성 점수 순으로 정렬
        sorted_files = sorted(unique_files.values(), key=lambda x: x["relevance_score"], reverse=True)
        total_found = len(sorted_files)
        
        # 페이지네이션 적용
        paginated_files = sorted_files[request.skip:request.skip + request.limit]
        
        # 5. 응답 모델에 맞게 변환
        search_results = []
        for file_data in paginated_files:
            search_results.append(FileSearchResult(**file_data))
        
        execution_time = time.time() - start_time
        
        # 디버그 정보 로깅
        logger.info(f"검색 완료 - 쿼리: '{request.query}', 타입: {request.search_type}, 폴더: {request.folder_id}, 결과: {total_found}개")
        
        return FileSearchResponse(
            files=search_results,
            total_found=total_found,
            query=request.query,
            search_type=request.search_type,
            execution_time=round(execution_time, 3)
        )
        
    except Exception as e:
        logger.error(f"파일 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 검색 중 오류가 발생했습니다: {str(e)}")

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

@router.get("/semantic-search")
async def semantic_search_files(
    q: str,  # 검색어
    k: int = 5,  # 결과 개수
    folder_id: Optional[str] = None
):
    """🧠 AI 기반 의미 검색 - 벡터 유사도로 파일 찾기"""
    try:
        db = await get_database()
        vector_search = VectorSearch(db)
        
        # 필터 조건 설정
        filter_dict = {}
        if folder_id and folder_id.strip():  # None이거나 빈 문자열이 아닐 때만
            filter_dict["folder_id"] = folder_id
        
        # 벡터 검색 실행
        search_results = await vector_search.search_similar(
            query=q,
            k=k * 3,  # 더 많이 찾아서 파일별로 그룹화
            filter_dict=filter_dict
        )
        
        # 파일별로 그룹화하고 최고 점수만 남기기
        file_groups = {}
        for result in search_results:
            chunk = result.get("chunk", {})
            document = result.get("document", {})
            score = result.get("score", 0.0)
            file_id = chunk.get("file_id")
            
            if file_id and (file_id not in file_groups or score > file_groups[file_id]["relevance_score"]):
                # chunks 개수 조회
                chunks_count = await db.chunks.count_documents({"file_id": file_id})
                
                file_groups[file_id] = {
                    "file_id": file_id,
                    "original_filename": document.get("original_filename", "알 수 없는 파일"),
                    "file_type": document.get("file_type", "unknown"),
                    "file_size": document.get("file_size", 0),
                    "processed_chunks": chunks_count,
                    "upload_time": document.get("upload_time"),
                    "folder_id": document.get("folder_id"),
                    "description": document.get("description"),
                    "match_type": "semantic",
                    "relevance_score": score,
                    "matched_content": chunk.get("text", "")[:200] + "..."
                }
        
        # 점수 순으로 정렬하고 상위 k개만 반환
        sorted_files = sorted(file_groups.values(), key=lambda x: x["relevance_score"], reverse=True)
        top_files = sorted_files[:k]
        
        # 응답 생성
        search_results = [FileSearchResult(**file_data) for file_data in top_files]
        
        return FileSearchResponse(
            files=search_results,
            total_found=len(search_results),
            query=q,
            search_type="semantic",
            execution_time=0.0  # 실제 시간 측정은 생략
        )
        
    except Exception as e:
        logger.error(f"의미 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"의미 검색 중 오류가 발생했습니다: {str(e)}")

@router.get("/content/{file_id}")
async def get_file_content(file_id: str):
    """파일의 원본 텍스트 내용 조회 (토글 표시용)"""
    try:
        db = await get_database()
        
        # documents 컬렉션에서 파일 정보 및 텍스트 조회
        document = await db.documents.find_one(
            {"file_id": file_id},
            {"original_filename": 1, "raw_text": 1, "processed_text": 1, "file_type": 1, "upload_time": 1}
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        return {
            "file_id": file_id,
            "original_filename": document["original_filename"],
            "file_type": document["file_type"],
            "upload_time": document["upload_time"],
            "raw_text": document.get("raw_text", ""),
            "processed_text": document.get("processed_text", ""),
            "text_length": len(document.get("raw_text", ""))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 내용 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{file_id}")
async def update_file_info(file_id: str, request: FileUpdateRequest):
    """파일 정보 업데이트 (파일명, 설명, 폴더 등)"""
    try:
        db = await get_database()
        
        # 업데이트할 필드 준비
        update_fields = {}
        if request.filename is not None and request.filename.strip():
            update_fields["original_filename"] = request.filename.strip()
        if request.description is not None:
            update_fields["description"] = request.description.strip() if request.description.strip() else None
        if request.folder_id is not None:
            update_fields["folder_id"] = request.folder_id.strip() if request.folder_id.strip() else None
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="업데이트할 내용이 없습니다.")
        
        # 파일 존재 확인
        document = await db.documents.find_one({"file_id": file_id})
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 문서 업데이트
        result = await db.documents.update_one(
            {"file_id": file_id},
            {"$set": update_fields}
        )
        
        if result.modified_count == 0:
            logger.warning(f"파일 정보 업데이트 결과 없음: {file_id}")
        
        # 폴더 변경시 chunks의 metadata도 업데이트
        if "folder_id" in update_fields:
            await db.chunks.update_many(
                {"file_id": file_id},
                {"$set": {"metadata.folder_id": update_fields["folder_id"]}}
            )
        
        logger.info(f"파일 정보 업데이트 완료: {file_id} - {update_fields}")
        
        return {
            "success": True,
            "message": "파일 정보가 업데이트되었습니다.",
            "updated_fields": update_fields
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 정보 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview/{file_id}")
async def get_file_preview(file_id: str, max_length: int = 500):
    """파일 미리보기 - 처음 몇 줄의 텍스트를 반환"""
    try:
        db = await get_database()
        
        # documents 컬렉션에서 파일 정보 조회
        document = await db.documents.find_one(
            {"file_id": file_id},
            {"original_filename": 1, "raw_text": 1, "file_type": 1, "text_length": 1}
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        raw_text = document.get("raw_text", "")
        file_type = document.get("file_type", "unknown")
        total_length = len(raw_text)
        
        # 미리보기 텍스트 생성
        preview_text = ""
        preview_type = "text"
        
        if raw_text:
            # 텍스트를 줄 단위로 분할하여 자연스러운 미리보기 생성
            lines = raw_text.split('\n')
            current_length = 0
            preview_lines = []
            
            for line in lines:
                if current_length + len(line) > max_length:
                    break
                preview_lines.append(line)
                current_length += len(line) + 1  # +1 for newline
            
            preview_text = '\n'.join(preview_lines)
            
            # 파일 타입에 따른 미리보기 타입 결정
            if file_type == "pdf":
                preview_type = "pdf_extract"
            elif file_type in ["docx", "doc"]:
                preview_type = "document_extract"
            else:
                preview_type = "text"
        else:
            preview_text = "텍스트를 추출할 수 없습니다."
        
        preview_length = len(preview_text)
        has_more = total_length > preview_length
        
        return FilePreviewResponse(
            file_id=file_id,
            original_filename=document["original_filename"],
            file_type=file_type,
            preview_text=preview_text,
            preview_length=preview_length,
            total_length=total_length,
            has_more=has_more,
            preview_type=preview_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 미리보기 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview/chunks/{file_id}")
async def get_file_preview_with_chunks(file_id: str, chunk_count: int = 3):
    """청크 기반 파일 미리보기 - 처음 몇 개 청크의 내용"""
    try:
        db = await get_database()
        
        # 파일 기본 정보 조회
        document = await db.documents.find_one(
            {"file_id": file_id},
            {"original_filename": 1, "file_type": 1}
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 처음 몇 개 청크 조회
        chunks_cursor = db.chunks.find(
            {"file_id": file_id}
        ).sort("sequence", 1).limit(chunk_count)
        
        chunks = await chunks_cursor.to_list(None)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="파일의 청크를 찾을 수 없습니다.")
        
        # 청크들의 텍스트를 합쳐서 미리보기 생성
        preview_texts = [chunk["text"] for chunk in chunks]
        preview_text = "\n\n--- 다음 섹션 ---\n\n".join(preview_texts)
        
        # 전체 청크 수 조회
        total_chunks = await db.chunks.count_documents({"file_id": file_id})
        
        return {
            "file_id": file_id,
            "original_filename": document["original_filename"],
            "file_type": document["file_type"],
            "preview_text": preview_text,
            "preview_chunks": len(chunks),
            "total_chunks": total_chunks,
            "has_more": total_chunks > len(chunks),
            "preview_type": "chunks"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"청크 기반 미리보기 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 