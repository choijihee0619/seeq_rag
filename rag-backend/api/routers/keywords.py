"""
키워드 추출 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ai_processing.labeler import AutoLabeler
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class KeywordsRequest(BaseModel):
    """키워드 추출 요청 모델 - 직접 텍스트 입력"""
    text: str
    max_keywords: int = 10

class FileKeywordsRequest(BaseModel):
    """파일 기반 키워드 추출 요청 모델"""
    file_id: Optional[str] = None
    folder_id: Optional[str] = None
    max_keywords: int = 10
    use_chunks: bool = True  # True: 청크들에서 추출, False: 원본 문서에서 추출

class KeywordsResponse(BaseModel):
    """키워드 추출 응답 모델"""
    keywords: List[str]
    count: int
    source_info: Optional[dict] = None  # 소스 정보 (파일명, 청크 수 등)

@router.post("/", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest):
    """키워드 추출 엔드포인트 - 직접 텍스트 입력"""
    try:
        labeler = AutoLabeler()
        
        # 키워드 추출
        keywords = await labeler.extract_keywords(
            text=request.text,
            max_keywords=request.max_keywords
        )
        
        return KeywordsResponse(
            keywords=keywords,
            count=len(keywords)
        )
        
    except Exception as e:
        logger.error(f"키워드 추출 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/from-file", response_model=KeywordsResponse)
async def extract_keywords_from_file(request: FileKeywordsRequest):
    """파일에서 키워드 추출 엔드포인트"""
    try:
        db = await get_database()
        labeler = AutoLabeler()
        
        combined_text = ""
        source_info = {}
        
        if request.file_id:
            # 특정 파일에서 키워드 추출
            if request.use_chunks:
                # 청크들에서 텍스트 수집
                chunks = await db.chunks.find({"file_id": request.file_id}).sort("sequence", 1).to_list(None)
                if not chunks:
                    raise HTTPException(status_code=404, detail="해당 파일의 청크를 찾을 수 없습니다.")
                
                combined_text = "\n\n".join([chunk["text"] for chunk in chunks])
                source_info = {
                    "file_id": request.file_id,
                    "source_type": "chunks",
                    "chunk_count": len(chunks)
                }
            else:
                # 원본 문서에서 텍스트 가져오기
                document = await db.documents.find_one({"file_id": request.file_id})
                if not document:
                    raise HTTPException(status_code=404, detail="해당 파일을 찾을 수 없습니다.")
                
                combined_text = document.get("processed_text", document.get("raw_text", ""))
                source_info = {
                    "file_id": request.file_id,
                    "filename": document.get("original_filename", "알 수 없는 파일"),
                    "source_type": "document",
                    "text_length": len(combined_text)
                }
        
        elif request.folder_id:
            # 폴더 내 모든 파일에서 키워드 추출
            documents = await db.documents.find({"folder_id": request.folder_id}).to_list(None)
            if not documents:
                raise HTTPException(status_code=404, detail="해당 폴더에 파일이 없습니다.")
            
            if request.use_chunks:
                # 모든 파일의 청크들에서 텍스트 수집
                file_ids = [doc["file_id"] for doc in documents]
                chunks = await db.chunks.find({"file_id": {"$in": file_ids}}).sort("file_id", 1).sort("sequence", 1).to_list(None)
                
                combined_text = "\n\n".join([chunk["text"] for chunk in chunks])
                source_info = {
                    "folder_id": request.folder_id,
                    "source_type": "chunks",
                    "file_count": len(documents),
                    "chunk_count": len(chunks)
                }
            else:
                # 모든 문서의 원본 텍스트 결합
                texts = []
                for doc in documents:
                    text = doc.get("processed_text", doc.get("raw_text", ""))
                    if text:
                        texts.append(text)
                
                combined_text = "\n\n".join(texts)
                source_info = {
                    "folder_id": request.folder_id,
                    "source_type": "documents",
                    "file_count": len(documents),
                    "total_text_length": len(combined_text)
                }
        
        else:
            raise HTTPException(status_code=400, detail="file_id 또는 folder_id 중 하나는 필수입니다.")
        
        if not combined_text.strip():
            raise HTTPException(status_code=404, detail="추출할 텍스트가 없습니다.")
        
        # 텍스트가 너무 긴 경우 제한 (키워드 추출 성능을 위해)
        if len(combined_text) > 10000:
            combined_text = combined_text[:10000] + "..."
            logger.warning("텍스트가 너무 길어 10000자로 제한했습니다.")
        
        # 키워드 추출
        keywords = await labeler.extract_keywords(
            text=combined_text,
            max_keywords=request.max_keywords
        )
        
        return KeywordsResponse(
            keywords=keywords,
            count=len(keywords),
            source_info=source_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 기반 키워드 추출 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/from-folder", response_model=KeywordsResponse)
async def extract_keywords_from_folder(folder_id: str, max_keywords: int = 10, use_chunks: bool = True):
    """폴더에서 키워드 추출 엔드포인트 (간단한 API)"""
    request = FileKeywordsRequest(
        folder_id=folder_id,
        max_keywords=max_keywords,
        use_chunks=use_chunks
    )
    return await extract_keywords_from_file(request)
