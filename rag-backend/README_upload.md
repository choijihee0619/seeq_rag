# 파일 업로드 API 사용법

## 개요
RAG 백엔드에 PDF, DOCX, TXT, HTML 파일을 업로드하고 자동으로 벡터화하여 검색 가능하게 만드는 API입니다.

## 주요 기능

### 1. 파일 업로드
- **엔드포인트**: `POST /upload/`
- **지원 형식**: PDF, DOCX, TXT, HTML, HTM
- **최대 크기**: 50MB
- **처리 과정**:
  1. 파일 업로드 및 검증
  2. 텍스트 추출
  3. 전처리 및 청킹
  4. 벡터 임베딩 생성
  5. MongoDB 저장

### 2. 파일 상태 확인
- **엔드포인트**: `GET /upload/status/{file_id}`
- **기능**: 업로드된 파일의 처리 상태 및 통계 조회

### 3. 파일 목록 조회
- **엔드포인트**: `GET /upload/list`
- **기능**: 업로드된 파일들의 목록 조회
- **필터**: folder_id로 폴더별 필터링 가능

### 4. 파일 삭제
- **엔드포인트**: `DELETE /upload/{file_id}`
- **기능**: 파일 및 관련 청크 데이터 완전 삭제

## API 사용 예시

### cURL 예시

```bash
# 파일 업로드
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@example.pdf" \
  -F "folder_id=my_folder" \
  -F "description=테스트 문서"

# 파일 상태 확인
curl -X GET "http://localhost:8000/upload/status/{file_id}"

# 파일 목록 조회
curl -X GET "http://localhost:8000/upload/list?folder_id=my_folder&limit=10"

# 파일 삭제
curl -X DELETE "http://localhost:8000/upload/{file_id}"
```

### Python 예시

```python
import requests

# 파일 업로드
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "folder_id": "my_folder",
        "description": "중요한 문서"
    }
    response = requests.post("http://localhost:8000/upload/", files=files, data=data)
    result = response.json()
    file_id = result["file_id"]

# 상태 확인
status = requests.get(f"http://localhost:8000/upload/status/{file_id}")
print(status.json())
```

## 데이터베이스 구조

### documents 컬렉션
```json
{
  "_id": "ObjectId",
  "file_id": "uuid",
  "original_filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 1234567,
  "folder_id": "my_folder",
  "description": "문서 설명",
  "raw_text": "원본 텍스트",
  "processed_text": "전처리된 텍스트",
  "text_length": 5000,
  "processing_status": "completed",
  "chunks_count": 15,
  "upload_time": "2024-12-19T10:00:00Z",
  "processing_time": "2024-12-19T10:00:05Z",
  "completed_at": "2024-12-19T10:00:30Z"
}
```

### chunks 컬렉션
```json
{
  "_id": "ObjectId",
  "file_id": "uuid",
  "document_id": "ObjectId",
  "chunk_id": "uuid_chunk_0",
  "sequence": 0,
  "text": "청크 텍스트 내용",
  "text_embedding": [0.1, 0.2, ...],
  "metadata": {
    "chunk_method": "recursive",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "file_type": "pdf",
    "source": "/path/to/file"
  },
  "created_at": "2024-12-19T10:00:30Z"
}
```

## 검색 시 활용

업로드된 문서는 자동으로 검색 시스템에 통합됩니다:

```python
# 특정 폴더의 문서들에서만 검색
response = requests.post("http://localhost:8000/query/", json={
    "query": "검색어",
    "folder_id": "my_folder",
    "top_k": 5
})

# 전체 문서에서 검색
response = requests.post("http://localhost:8000/query/", json={
    "query": "검색어",
    "top_k": 5
})
```

## 주의사항

1. **파일 크기**: 최대 50MB까지 지원
2. **처리 시간**: 파일 크기에 따라 처리 시간이 달라짐
3. **임베딩 비용**: OpenAI API 사용량에 따른 비용 발생
4. **임시 파일**: 처리 완료 후 서버의 임시 파일은 자동 삭제
5. **동시 업로드**: 대용량 파일의 동시 업로드 시 서버 리소스 고려 필요

## 오류 처리

- **400**: 잘못된 파일 형식 또는 크기 초과
- **404**: 파일을 찾을 수 없음
- **500**: 서버 내부 오류 (처리 실패)

모든 오류는 로그에 기록되며, 실패한 업로드는 임시 파일이 자동으로 정리됩니다. 