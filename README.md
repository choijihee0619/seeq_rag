# RAG 백엔드 시스템

OpenAI GPT-4o-mini와 MongoDB를 활용한 통합 RAG(Retrieval Augmented Generation) 백엔드 시스템

## 시스템 개요

본 시스템은 다양한 소스의 텍스트 데이터를 수집, 처리, 저장하고 사용자 질의에 대해 AI 기반 응답을 생성하는 통합 RAG 백엔드입니다.

### 주요 기술 스택
- **LLM**: GPT-4o-mini (OpenAI API)
- **임베딩**: text-embedding-3-large (OpenAI Embedding API)
- **데이터베이스**: MongoDB (벡터 및 일반 데이터 통합 관리)
- **프레임워크**: LangChain, LangServe
- **API 서버**: FastAPI

## 시스템 아키텍처

### 데이터 처리 흐름

```
[입력] → [전처리/청킹] → [임베딩/라벨링] → [MongoDB 저장]
                                ↓
[API 응답] ← [LLM 생성] ← [컨텍스트 조립] ← [RAG 검색]
```

## MongoDB 데이터베이스 구조

### 1. folders 컬렉션
```json
{
  "_id": "ObjectId",
  "name": "폴더명",
  "description": "폴더 설명",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "metadata": {
    "source": "업로드 소스",
    "tags": ["태그1", "태그2"]
  }
}
```

### 2. documents 컬렉션
```json
{
  "_id": "ObjectId",
  "folder_id": "ObjectId",
  "chunk_id": "청크 고유 ID",
  "sequence": 1,
  "text": "원본 텍스트 내용",
  "text_embedding": [0.1, 0.2, ...],  // 1536차원 벡터
  "metadata": {
    "source_file": "원본파일명.pdf",
    "page_number": 1,
    "chunk_method": "sliding_window",
    "chunk_size": 500
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 3. labels 컬렉션
```json
{
  "_id": "ObjectId",
  "document_id": "ObjectId",
  "folder_id": "ObjectId",
  "main_topic": "주요 주제",
  "tags": ["태그1", "태그2", "태그3"],
  "category": "카테고리",
  "confidence": 0.95,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 4. qapairs 컬렉션
```json
{
  "_id": "ObjectId",
  "document_id": "ObjectId",
  "folder_id": "ObjectId",
  "question": "질문 내용",
  "answer": "답변 내용",
  "question_type": "factoid",  // factoid, reasoning, summary
  "difficulty": "medium",  // easy, medium, hard
  "quiz_options": ["선택지1", "선택지2", "선택지3", "선택지4"],
  "correct_option": 0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 5. recommendations 컬렉션 (확장)
```json
{
  "_id": "ObjectId",
  "keyword": "키워드",
  "content_type": "book",  // book, movie, video
  "content_id": "외부API콘텐츠ID",
  "title": "콘텐츠 제목",
  "description": "콘텐츠 설명",
  "source": "naver_books",
  "metadata": {
    "author": "저자",
    "rating": 4.5,
    "url": "https://..."
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

## 설치 및 실행

### 1. 환경 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=mongodb://username:password@host:port/database
MONGODB_DB_NAME=rag_database
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
# 개발 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 또는
python main.py
```

## API 엔드포인트

### 1. 질의 응답
```http
POST /query
Content-Type: application/json

{
  "query": "사용자 질문",
  "folder_id": "폴더ID (선택)",
  "top_k": 5
}
```

### 2. 문서 요약
```http
POST /summary
Content-Type: application/json

{
  "document_ids": ["문서ID1", "문서ID2"],
  "summary_type": "brief"  // brief, detailed
}
```

### 3. 퀴즈 생성
```http
POST /quiz
Content-Type: application/json

{
  "topic": "주제",
  "difficulty": "medium",
  "count": 10
}
```

### 4. 키워드 추출
```http
POST /keywords
Content-Type: application/json

{
  "text": "분석할 텍스트",
  "max_keywords": 10
}
```

### 5. 마인드맵 생성
```http
POST /mindmap
Content-Type: application/json

{
  "root_keyword": "중심 키워드",
  "depth": 3
}
```

### 6. 콘텐츠 추천
```http
POST /recommend
Content-Type: application/json

{
  "keywords": ["키워드1", "키워드2"],
  "content_types": ["book", "movie"]
}
```

## 사용 예시

### 1. 문서 업로드 및 처리
```python
import requests

# 문서 업로드
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
folder_id = response.json()['folder_id']
```

### 2. 질문에 대한 답변 받기
```python
# RAG 기반 질의응답
query_data = {
    "query": "이 문서의 주요 내용은 무엇인가요?",
    "folder_id": folder_id,
    "top_k": 5
}
response = requests.post('http://localhost:8000/query', json=query_data)
answer = response.json()['answer']
```

### 3. 퀴즈 생성
```python
# 자동 퀴즈 생성
quiz_data = {
    "topic": "머신러닝",
    "difficulty": "medium",
    "count": 5
}
response = requests.post('http://localhost:8000/quiz', json=quiz_data)
quizzes = response.json()['quizzes']
```

## 주요 기능

### 1. 데이터 처리 파이프라인
- 다양한 포맷 지원 (PDF, TXT, HTML, MS Office)
- 자동 전처리 및 클렌징
- 지능형 청킹 (문장, 문단, 슬라이딩 윈도우)
- OpenAI 임베딩 생성 및 저장

### 2. AI 기반 자동화
- GPT-4o-mini를 활용한 자동 라벨링
- 질문-답변 쌍 자동 생성
- 퀴즈 및 평가 문항 생성
- 키워드 추출 및 관계 분석

### 3. 통합 검색
- 벡터 유사도 검색 (MongoDB Atlas Search)
- 라벨/카테고리 기반 필터링
- 하이브리드 검색 (벡터 + 텍스트)
- 컨텍스트 기반 재순위화

### 4. 확장 기능
- 마인드맵 자동 생성
- 외부 콘텐츠 추천 (도서, 영화, 동영상)
- 멀티턴 대화 지원
- 사용자 피드백 및 학습

## 성능 최적화

- 배치 임베딩 처리
- MongoDB 인덱스 최적화
- 비동기 처리 및 캐싱
- 컨넥션 풀링

## 보안 고려사항

- API 키 환경변수 관리
- MongoDB 접근 권한 설정
- CORS 설정
- Rate limiting

## 라이선스

MIT License

# 서버 실행
cd rag-backend
python main.py

# 파일 업로드 테스트
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@test.pdf" \
  -F "folder_id=test_folder"

# 업로드된 문서로 검색
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "검색어", "folder_id": "test_folder"}'

# 🎯 API 엔드포인트 테스트 가이드

Swagger UI(`http://0.0.0.0:8000/docs`)에서 각 엔드포인트를 체계적으로 테스트하는 완전 가이드입니다.

## 📋 테스트 준비사항

### 1. 서버 실행 확인
```bash
# 서버 시작
cd rag-backend
conda activate seeq_rag
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 환경 변수 설정 확인
```bash
# .env 파일에 다음 항목들이 설정되어 있는지 확인
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=seeq_rag
```

### 3. 테스트용 샘플 파일 준비
- `sample.txt` (텍스트 파일)
- `sample.pdf` (PDF 파일)  
- `sample.docx` (Word 문서)

## 🔄 엔드포인트 테스트 시나리오

### **1단계: 파일 업로드** 📤

#### **POST /upload/** - 파일 업로드
```
🎯 목적: 문서를 시스템에 업로드하고 처리
📍 위치: Swagger UI > Upload > POST /upload/

테스트 절차:
1. "Try it out" 버튼 클릭
2. 파일 선택:
   - file: PDF/DOCX/TXT/HTML 파일 선택
   - folder_id: (선택사항) "test_folder" 입력
   - description: (선택사항) "테스트 문서입니다" 입력
3. "Execute" 버튼 클릭
4. ✅ 성공 시: file_id 반환값 메모
```

**예상 응답:**
```json
{
  "success": true,
  "message": "파일 업로드 및 처리가 완료되었습니다.",
  "file_id": "06d0b705-1e9d-434b-9259-c9eada4263c3",
  "original_filename": "sample.pdf",
  "processed_chunks": 15,
  "storage_path": null
}
```

#### **GET /upload/status/{file_id}** - 파일 상태 조회
```
🎯 목적: 업로드된 파일의 처리 상태 확인
📍 위치: Swagger UI > Upload > GET /upload/status/{file_id}

테스트 절차:
1. 위에서 얻은 file_id 복사
2. "Try it out" 버튼 클릭  
3. file_id 필드에 복사한 ID 입력
4. "Execute" 버튼 클릭
```

#### **GET /upload/list** - 파일 목록 조회
```
🎯 목적: 업로드된 모든 파일 목록 확인
📍 위치: Swagger UI > Upload > GET /upload/list

테스트 절차:
1. "Try it out" 버튼 클릭
2. 쿼리 파라미터 설정:
   - folder_id: (선택사항) "test_folder"
   - limit: 50
   - skip: 0
3. "Execute" 버튼 클릭
```

### **2단계: 질의응답 테스트** 🔍

#### **POST /query/** - RAG 기반 질의응답
```
🎯 목적: 업로드된 문서를 기반으로 질문에 답변
📍 위치: Swagger UI > Query > POST /query/

테스트 절차:
1. "Try it out" 버튼 클릭
2. Request body 입력:
```

**Request Body 예시:**
```json
{
  "query": "업로드한 문서의 주요 내용을 요약해주세요",
  "folder_id": "test_folder",
  "top_k": 5,
  "include_sources": true
}
```

**다양한 질문 예시:**
```json
// 일반적인 질문
{
  "query": "이 문서에서 가장 중요한 개념은 무엇인가요?",
  "folder_id": null,
  "top_k": 3,
  "include_sources": true
}

// 구체적인 질문  
{
  "query": "SQL의 JOIN 연산에 대해 설명해주세요",
  "folder_id": "test_folder",
  "top_k": 5,
  "include_sources": true
}
```

### **3단계: 문서 요약** 📝

#### **POST /summary/** - 문서 요약 생성
```
🎯 목적: 업로드된 문서들의 요약 생성
📍 위치: Swagger UI > Summary > POST /summary/

테스트 절차:
1. "Try it out" 버튼 클릭
2. Request body 입력:
```

**Request Body 예시:**
```json
{
  "document_ids": null,
  "folder_id": "test_folder", 
  "summary_type": "brief"
}
```

**요약 타입 옵션:**
- `"brief"`: 간단한 요약
- `"detailed"`: 상세한 요약  
- `"bullets"`: 불릿 포인트 형태

### **4단계: 퀴즈 생성** 🧩

#### **POST /quiz/** - 자동 퀴즈 생성
```
🎯 목적: 업로드된 문서 내용 기반 퀴즈 생성
📍 위치: Swagger UI > Quiz > POST /quiz/

테스트 절차:
1. "Try it out" 버튼 클릭
2. Request body 입력:
```

**Request Body 예시:**
```json
{
  "topic": "데이터베이스 기초",
  "folder_id": "test_folder",
  "difficulty": "medium",
  "count": 5,
  "quiz_type": "multiple_choice"
}
```

**옵션 설명:**
- **difficulty**: `"easy"`, `"medium"`, `"hard"`
- **quiz_type**: `"multiple_choice"`, `"true_false"`, `"short_answer"`
- **count**: 생성할 문제 수 (1-20)

### **5단계: 키워드 추출** 🏷️

#### **POST /keywords/** - 텍스트 키워드 추출
```
🎯 목적: 텍스트에서 주요 키워드 자동 추출
📍 위치: Swagger UI > Keywords > POST /keywords/

테스트 절차:
1. "Try it out" 버튼 클릭
2. Request body 입력:
```

**Request Body 예시:**
```json
{
  "text": "SQL은 Structured Query Language의 줄임말로, 관계형 데이터베이스에서 데이터를 관리하기 위한 프로그래밍 언어입니다. SELECT, INSERT, UPDATE, DELETE 등의 명령어를 사용하여 데이터를 조작할 수 있습니다.",
  "max_keywords": 10
}
```

### **6단계: 마인드맵 생성** 🧠

#### **POST /mindmap/** - 개념 마인드맵 생성
```
🎯 목적: 중심 키워드를 기반으로 관련 개념들의 마인드맵 생성
📍 위치: Swagger UI > Mindmap > POST /mindmap/

테스트 절차:
1. 먼저 관련 문서들이 업로드되어 있어야 함
2. "Try it out" 버튼 클릭
3. Request body 입력:
```

**Request Body 예시:**
```json
{
  "root_keyword": "데이터베이스",
  "depth": 3,
  "max_nodes": 20
}
```

### **7단계: 콘텐츠 추천** 💡

#### **POST /recommend/** - 관련 콘텐츠 추천
```
🎯 목적: 키워드 기반 학습 콘텐츠 추천
📍 위치: Swagger UI > Recommend > POST /recommend/

테스트 절차:
1. "Try it out" 버튼 클릭
2. Request body 입력:
```

**Request Body 예시:**
```json
{
  "keywords": ["SQL", "데이터베이스", "프로그래밍"],
  "content_types": ["book", "video", "article"],
  "max_items": 10
}
```

**content_types 옵션:**
- `"book"`: 도서
- `"movie"`: 영화  
- `"video"`: 동영상
- `"article"`: 아티클

### **8단계: 파일 관리** 🗂️

#### **DELETE /upload/{file_id}** - 파일 삭제
```
🎯 목적: 업로드된 파일과 관련 데이터 삭제
📍 위치: Swagger UI > Upload > DELETE /upload/{file_id}

⚠️ 주의: 삭제된 파일은 복구할 수 없습니다!

테스트 절차:
1. 삭제할 file_id 준비
2. "Try it out" 버튼 클릭
3. file_id 입력 후 "Execute"
```

## 🔄 권장 테스트 워크플로우

### **기본 시나리오**
```
1. 📤 파일 업로드 (POST /upload/)
   ↓ file_id 획득
   
2. 📋 업로드 확인 (GET /upload/list)
   ↓ 파일 목록에서 확인
   
3. 🔍 질의응답 (POST /query/)
   ↓ 문서 기반 질문
   
4. 📝 요약 생성 (POST /summary/)
   ↓ 문서 내용 요약
   
5. 🧩 퀴즈 생성 (POST /quiz/)
   ↓ 학습용 문제 생성
   
6. 🏷️ 키워드 추출 (POST /keywords/)
   ↓ 주요 개념 추출
   
7. 🧠 마인드맵 (POST /mindmap/)
   ↓ 개념 연관관계
   
8. 💡 콘텐츠 추천 (POST /recommend/)
   ↓ 추가 학습자료
```

### **고급 테스트 시나리오**
```
멀티 문서 테스트:
1. 여러 관련 문서 업로드 (같은 folder_id 사용)
2. 폴더 단위 요약 생성
3. 통합 마인드맵 생성
4. 크로스 문서 질의응답

성능 테스트:
1. 대용량 파일 업로드 (최대 50MB)
2. 복잡한 질문으로 응답 시간 측정
3. 다중 키워드 추천 테스트
```

## ⚠️ 트러블슈팅

### **자주 발생하는 오류와 해결방법**

#### 1. 파일 업로드 실패
```
오류: "파일 처리 중 오류가 발생했습니다"
해결: 
- OpenAI API 키 확인
- MongoDB 연결 상태 확인
- 파일 형식 확인 (PDF, DOCX, TXT, HTML만 지원)
- 파일 크기 확인 (최대 50MB)
```

#### 2. 질의응답 오류
```
오류: "질의 처리 실패"
해결:
- 먼저 문서가 업로드되어 있는지 확인
- folder_id가 올바른지 확인
- 업로드된 문서가 처리 완료되었는지 확인
```

#### 3. 데이터베이스 연결 오류
```
오류: "Database objects do not implement truth value testing"
해결:
- MongoDB 서버가 실행 중인지 확인
- .env 파일의 MONGODB_URI 확인
- 최신 motor 라이브러리 설치 확인
```

## 🎯 테스트 체크리스트

### **기본 기능 테스트**
- [ ] 파일 업로드 성공
- [ ] 파일 상태 조회 성공
- [ ] 파일 목록 조회 성공
- [ ] 질의응답 정상 동작
- [ ] 문서 요약 생성 성공

### **고급 기능 테스트**  
- [ ] 퀴즈 생성 정상 동작
- [ ] 키워드 추출 성공
- [ ] 마인드맵 생성 성공
- [ ] 콘텐츠 추천 정상 동작
- [ ] 파일 삭제 성공

### **에러 처리 테스트**
- [ ] 잘못된 파일 형식 업로드 시 적절한 오류 메시지
- [ ] 존재하지 않는 file_id 조회 시 404 에러
- [ ] 빈 질문 입력 시 적절한 처리
- [ ] API 한도 초과 시 오류 처리

이제 `http://0.0.0.0:8000/docs`에서 위의 가이드를 따라 체계적으로 테스트해보세요! 🚀