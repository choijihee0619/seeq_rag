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
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib httpx
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

### 7. 콘텐츠 추천
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

#### **A. POST /keywords/from-file** - 파일에서 키워드 추출 ⭐
```
🎯 목적: 데이터베이스에 저장된 파일에서 주요 키워드 자동 추출
📍 위치: Swagger UI > Keywords > POST /keywords/from-file

✨ 이 기능이 더 실용적입니다! 업로드된 문서에서 자동으로 키워드를 추출합니다.

테스트 절차:
1. 먼저 파일을 업로드하고 file_id 확인
2. "Try it out" 버튼 클릭
3. Request body 입력:
```

**특정 파일에서 키워드 추출:**
```json
{
  "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
  "max_keywords": 10,
  "use_chunks": true
}
```

**폴더 전체에서 키워드 추출:**
```json
{
  "folder_id": "test_folder",
  "max_keywords": 15,
  "use_chunks": true
}
```

**옵션 설명:**
- **file_id**: 특정 파일의 고유 ID (file_id 또는 folder_id 중 하나 필수)
- **folder_id**: 폴더 ID (해당 폴더의 모든 파일에서 키워드 추출)
- **max_keywords**: 추출할 최대 키워드 수 (기본값: 10)
- **use_chunks**: 
  - `true`: 청크 단위로 분할된 텍스트에서 추출 (더 정확, 권장)
  - `false`: 원본 문서 전체에서 추출 (더 포괄적)

**응답 예시:**
```json
{
  "keywords": ["데이터 모델링", "엔터티", "관계", "속성", "정규화", "SQL", "데이터베이스", "식별자", "ERD", "트랜잭션"],
  "count": 10,
  "source_info": {
    "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
    "source_type": "chunks",
    "chunk_count": 27
  }
}
```

#### **B. POST /keywords/from-folder** - 간단한 폴더 키워드 추출
```
🎯 목적: URL 파라미터로 간단하게 폴더에서 키워드 추출
📍 위치: Swagger UI > Keywords > POST /keywords/from-folder

테스트 절차:
1. "Try it out" 버튼 클릭
2. Query Parameters 입력:
   - folder_id: "test_folder"
   - max_keywords: 10
   - use_chunks: true
```

#### **C. POST /keywords/** - 직접 텍스트에서 키워드 추출
```
🎯 목적: 사용자가 입력한 텍스트에서 주요 키워드 자동 추출
📍 위치: Swagger UI > Keywords > POST /keywords/

💡 사용 시나리오: 외부 텍스트나 임시 텍스트에서 키워드를 추출하고 싶을 때

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

**응답 예시:**
```json
{
  "keywords": ["SQL", "데이터베이스", "프로그래밍", "언어", "SELECT"],
  "count": 5,
  "source_info": null
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

#### **🚀 NEW! POST /recommend/from-file** - 업로드된 파일 기반 자동 추천 (권장)
```
🎯 목적: 업로드된 파일의 내용을 자동 분석하여 관련 학습 콘텐츠 추천
📍 위치: Swagger UI > Recommend > POST /recommend/from-file

✨ 이것이 바로 원하는 기능입니다! 파일을 업로드하면 자동으로 키워드를 추출하고 관련 콘텐츠를 추천합니다.

테스트 절차:
1. 먼저 파일을 업로드하고 file_id 확인 (POST /upload/)
2. "Try it out" 버튼 클릭
3. Request body 작성
```

#### **자동 추천 테스트 예시:**

**📚 1. 특정 파일 기반 자동 추천**
```json
{
  "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
  "content_types": ["book", "youtube_video", "article"],
  "max_items": 10,
  "include_youtube": true,
  "youtube_max_per_keyword": 3,
  "max_keywords": 5
}
```

**📁 2. 폴더 전체 기반 자동 추천**
```json
{
  "folder_id": "programming_docs",
  "content_types": ["book", "youtube_video", "movie"],
  "max_items": 15,
  "include_youtube": true,
  "youtube_max_per_keyword": 4,
  "max_keywords": 7
}
```

**🎥 3. YouTube 중심 자동 추천**
```json
{
  "file_id": "data_science_paper.pdf",
  "content_types": ["youtube_video"],
  "max_items": 20,
  "include_youtube": true,
  "youtube_max_per_keyword": 5,
  "max_keywords": 6
}
```

**📖 4. 도서 중심 자동 추천**
```json
{
  "folder_id": "business_analysis",
  "content_types": ["book"],
  "max_items": 8,
  "include_youtube": false,
  "max_keywords": 4
}
```

#### **예상 응답 구조:**
```json
{
  "recommendations": [
    {
      "title": "파이썬 완전 정복 - 기초부터 실무까지",
      "content_type": "book",
      "description": "초보자를 위한 Python 프로그래밍 완벽 가이드",
      "source": "교보문고",
      "metadata": {
        "author": "홍길동",
        "publisher": "프로그래밍 출판사",
        "rating": 4.5
      },
      "keyword": "Python",
      "recommendation_source": "database"
    },
    {
      "title": "Python 머신러닝 강의 - 1시간 완성",
      "content_type": "youtube_video",
      "description": "Python을 활용한 머신러닝 기초 강의",
      "source": "https://youtube.com/watch?v=...",
      "metadata": {
        "video_id": "abc123",
        "channel_title": "코딩 교육 채널",
        "view_count": 150000,
        "duration": "1:05:30"
      },
      "keyword": "머신러닝",
      "recommendation_source": "youtube_realtime"
    }
  ],
  "total_count": 15,
  "youtube_included": true,
  "sources_summary": {
    "database": 8,
    "youtube_realtime": 7
  },
  "extracted_keywords": ["Python", "머신러닝", "데이터 분석", "FastAPI", "프로그래밍"]
}
```

#### **🎯 실전 활용 시나리오:**

**시나리오 1: 논문 기반 학습 자료 추천**
1. 연구 논문 PDF 업로드
2. `/recommend/from-file`로 자동 추천 요청
3. 논문 주제 관련 YouTube 강의, 교재, 참고 도서 자동 추천
4. 추출된 키워드 확인으로 학습 방향 설정

**시나리오 2: 프로젝트 문서 기반 스킬업**
1. 프로젝트 명세서나 기술 문서 업로드
2. 필요 기술 스택 자동 분석
3. 관련 온라인 강의, 튜토리얼, 도서 추천
4. 개발 역량 향상을 위한 학습 경로 제안

**시나리오 3: 교육과정 폴더 기반 통합 추천**
1. 여러 강의 자료를 하나의 폴더에 업로드
2. 폴더 전체 기반 종합 추천
3. 심화 학습을 위한 추가 자료 발견
4. 다양한 매체(책, 동영상, 영화)를 통한 입체적 학습

---

#### **POST /recommend/** - 키워드 직접 입력 추천 (기존 방식)
```
🎯 목적: 사용자가 직접 입력한 키워드로 학습 콘텐츠 추천
📍 위치: Swagger UI > Recommend > POST /recommend/

💡 사용 시나리오: 특정 키워드에 대한 추천이 필요할 때 (파일 없이)

테스트 절차:
1. "Try it out" 버튼 클릭
2. 원하는 키워드와 설정으로 Request body 작성
```

#### **키워드 직접 입력 테스트 예시:**

**📚 1. Python 학습 자료 추천**
```json
{
  "keywords": ["Python", "프로그래밍", "FastAPI"],
  "content_types": ["book", "youtube_video", "article"],
  "max_items": 10,
  "include_youtube": true,
  "youtube_max_per_keyword": 3
}
```

**🎥 2. YouTube 중심 추천 (머신러닝)**
```json
{
  "keywords": ["머신러닝", "데이터 사이언스", "딥러닝"],
  "content_types": ["youtube_video"],
  "max_items": 15,
  "include_youtube": true,
  "youtube_max_per_keyword": 5
}
```

**📖 3. 비즈니스 도서 추천**
```json
{
  "keywords": ["경영학", "마케팅", "비즈니스 전략"],
  "content_types": ["book"],
  "max_items": 8,
  "include_youtube": false
}
```

**🎬 4. 역사 영화/다큐 추천**
```json
{
  "keywords": ["한국사", "조선시대", "임진왜란"],
  "content_types": ["movie"],
  "max_items": 5,
  "include_youtube": false
}
```

**🌟 5. 종합 추천 (모든 콘텐츠 타입)**
```json
{
  "keywords": ["인공지능", "AI", "ChatGPT"],
  "content_types": ["book", "movie", "video", "youtube_video", "article"],
  "max_items": 20,
  "include_youtube": true,
  "youtube_max_per_keyword": 2
}
```

#### **YouTube 특화 추천 엔드포인트들:**

**🔥 POST /recommend/youtube/trending** - 인기 YouTube 동영상
```json
{
  "category_id": "28",
  "max_results": 10
}
```
**카테고리 ID 참고:**
- `"28"`: 과학&기술
- `"27"`: 교육  
- `"22"`: 사람&블로그
- `"24"`: 엔터테인먼트
- `"10"`: 음악

**📋 GET /recommend/youtube/categories** - YouTube 카테고리 목록 조회
```
🎯 목적: 사용할 수 있는 YouTube 카테고리 확인
📍 위치: Swagger UI > Recommend > GET /recommend/youtube/categories
```

**💾 POST /recommend/youtube/save** - 좋은 YouTube 동영상 DB 저장
```json
{
  "keyword": "Python 프로그래밍 기초",
  "video_id": "kqtD5dpn9C8"
}
```

#### **추천 시스템 전체 워크플로우:**

```
1. 📤 파일 업로드 (POST /upload/)
   ↓ file_id 획득
   
2. 🔍 자동 키워드 추출 및 추천 (POST /recommend/from-file) ⭐
   ↓ 파일 내용 분석 → 키워드 추출 → 관련 콘텐츠 추천
   
3. 📚 추천 결과 활용
   ↓ YouTube 강의, 관련 도서, 참고 영화 등
   
4. 💾 좋은 콘텐츠 저장 (POST /recommend/youtube/save)
   ↓ 추후 동일 키워드 검색 시 우선 추천
```

**content_types 전체 옵션:**
- `"book"`: 도서 추천
- `"movie"`: 영화/다큐멘터리
- `"video"`: 일반 동영상  
- `"youtube_video"`: YouTube 동영상 (실시간 검색)
- `"article"`: 온라인 아티클/블로그

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
   
6. 🏷️ 키워드 추출 (POST /keywords/from-file)
   ↓ 파일에서 주요 개념 추출
   
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
3. 폴더 전체 키워드 추출 (POST /keywords/from-file)
4. 통합 마인드맵 생성
5. 크로스 문서 질의응답

성능 테스트:
1. 대용량 파일 업로드 (최대 50MB)
2. 복잡한 질문으로 응답 시간 측정
3. 대량 텍스트에서 키워드 추출 성능 테스트
4. 다중 키워드 추천 테스트
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

#### 3. 키워드 추출 오류
```
오류: "키워드 추출 실패" 또는 422 Unprocessable Entity
해결:
- 파일 기반 추출: file_id 또는 folder_id가 올바른지 확인
- 직접 텍스트 추출: text 필드가 비어있지 않은지 확인
- 존재하는 파일 ID인지 확인 (GET /upload/list로 확인)
- 텍스트가 너무 짧지 않은지 확인 (최소 50자 이상 권장)
```

#### 4. 데이터베이스 연결 오류
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
- [ ] 파일 기반 키워드 추출 성공 (POST /keywords/from-file)
- [ ] 직접 텍스트 키워드 추출 성공 (POST /keywords/)
- [ ] 폴더 전체 키워드 추출 성공
- [ ] 마인드맵 생성 성공
- [ ] 콘텐츠 추천 정상 동작
- [ ] 파일 삭제 성공

### **키워드 추출 세부 테스트**
- [ ] 특정 파일에서 키워드 추출 (use_chunks: true)
- [ ] 특정 파일에서 키워드 추출 (use_chunks: false)
- [ ] 폴더 전체에서 키워드 추출
- [ ] 추출된 키워드 수가 요청한 max_keywords와 일치
- [ ] source_info에 올바른 메타데이터 포함

### **에러 처리 테스트**
- [ ] 잘못된 파일 형식 업로드 시 적절한 오류 메시지
- [ ] 존재하지 않는 file_id 조회 시 404 에러
- [ ] 빈 질문 입력 시 적절한 처리
- [ ] 빈 text 필드로 키워드 추출 시 422 에러
- [ ] API 한도 초과 시 오류 처리

이제 `http://0.0.0.0:8000/docs`에서 위의 가이드를 따라 체계적으로 테스트해보세요! 🚀