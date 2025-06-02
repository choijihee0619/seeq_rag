# 🚀 SEEQ RAG 백엔드 시스템

**AI 기반 통합 문서 관리 및 질의응답 시스템**

OpenAI GPT-4o-mini와 MongoDB를 활용한 차세대 RAG(Retrieval Augmented Generation) 백엔드로, 문서 업로드부터 AI 기반 분석, 추천까지 원스톱 솔루션을 제공합니다.

## 📋 목차

- [시스템 개요](#-시스템-개요)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [시스템 아키텍처](#-시스템-아키텍처)
- [데이터베이스 구조](#-데이터베이스-구조)
- [설치 및 실행](#-설치-및-실행)
- [API 엔드포인트 가이드](#-api-엔드포인트-가이드)
- [사용 예시](#-사용-예시)
- [프론트엔드 통합 가이드](#-프론트엔드-통합-가이드)
- [트러블슈팅](#-트러블슈팅)

## 🎯 시스템 개요

SEEQ RAG는 다양한 문서 포맷을 자동 처리하여 AI 기반 질의응답, 요약, 키워드 추출, 콘텐츠 추천을 제공하는 통합 백엔드 시스템입니다.

### 핵심 특징
- **🤖 AI 통합 분석**: GPT-4o-mini 기반 질의응답 및 문서 분석
- **📁 스마트 폴더 관리**: folder_id 기반 논리적 문서 그룹화
- **🔍 하이브리드 검색**: 키워드 검색 + AI 의미 검색
- **📊 자동 콘텐츠 생성**: 키워드, 요약, 퀴즈, 마인드맵 자동 생성
- **🎨 멀티소스 추천**: 웹 검색 + YouTube + DB 통합 실시간 추천
- **⚡ 최적화된 성능**: 리팩토링된 코드로 중복 제거 및 성능 향상

## ✨ 주요 기능

### 📤 문서 관리
- **다중 포맷 지원**: PDF, DOCX, TXT, DOC, MD (최대 10MB)
- **자동 텍스트 추출**: 파일 타입별 최적화된 파서
- **스마트 청킹**: 500자 단위, 50자 오버랩으로 컨텍스트 보존
- **폴더 기반 관리**: 사용자 정의 폴더로 문서 논리적 분류

### 🤖 AI 기능
- **질의응답**: RAG 기반 문맥 인식 Q&A (출처 정보 포함)
- **문서 요약**: brief/detailed/bullets 형태로 맞춤 요약
- **키워드 추출**: AI 기반 핵심 개념 자동 추출
- **퀴즈 생성**: 객관식/OX/주관식 문제 자동 생성
- **마인드맵**: 개념 간 연관관계 시각화 데이터

### 🔍 검색 엔진
- **자연어 파일 검색**: 파일명 + 내용 통합 검색
- **AI 의미 검색**: 벡터 임베딩 기반 유사도 검색
- **폴더 필터링**: 특정 폴더 내 검색 제한 가능

### 💡 추천 시스템 (하이브리드 멀티소스)
- **🌐 웹 검색 추천**: LLM과 실시간 웹 검색을 통한 도서/영화/비디오 추천
  - **🛡️ 환각 방지 시스템**: 검색 결과에 명시된 정보만 사용, 추측 금지
  - **✅ 신뢰성 검증**: 추천 내용의 필수 필드 및 키워드 포함 여부 검증
  - **⚠️ 면책 메타데이터**: 모든 웹 추천에 확인 필요 안내 포함
- **🔴 YouTube 실시간**: YouTube API 기반 관련 교육 동영상 추천  
- **🗄️ 데이터베이스**: 저장된 추천 데이터 검색
- **📁 파일 기반**: 업로드 문서 자동 분석 후 맞춤 콘텐츠 추천

#### 🛡️ 웹 검색 추천 환각 방지 시스템

**LLM 프롬프트 템플릿 강화:**
- 검색 결과에 명시된 정보만 사용하도록 제한
- 추측이나 가정 기반 내용 생성 금지
- 불확실한 정보 포함 시 빈 배열 반환
- Temperature 0.1로 창의성 최소화

**신뢰성 검증 시스템:**
- `_validate_recommendation()`: 추천 결과 품질 검증
- 필수 필드 존재 여부 확인 (title, content_type, description, source)
- 키워드 포함 여부 검증
- 설명 길이 최소 기준 (20자 이상) 확인

**면책 메타데이터 자동 추가:**
모든 웹 검색 기반 추천에 다음 메타데이터 포함:
```json
{
  "metadata": {
    "disclaimer": "웹 검색 기반 일반적 추천으로, 실제 존재 여부를 확인하시기 바랍니다",
    "recommendation_type": "general_guidance",
    "verification_required": true,
    "generated_by": "web_search_template",
    "reliability": "template_based"
  }
}
```

## 🛠️ 기술 스택

| 구분 | 기술 | 용도 |
|------|------|------|
| **LLM** | GPT-4o-mini | 질의응답, 요약, 키워드 추출 |
| **임베딩** | text-embedding-3-large | 1536차원 벡터 생성 |
| **데이터베이스** | MongoDB | 문서/벡터 통합 저장 |
| **웹 프레임워크** | FastAPI | REST API 서버 |
| **AI 프레임워크** | LangChain | LLM 체인 관리 |
| **비동기 처리** | motor | MongoDB 비동기 드라이버 |
| **외부 API** | YouTube Data API v3 | 실시간 동영상 추천 |
| **웹 검색** | httpx + LLM | 실시간 웹 크롤링 및 콘텐츠 파싱 |
| **문서 처리** | PyPDF2, python-docx | 다양한 포맷 파싱 |
| **🛡️ 환각 방지** | Temperature 0.1 + 검증 시스템 | LLM 환각 최소화 및 신뢰성 검증 |

## 🏗️ 시스템 아키텍처

### 데이터 처리 파이프라인
```
📁 파일 업로드
    ↓
🔍 파일 검증 (포맷/크기)
    ↓
📄 텍스트 추출 (PDF/DOCX/TXT)
    ↓
✂️ 텍스트 청킹 (500자 단위)
    ↓
🧠 임베딩 생성 (OpenAI)
    ↓
💾 MongoDB 저장 (documents + chunks)
    ↓
🛡️ AI 라벨링 (키워드/카테고리) + 🛡️ 환각 방지 검증
    ↓
✅ 처리 완료
```

**🛡️ AI 라벨링 환각 방지 시스템:**
- **Temperature 0.1**: LLM 창의성 최소화로 환각 위험 감소
- **텍스트 기반 분석**: 제공된 문서 내용에만 기반한 키워드 추출
- **일반어 필터링**: "있다", "하다" 등 의미 없는 키워드 자동 제거
- **신뢰도 조정**: 환각 방지로 인한 보수적 신뢰도 점수 적용 (0.8 → 0.3)

### 질의응답 흐름
```
❓ 사용자 질의
    ↓
🔍 벡터 유사도 검색 (chunks 컬렉션)
    ↓
📋 관련 문서 청크 수집
    ↓
🤖 LLM 컨텍스트 생성
    ↓
💬 GPT-4o-mini 답변 생성
    ↓
📎 출처 정보 첨부
    ↓
✨ 최종 응답 반환
```

### 모듈 구조
```
rag-backend/
├── 🔧 설정 및 환경 파일
│   ├── .env                          # 환경 변수 설정 (API 키, DB 연결 정보)
│   ├── .env.example                  # 환경 변수 템플릿 파일
│   ├── .gitignore                    # Git 버전 관리 제외 파일 목록
│   ├── main.py                       # FastAPI 메인 애플리케이션 & 서버 진입점
│   └── requirements.txt              # Python 의존성 패키지 목록 (최적화됨)
│
├── 📚 API 계층 (FastAPI 라우터 & 비즈니스 로직)
│   ├── api/
│   │   ├── __init__.py              # API 패키지 초기화
│   │   ├── 🌐 routers/              # REST API 엔드포인트
│   │   │   ├── __init__.py          # 라우터 패키지 초기화
│   │   │   ├── upload.py            # 📤 파일 업로드/관리/검색 API (709 lines)
│   │   │   ├── query.py             # 💬 RAG 질의응답 API
│   │   │   ├── summary.py           # 📄 문서 요약 API
│   │   │   ├── quiz.py              # 🧩 퀴즈 생성 API (객관식/OX/단답형/빈칸)
│   │   │   ├── keywords.py          # 🏷️ 키워드 추출 API
│   │   │   ├── mindmap.py           # 🧠 마인드맵 생성 API
│   │   │   └── recommend.py         # 💡 콘텐츠 추천 API (멀티소스)
│   │   └── 🔗 chains/               # LangChain 비즈니스 로직
│   │       ├── __init__.py          # 체인 패키지 초기화
│   │       ├── query_chain.py       # RAG 질의응답 체인 로직
│   │       ├── summary_chain.py     # 문서 요약 체인 로직
│   │       ├── quiz_chain.py        # 퀴즈 생성 체인 로직
│   │       └── recommend_chain.py   # 추천 체인 로직 (335 lines)
│
├── 🤖 AI 처리 모듈 (LLM 및 AI 기능)
│   ├── ai_processing/
│   │   ├── __init__.py              # AI 처리 패키지 초기화
│   │   ├── llm_client.py            # OpenAI GPT-4o-mini 클라이언트
│   │   ├── labeler.py               # 🛡️ 환각 방지 자동 라벨링 & 키워드 추출
│   │   └── qa_generator.py          # 퀴즈 & Q&A 자동 생성기
│
├── 📄 문서 처리 파이프라인 (파일 → 텍스트 → 벡터)
│   ├── data_processing/
│   │   ├── __init__.py              # 문서 처리 패키지 초기화
│   │   ├── document_processor.py    # 🔄 통합 문서 처리 파이프라인 (260 lines)
│   │   ├── loader.py                # 파일 로더 (PDF/DOCX/TXT/DOC/MD)
│   │   ├── preprocessor.py          # 텍스트 전처리 (정제/정규화)
│   │   ├── chunker.py              # 텍스트 청킹 (500자 단위, 50자 오버랩)
│   │   └── embedder.py             # OpenAI 임베딩 생성 (text-embedding-3-large)
│
├── 💾 데이터베이스 관리 (MongoDB 연결 & 조작)
│   ├── database/
│   │   ├── __init__.py              # 데이터베이스 패키지 초기화
│   │   ├── connection.py            # MongoDB 비동기 연결 관리 (motor)
│   │   └── operations.py            # CRUD 연산 & 데이터베이스 유틸리티
│
├── 🔍 검색 엔진 (벡터 검색 & 하이브리드 검색)
│   ├── retrieval/
│   │   ├── __init__.py              # 검색 패키지 초기화
│   │   ├── vector_search.py         # 벡터 유사도 검색 (numpy 기반)
│   │   ├── hybrid_search.py         # 하이브리드 검색 (키워드 + 벡터)
│   │   └── context_builder.py       # RAG용 컨텍스트 구성기
│
├── 🛠️ 유틸리티 모듈 (공통 기능 & 외부 API)
│   ├── utils/
│   │   ├── __init__.py              # 유틸리티 패키지 초기화
│   │   ├── logger.py                # Loguru 기반 로깅 시스템
│   │   ├── validators.py            # 입력 데이터 검증 함수
│   │   ├── text_collector.py        # 📝 텍스트 수집 통합 유틸리티 (185 lines)
│   │   ├── youtube_api.py           # 🔴 YouTube Data API v3 연동 (346 lines)
│   │   └── web_recommendation.py    # 🛡️ 웹 검색 기반 추천 (환각 방지, 381 lines)
│
├── ⚙️ 설정 관리
│   ├── config/
│   │   ├── __init__.py              # 설정 패키지 초기화
│   │   └── settings.py              # Pydantic 기반 환경 설정 관리
│
└── 📝 로그 디렉토리 (실행 로그 저장소)
    └── logs/                        # 애플리케이션 로그 파일 저장 (현재 비어있음)
```

## 💾 데이터베이스 구조

### 1. `documents` 컬렉션 (문서 메타데이터)
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "uuid-generated-string",           // 자동 생성 고유 ID
  "original_filename": "SQL기초_강의자료.pdf",
  "file_type": "pdf",                           // pdf, docx, txt, doc, md
  "file_size": 1024000,                         // 바이트 단위
  "folder_id": "programming_docs",              // 사용자 정의 폴더명
  "description": "SQL 기초 학습 자료",
  "raw_text": "원본 추출 텍스트...",
  "processed_text": "전처리된 텍스트...",
  "text_length": 15000,
  "processing_status": "completed",
  "chunks_count": 30,
  "upload_time": ISODate("2024-01-21T10:00:00Z"),
  "processing_time": ISODate("2024-01-21T10:01:00Z")
}
```

### 2. `chunks` 컬렉션 (벡터 저장소)
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "uuid-generated-string",
  "chunk_id": "file-id_chunk_0",
  "sequence": 0,                                // 청크 순서
  "text": "SQL은 Structured Query Language...",
  "text_embedding": [0.1, 0.2, 0.3, ...],     // 1536차원 벡터
  "metadata": {
    "source": "SQL기초_강의자료.pdf",
    "file_type": "pdf",
    "folder_id": "programming_docs",            // 폴더 필터링용
    "chunk_method": "sliding_window",
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "created_at": ISODate("2024-01-21T10:02:00Z")
}
```

### 3. `labels` 컬렉션 (AI 자동 라벨링)
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "uuid-generated-string",
  "folder_id": "programming_docs",
  "main_topic": "데이터베이스 기초",
  "tags": ["SQL", "데이터베이스", "RDBMS", "쿼리"],
  "category": "프로그래밍",
  "confidence": 0.92,
  "created_at": ISODate("2024-01-21T10:03:00Z")
}
```

### 4. `qapairs` 컬렉션 (Q&A 및 퀴즈)
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "uuid-generated-string",
  "folder_id": "programming_docs",
  "question": "SQL에서 JOIN의 종류는?",
  "answer": "INNER, LEFT, RIGHT, FULL OUTER JOIN",
  "question_type": "factoid",                   // factoid, concept, application
  "difficulty": "medium",                       // easy, medium, hard
  "quiz_options": ["A", "B", "C", "D"],        // 객관식 선택지
  "correct_option": 2,                          // 정답 인덱스
  "created_at": ISODate("2024-01-21T10:04:00Z")
}
```

### 5. `recommendations` 컬렉션 (추천 콘텐츠)
```javascript
{
  "_id": ObjectId("..."),
  "keyword": "SQL",
  "content_type": "youtube_video",              // book, movie, video, youtube_video
  "title": "SQL 기초부터 고급까지",
  "description": "3시간 완성 SQL 강의",
  "source": "https://youtube.com/watch?v=...",
  "metadata": {
    "video_id": "abc123",
    "channel_title": "코딩 교육",
    "view_count": 150000,
    "duration": "3:15:30",
    "thumbnail": "https://img.youtube.com/..."
  },
  "recommendation_source": "youtube_realtime", // database, youtube_realtime, fallback
  "created_at": ISODate("2024-01-21T10:05:00Z")
}
```

### 📊 주요 인덱스
```javascript
// 성능 최적화를 위한 인덱스
db.documents.createIndex({ "file_id": 1 }, { unique: true })
db.documents.createIndex({ "folder_id": 1, "upload_time": -1 })
db.chunks.createIndex({ "file_id": 1, "sequence": 1 })
db.chunks.createIndex({ "metadata.folder_id": 1 })
db.chunks.createIndex({ "text_embedding": "2dsphere" })  // 벡터 검색용
```

## ⚙️ 설치 및 실행

### 1. 환경 설정

**`.env` 파일 생성 (프로젝트 루트)**
```bash
# OpenAI API (필수)
OPENAI_API_KEY=sk-your-openai-api-key

# MongoDB (필수)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=seeq_rag

# YouTube API (선택사항 - 추천 기능 강화)
YOUTUBE_API_KEY=your-youtube-api-key

# 서버 설정
API_HOST=0.0.0.0
API_PORT=8000

# 처리 설정
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_TOP_K=5
LOG_LEVEL=INFO
```

### 2. 의존성 설치
```bash
cd rag-backend
pip install -r requirements.txt
```

**주요 라이브러리:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
langchain==0.1.5
langchain-openai==0.0.5
openai==1.10.0
pymongo==4.6.1
motor==3.3.2
pypdf==3.17.4
python-docx==1.1.0
google-api-python-client==2.108.0
```

### 3. MongoDB 설정
```bash
# MongoDB 서비스 시작
sudo systemctl start mongod

# 연결 확인
mongosh --eval "db.runCommand('ping')"
```

### 4. 서버 실행
```bash
# 개발 서버 (자동 리로드)
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**접속 확인:**
- 🌐 API 문서: http://localhost:8000/docs
- 🔧 서버 상태: http://localhost:8000/

## 📚 API 엔드포인트 가이드

### 📤 파일 관리

#### `POST /upload/` - 파일 업로드 (핵심)
```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@document.pdf" \
  -F "folder_id=my_documents" \
  -F "description=중요한 학습 자료"
```

**응답:**
```json
{
  "success": true,
  "message": "파일 업로드가 완료되었습니다.",
  "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
  "original_filename": "document.pdf",
  "processed_chunks": 25
}
```

#### `GET /upload/list` - 파일 목록
```bash
curl "http://localhost:8000/upload/list?folder_id=my_documents&limit=10"
```

#### `POST /upload/search` - 자연어 파일 검색
```json
{
  "query": "SQL 데이터베이스",
  "search_type": "both",     // filename, content, both
  "folder_id": "programming_docs",
  "limit": 10
}
```

#### `GET /upload/semantic-search` - AI 의미 검색
```bash
curl "http://localhost:8000/upload/semantic-search?q=데이터%20분석&k=5"
```

#### `GET /upload/content/{file_id}` - 파일 원본 텍스트
#### `GET /upload/preview/{file_id}` - 파일 미리보기
#### `PUT /upload/{file_id}` - 파일 정보 수정

### 💬 질의응답

#### `POST /query/` - RAG 질의응답
```json
{
  "query": "이 문서에서 JOIN 연산에 대해 설명해주세요",
  "folder_id": "programming_docs",
  "top_k": 5,
  "include_sources": true
}
```

**응답:**
```json
{
  "answer": "JOIN 연산은 두 개 이상의 테이블을 연결하여...",
  "sources": [
    {
      "chunk_id": "abc123_chunk_5",
      "text": "JOIN 연산은...",
      "relevance_score": 0.95,
      "file_name": "SQL기초.pdf"
    }
  ],
  "confidence": 0.92
}
```

### 📄 문서 분석

#### `POST /summary/` - 문서 요약
```json
{
  "folder_id": "programming_docs",
  "summary_type": "detailed"    // brief, detailed, bullets
}
```

#### `POST /keywords/from-file` - 키워드 추출 (권장)
```json
{
  "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
  "max_keywords": 10,
  "use_chunks": true
}
```

**폴더 전체 키워드:**
```json
{
  "folder_id": "programming_docs",
  "max_keywords": 15
}
```

#### `POST /mindmap/` - 마인드맵 생성
```json
{
  "root_keyword": "데이터베이스",
  "depth": 3,
  "max_nodes": 20,
  "folder_id": "programming_docs"
}
```

### 🧩 퀴즈 & 학습

#### `POST /quiz/` - 퀴즈 생성

**지원하는 퀴즈 타입:**
- `multiple_choice` - 4지선다 객관식 (기본값)
- `true_false` - 참/거짓 문제
- `short_answer` - 단답형 문제
- `fill_in_blank` - 빈 칸 채우기

#### 1. 객관식 퀴즈 (Multiple Choice)
```json
{
  "topic": "SQL 기초",
  "folder_id": "programming_docs",
  "difficulty": "medium",
  "count": 5,
  "quiz_type": "multiple_choice"
}
```

**응답 예시:**
```json
{
  "quizzes": [
    {
      "question": "SQL에서 ALIAS를 사용할 때 어떤 기호를 사용해야 하는가?",
      "quiz_type": "multiple_choice",
      "options": ["이중 인용부호", "단일 인용부호", "괄호", "없음"],
      "correct_option": 0,
      "difficulty": "medium",
      "explanation": "ALIAS가 공백이나 특수문자를 포함하는 경우에는 이중 인용부호를 사용해야 합니다."
    }
  ],
  "topic": "SQL 기초",
  "total_count": 5
}
```

#### 2. 참/거짓 퀴즈 (True/False)
```json
{
  "topic": "데이터베이스",
  "difficulty": "easy",
  "count": 3,
  "quiz_type": "true_false"
}
```

**응답 예시:**
```json
{
  "quizzes": [
    {
      "question": "SQL에서 PRIMARY KEY는 NULL 값을 가질 수 있다.",
      "quiz_type": "true_false",
      "options": ["참", "거짓"],
      "correct_option": 1,
      "difficulty": "easy",
      "explanation": "PRIMARY KEY는 NULL 값을 가질 수 없습니다."
    }
  ]
}
```

#### 3. 단답형 퀴즈 (Short Answer)
```json
{
  "topic": "SQL JOIN",
  "difficulty": "medium",
  "count": 2,
  "quiz_type": "short_answer"
}
```

**응답 예시:**
```json
{
  "quizzes": [
    {
      "question": "두 테이블에서 공통된 값을 가진 행만 반환하는 JOIN의 종류는?",
      "quiz_type": "short_answer",
      "correct_answer": "INNER JOIN",
      "difficulty": "medium",
      "explanation": "INNER JOIN은 양쪽 테이블에 모두 존재하는 데이터만 반환합니다."
    }
  ]
}
```

#### 4. 빈 칸 채우기 퀴즈 (Fill in the Blank)
```json
{
  "topic": "데이터베이스 정규화",
  "difficulty": "hard",
  "count": 2,
  "quiz_type": "fill_in_blank"
}
```

**응답 예시:**
```json
{
  "quizzes": [
    {
      "question": "데이터베이스에서 ___은 중복 데이터를 제거하고 데이터 일관성을 유지하는 과정이다.",
      "quiz_type": "fill_in_blank",
      "correct_answer": "정규화",
      "difficulty": "hard",
      "explanation": "정규화는 데이터베이스 설계에서 중복을 제거하는 중요한 과정입니다."
    }
  ]
}
```

**퀴즈 공통 파라미터:**
- `topic` (Optional): 퀴즈 주제 (없으면 폴더 전체에서 생성)
- `folder_id` (Optional): 특정 폴더 내 문서만 사용
- `difficulty`: "easy", "medium", "hard" 중 선택 (기본값: "medium")
- `count`: 생성할 퀴즈 개수 (기본값: 5)
- `quiz_type`: 퀴즈 타입 (기본값: "multiple_choice")

### 💡 추천 시스템

#### `POST /recommend/from-file` - 파일 기반 추천 (핵심)
```json
{
  "file_id": "e3b4ffab-4bd0-4fa0-8a94-862b138d6b41",
  "content_types": ["book", "youtube_video"],
  "max_items": 10,
  "include_youtube": true
}
```

**응답:**
```json
{
  "recommendations": [
    {
      "title": "SQL 완벽 가이드",
      "content_type": "book",
      "description": "데이터베이스 초보자를 위한...",
      "source": "교보문고",
      "keyword": "SQL",
      "recommendation_source": "database"
    },
    {
      "title": "SQL 기초 강의",
      "content_type": "youtube_video",
      "source": "https://youtube.com/watch?v=...",
      "metadata": {
        "channel_title": "코딩 교육",
        "view_count": 85000,
        "duration": "10:30"
      },
      "recommendation_source": "youtube_realtime"
    },
    {
      "title": "데이터베이스 핵심 이론과 실무 적용",
      "content_type": "book",
      "description": "데이터베이스 분야의 기본 개념부터 실무 적용 사례까지 체계적으로 다룬 입문서",
      "source": "웹 검색 기반 추천",
      "keyword": "데이터베이스",
      "recommendation_source": "web_realtime",
      "metadata": {
        "category": "전문서적",
        "target_audience": "입문자~중급자",
        "content_type_detail": "이론서",
        "search_source": "web_realtime",
        "reliability": "template_based",
        "disclaimer": "웹 검색 기반 일반적 추천으로, 실제 존재 여부를 확인하시기 바랍니다",
        "recommendation_type": "general_guidance",
        "verification_required": true,
        "generated_by": "web_search_template"
      }
    }
  ],
  "extracted_keywords": ["SQL", "데이터베이스", "JOIN"]
}
```

#### 📊 추천 소스별 특징

**1. `database` (저장된 데이터)**
- 검증된 추천 데이터 활용
- 높은 신뢰도와 정확성
- 즉시 응답 가능

**2. `youtube_realtime` (YouTube API)**
- 실시간 YouTube 검색 결과
- 조회수, 채널명, 재생시간 등 메타데이터 포함
- API 제한에 따른 일일 할당량 존재

**3. `web_realtime` (웹 검색 + LLM)**
- ⚠️ **환각 방지 시스템 적용**
- 모든 추천에 `verification_required: true` 포함
- 사용자 확인 필수 안내 메타데이터
- 템플릿 기반 안전한 추천 생성

**4. `fallback` (기본 추천)**
- 모든 소스 실패 시 사용
- 일반적인 학습 콘텐츠 제공

## 🔧 사용 예시

### 완전한 워크플로우

#### 1. 파일 업로드 및 처리
```bash
# 1. PDF 파일 업로드
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@SQL기초강의.pdf" \
  -F "folder_id=database_learning" \
  -F "description=SQL 기초 학습 자료"

# 응답에서 file_id 확인: "abc123-def456-..."
```

#### 2. 업로드된 파일 검색
```bash
# 자연어 검색
curl -X POST "http://localhost:8000/upload/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "SQL 기초", "search_type": "both"}'

# AI 의미 검색
curl "http://localhost:8000/upload/semantic-search?q=데이터%20분석&k=3"
```

#### 3. 문서 내용 질의응답
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "INNER JOIN과 LEFT JOIN의 차이점은?",
    "folder_id": "database_learning",
    "include_sources": true
  }'
```

#### 4. 자동 키워드 추출
```bash
curl -X POST "http://localhost:8000/keywords/from-file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123-def456-...",
    "max_keywords": 10,
    "use_chunks": true
  }'
```

#### 5. 개인화된 콘텐츠 추천
```bash
curl -X POST "http://localhost:8000/recommend/from-file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123-def456-...",
    "content_types": ["book", "youtube_video"],
    "max_items": 10,
    "include_youtube": true
  }'
```

#### 6. 학습 퀴즈 생성

**객관식 퀴즈:**
```bash
curl -X POST "http://localhost:8000/quiz/" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "SQL JOIN 연산",
    "folder_id": "database_learning",
    "difficulty": "medium",
    "count": 5,
    "quiz_type": "multiple_choice"
  }'
```

**참/거짓 퀴즈:**
```bash
curl -X POST "http://localhost:8000/quiz/" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "데이터베이스 기초",
    "difficulty": "easy",
    "count": 3,
    "quiz_type": "true_false"
  }'
```

**단답형 퀴즈:**
```bash
curl -X POST "http://localhost:8000/quiz/" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "SQL 명령어",
    "difficulty": "medium",
    "count": 4,
    "quiz_type": "short_answer"
  }'
```

**빈 칸 채우기 퀴즈:**
```bash
curl -X POST "http://localhost:8000/quiz/" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "데이터베이스 정규화",
    "difficulty": "hard",
    "count": 2,
    "quiz_type": "fill_in_blank"
  }'
```

**퀴즈 공통 파라미터:**
- `topic` (Optional): 퀴즈 주제 (없으면 폴더 전체에서 생성)
- `folder_id` (Optional): 특정 폴더 내 문서만 사용
- `difficulty`: "easy", "medium", "hard" 중 선택 (기본값: "medium")
- `count`: 생성할 퀴즈 개수 (기본값: 5)
- `quiz_type`: 퀴즈 타입 (기본값: "multiple_choice")

### 고급 사용 시나리오

#### 폴더 전체 분석
```bash
# 폴더 내 모든 파일 요약
curl -X POST "http://localhost:8000/summary/" \
  -H "Content-Type: application/json" \
  -d '{"folder_id": "database_learning", "summary_type": "detailed"}'

# 폴더 전체 키워드 추출
curl -X POST "http://localhost:8000/keywords/from-file" \
  -H "Content-Type: application/json" \
  -d '{"folder_id": "database_learning", "max_keywords": 20}'
```

#### 마인드맵 생성
```bash
curl -X POST "http://localhost:8000/mindmap/" \
  -H "Content-Type: application/json" \
  -d '{
    "root_keyword": "데이터베이스",
    "depth": 3,
    "max_nodes": 15,
    "folder_id": "database_learning"
  }'
```

## 🎨 프론트엔드 통합 가이드

### 기본 통합 개념

#### 1. 파일 관리 시스템
My Library 대시보드를 구현할 때는 `/upload/list` API를 사용하여 파일 목록을 가져와야 합니다. 응답에서 받은 file_id는 시스템에서 자동 생성된 고유 식별자이며, original_filename은 사용자가 업로드한 원본 파일명입니다. folder_id는 사용자가 정의한 폴더명으로 논리적 그룹화에 사용됩니다.

#### 2. 파일 업로드
파일 업로드 시에는 FormData를 사용하여 multipart/form-data 형태로 전송해야 합니다. 필수 필드는 file이며, folder_id와 description은 선택사항입니다. 업로드 완료 후 응답에서 file_id를 받아와서 이후 모든 API 호출에 사용해야 합니다.

### 핵심 UI 컴포넌트 구현

#### 1. 파일 상세 페이지 구현

**좌측 파일 목록 + Raw Text 토글**
파일 상세 페이지에서는 좌측에 파일 목록을 표시하고, 우측에 파일 내용을 보여줘야 합니다. Raw Text 토글 기능을 구현하여 사용자가 원본 텍스트와 처리된 청크 미리보기를 선택할 수 있도록 해야 합니다. Raw text 조회는 `/upload/content/{file_id}` API를, 미리보기는 `/upload/preview/chunks/{file_id}` API를 사용합니다.

#### 2. 드래그 앤 드롭 모듈 시스템

**키워드 모듈**
키워드 모듈은 파일이 드롭되면 즉시 로딩 상태를 표시하고, `/keywords/from-file` API를 호출하여 기본값(max_keywords: 10, use_chunks: true)으로 키워드를 추출합니다. 결과 표시 후 설정 패널을 제공하여 사용자가 키워드 수나 청크 사용 여부를 조정할 수 있도록 해야 합니다.

**마인드맵 모듈**
마인드맵 모듈은 파일 드롭 시 중심 키워드 입력 모달을 먼저 표시해야 합니다. 사용자가 키워드를 입력하면 `/mindmap/` API를 호출하여 마인드맵 데이터를 생성합니다. 기본 설정값은 depth: 3, max_nodes: 20입니다. 생성 후 설정 패널에서 깊이와 노드 수를 조절할 수 있어야 합니다.

*마인드맵 시각화 구현 가이드라인:*
- **엣지 스타일링**: 가중치 0.2 이상은 굵은 선으로, 0.2 미만은 가는 선으로 표시
- **노드 배치**: 루트 노드는 중앙 배치, 나머지 노드들은 루트 중심으로 원형 배열
- **계층 구조**: 레벨별 색상 구분으로 시각적 계층구조 표현 (루트: 빨간색, 레벨1: 파란색, 레벨2: 초록색)
- **노드 크기**: 루트 노드는 더 크게, 하위 레벨은 상대적으로 작게 표시
- **상호작용**: 노드 클릭 시 관련 정보 툴팁 표시, 드래그로 배치 조정 가능

**추천 모듈**
추천 모듈은 파일 드롭 시 자동으로 `/recommend/from-file` API를 호출하여 관련 콘텐츠를 추천받습니다. 기본적으로 도서와 YouTube 동영상을 포함하며, 추출된 키워드도 함께 표시해야 합니다. 각 추천 항목은 제목, 설명, 콘텐츠 타입, 외부 링크를 포함해야 합니다.

#### 3. 챗봇 인터페이스
질의응답 챗봇은 `/query/` API를 사용하여 구현합니다. 사용자 메시지 전송 시 folder_id를 포함하여 특정 폴더 내에서만 검색하도록 할 수 있습니다. 응답에는 답변과 함께 출처 정보와 신뢰도 점수가 포함되므로 이를 UI에 표시해야 합니다.

### 파일 검색 기능 구현

#### 통합 검색 인터페이스
검색 기능은 두 가지 API를 조합하여 구현해야 합니다. `/upload/search`는 파일명과 내용을 대상으로 하는 일반 검색이고, `/upload/semantic-search`는 AI 기반 의미 검색입니다. 검색 결과를 각각 다른 섹션으로 나누어 표시하여 사용자가 구분할 수 있도록 해야 합니다.

### 폴더 관리 시스템
폴더는 MongoDB에 별도 컬렉션으로 생성되지 않고 folder_id 필드로만 관리됩니다. 새 폴더는 파일 업로드 시 자동으로 생성되며, 폴더명은 사용자가 자유롭게 설정할 수 있습니다. 폴더 간 파일 이동은 `/upload/{file_id}` PUT API를 사용하여 folder_id를 변경하면 됩니다.

### 필수 구현 가이드라인

#### 1. 사용자 경험 최적화
- **자동 완성**: folder_id 입력 시 기존 폴더명 자동완성 기능 구현
- **실시간 미리보기**: 파일 선택 시 즉시 미리보기 제공
- **모듈 설정**: 드롭 후 기본값 표시 + 설정 변경 옵션 제공

#### 2. 오류 처리
모든 API 호출에 대해 표준 오류 처리 로직을 구현해야 합니다. HTTP 상태 코드를 확인하고, 실패 시 적절한 에러 메시지를 사용자에게 표시해야 합니다. 네트워크 오류나 서버 오류에 대한 재시도 로직도 고려해야 합니다.

#### 3. 로딩 상태 관리
API 호출 중에는 로딩 스피너와 처리 중 메시지를 표시해야 합니다. 특히 파일 업로드나 키워드 추출 등 시간이 오래 걸리는 작업에서는 사용자에게 진행 상황을 명확히 알려주어야 합니다.

### 권장 UI/UX 패턴

1. **file_id 숨김**: 사용자에게는 파일명만 표시하고, file_id는 내부적으로만 사용
2. **폴더 시각화**: 트리 구조 또는 태그 형태로 folder_id 표시
3. **모듈 설정 패널**: 드롭 후 기본 결과 표시 + 토글/슬라이더로 설정 변경 가능
4. **검색 탭**: 일반 검색과 AI 검색을 탭으로 분리
5. **실시간 피드백**: API 호출 중 로딩 스피너, 완료 시 성공 메시지 표시

## 🐛 트러블슈팅

### 자주 발생하는 문제

#### 1. OpenAI API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY

# 사용량 확인
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 2. MongoDB 연결 실패
```bash
# 서비스 상태 확인
systemctl status mongod

# 연결 테스트
mongosh --eval "db.runCommand('ping')"
```

#### 3. 파일 업로드 실패
- **크기 제한**: 10MB 이하인지 확인
- **포맷 지원**: PDF, DOCX, TXT, DOC, MD만 가능
- **권한 확인**: uploads 폴더 쓰기 권한

#### 4. 키워드 추출 오류
- **텍스트 길이**: 최소 50자 이상 필요
- **처리 상태**: 파일 업로드 완료 후 키워드 추출
- **API 한도**: OpenAI API 사용량 확인

#### 5. 추천 기능 오류
- **YouTube API**: 선택사항이므로 키 없어도 기본 추천 동작
- **키워드 품질**: 의미있는 키워드 추출되었는지 확인

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/app.log

# 오류 로그만 확인
grep ERROR logs/app.log
```

## 📈 향후 개발 계획

### 단기 계획 (1-2개월)
- [ ] **성능 최적화**: 대용량 파일 처리 속도 향상
- [ ] **배치 처리**: 여러 파일 동시 업로드 기능
- [ ] **고급 검색**: 날짜/파일타입/크기 필터 추가
- [ ] **사용자 인증**: JWT 기반 사용자 관리

### 중기 계획 (3-6개월)
- [ ] **다국어 지원**: 영어/일본어 문서 처리
- [ ] **실시간 협업**: 여러 사용자 동시 작업
- [ ] **모바일 지원**: 반응형 API 및 모바일 최적화
- [ ] **고급 분석**: 문서 간 유사도 및 관계 분석

### 장기 계획 (6개월+)
- [ ] **GraphQL API**: RESTful 외 GraphQL 지원
- [ ] **캐싱 시스템**: Redis 활용 응답 속도 향상
- [ ] **AI 모델 학습**: 도메인 특화 모델 파인튜닝
- [ ] **클라우드 배포**: AWS/GCP 배포 및 확장

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

**💬 문의사항**: 이슈 트래커를 통해 버그 리포트 및 기능 요청 환영  
**🔄 최종 업데이트**: 2024년 1월 21일  
**⭐ 버전**: v2.0 (리팩토링 완료)