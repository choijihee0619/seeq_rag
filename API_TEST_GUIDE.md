# RAG 백엔드 API 테스트 가이드

## 개요
이 문서는 RAG 백엔드 API의 모든 엔드포인트와 JSON 파라미터 예시를 정리한 종합 가이드입니다.

**기본 정보:**
- 서버 URL: `http://localhost:8000`
- API 문서: `http://localhost:8000/docs` (Swagger UI)
- API 버전: 1.0.0

---

## 1. 폴더 관리 (Folders) - `/folders`

### 1.1 폴더 생성
**POST** `/folders/`

**JSON 파라미터:**
```json
{
  "title": "AI 학습 자료",
  "folder_type": "academic",
  "cover_image_url": "https://example.com/image.jpg"
}
```

**파라미터 설명:**
- `title` (필수): 폴더명 (최대 100자)
- `folder_type` (선택): 폴더 유형 (기본값: "general")
- `cover_image_url` (선택): 커버 이미지 URL

### 1.2 폴더 목록 조회
**GET** `/folders/`

**Query 파라미터:**
- `limit` (선택): 조회할 폴더 수 (기본값: 50)
- `skip` (선택): 건너뛸 폴더 수 (기본값: 0)

### 1.3 특정 폴더 조회
**GET** `/folders/{folder_id}`

**Path 파라미터:**
- `folder_id` (필수): 폴더 ObjectId

### 1.4 폴더 정보 수정
**PUT** `/folders/{folder_id}`

**JSON 파라미터:**
```json
{
  "title": "수정된 폴더명",
  "folder_type": "research"
}
```

### 1.5 폴더 삭제
**DELETE** `/folders/{folder_id}`

**Query 파라미터:**
- `force` (선택): 강제 삭제 여부 (기본값: false)

---

## 2. 파일 업로드 및 관리 (Upload) - `/upload`

### 2.1 파일 업로드
**POST** `/upload/`

**Form Data 파라미터:**

**방법 1: folder_id로 폴더 지정**
```json
{
  "file": "업로드할 파일",
  "folder_id": "64f7b8a12345678901234567",
  "description": "AI 관련 연구 논문"
}
```

**방법 2: folder_title로 폴더 지정**
```json
{
  "file": "업로드할 파일",
  "folder_title": "AI 학습 자료",
  "description": "마케팅 관리 중간고사 정리 자료"
}
```

**방법 3: 폴더 지정 없이 업로드**
```json
{
  "file": "업로드할 파일",
  "description": "독립적인 파일"
}
```

**파라미터 설명:**
- `file` (필수): 업로드할 파일 (.txt, .pdf, .docx, .doc, .md)
- `folder_id` (선택): 폴더 ObjectId - 기존 폴더의 고유 ID
- `folder_title` (선택): 폴더 제목 - 폴더 생성시 입력한 title 값
- `description` (선택): 파일 설명

**📌 중요 사항:**
- `folder_id`와 `folder_title`은 **동시에 입력할 수 없습니다** (둘 중 하나만 선택)
- `folder_title` 사용시 해당 제목의 폴더가 미리 생성되어 있어야 합니다
- 둘 다 입력하지 않으면 폴더에 속하지 않는 독립 파일로 저장됩니다
- 최대 파일 크기: 10MB
- 업로드 실패시 데이터베이스 내 file_info에 실패 로그 생성

### 2.2 파일 상태 조회
**GET** `/upload/status/{file_id}`

**Path 파라미터:**
- `file_id` (필수): 파일 UUID

### 2.3 파일 검색
**POST** `/upload/search`

**JSON 파라미터:**
```json
{
  "query": "머신러닝",
  "search_type": "both",
  "folder_id": "64f7b8a12345678901234567",
  "limit": 10,
  "skip": 0
}
```

**파라미터 설명:**
- `query` (필수): 검색어
- `search_type` (선택): 검색 유형 ("filename", "content", "both", 기본값: "both")
- `folder_id` (선택): 폴더 ID로 필터링
- `limit` (선택): 결과 수 제한 (기본값: 20)
- `skip` (선택): 건너뛸 결과 수 (기본값: 0)

### 2.4 파일 목록 조회
**GET** `/upload/list`

**Query 파라미터:**
- `folder_id` (선택): 폴더 ID로 필터링
- `limit` (선택): 결과 수 제한 (기본값: 50)
- `skip` (선택): 건너뛸 결과 수 (기본값: 0)

### 2.5 시맨틱 검색 ⚠️ 한국어 검색어 주의
**GET** `/upload/semantic-search`

**Query 파라미터:**
- `q` (필수): 검색어 (**한국어는 URL 인코딩 필요**)
- `k` (선택): 결과 개수 (기본값: 5)
- `folder_id` (선택): 폴더 ID

**📌 한국어 검색어 처리:**
- 한국어 검색어는 URL 인코딩이 필요합니다
- 예: `금융` → `%EA%B8%88%EC%9C%B5`
- 영어 검색어는 인코딩 없이 사용 가능

### 2.6 파일 내용 조회
**GET** `/upload/content/{file_id}`

**Path 파라미터:**
- `file_id` (필수): 파일 UUID

### 2.7 파일 정보 수정
**PUT** `/upload/{file_id}`

**JSON 파라미터:**
```json
{
  "filename": "새로운_파일명.pdf",
  "description": "수정된 설명",
  "folder_id": "64f7b8a12345678901234567"
}
```

**파라미터 설명:**
- `filename` (선택): 새로운 파일명
- `description` (선택): 새로운 설명
- `folder_id` (선택): 이동할 폴더 ID
- `folder_title` (선택): 이동할 폴더 제목

### 2.8 파일 미리보기
**GET** `/upload/preview/{file_id}`

**Query 파라미터:**
- `max_length` (선택): 최대 미리보기 길이 (기본값: 500)

### 2.9 파일 삭제
**DELETE** `/upload/{file_id}`

**Path 파라미터:**
- `file_id` (필수): 파일 UUID

---

## 3. 질의응답 (Query) - `/query`

### 3.1 질의 처리
**POST** `/query/`

**JSON 파라미터:**
```json
{
  "query": "머신러닝이란 무엇인가요?",
  "folder_id": "64f7b8a12345678901234567",
  "top_k": 5,
  "include_sources": true
}
```

**파라미터 설명:**
- `query` (필수): 질의 내용
- `folder_id` (선택): 특정 폴더에서만 검색
- `top_k` (선택): 검색할 문서 수 (기본값: 5)
- `include_sources` (선택): 소스 포함 여부 (기본값: true)

---

## 4. 요약 (Summary) - `/summary`

### 4.1 요약 생성
**POST** `/summary/`

**JSON 파라미터:**
```json
{
  "folder_id": "64f7b8a12345678901234567",
  "summary_type": "detailed"
}
```

**다른 예시:**
```json
{
  "document_ids": ["file1", "file2", "file3"],
  "summary_type": "brief"
}
```

**파라미터 설명:**
- `document_ids` (선택): 특정 문서 ID 리스트
- `folder_id` (선택): 폴더 ID
- `summary_type` (선택): 요약 유형 ("brief", "detailed", "bullets", 기본값: "brief")

### 4.2 캐시된 요약 목록 조회
**GET** `/summary/cached`

**Query 파라미터:**
- `folder_id` (선택): 폴더 ID로 필터링
- `limit` (선택): 결과 수 제한 (기본값: 10)

### 4.3 요약 캐시 삭제
**DELETE** `/summary/cached/{cache_id}`

**Path 파라미터:**
- `cache_id` (필수): 캐시 ID

---

## 5. 퀴즈 (Quiz) - `/quiz`

### 5.1 퀴즈 생성
**POST** `/quiz/`

**JSON 파라미터:**
```json
{
  "topic": "머신러닝",
  "folder_id": "64f7b8a12345678901234567",
  "difficulty": "medium",
  "count": 5,
  "quiz_type": "multiple_choice"
}
```

**파라미터 설명:**
- `topic` (선택): 퀴즈 주제
- `folder_id` (선택): 폴더 ID
- `difficulty` (선택): 난이도 ("easy", "medium", "hard", 기본값: "medium")
- `count` (선택): 퀴즈 개수 (기본값: 5)
- `quiz_type` (선택): 퀴즈 유형 ("multiple_choice", "true_false", "short_answer", "fill_in_blank")

### 5.2 퀴즈 히스토리 조회
**GET** `/quiz/history`

**Query 파라미터:**
- `folder_id` (선택): 폴더 ID로 필터링
- `limit` (선택): 결과 수 제한 (기본값: 20)

### 5.3 퀴즈 통계 조회
**GET** `/quiz/stats`

**Query 파라미터:**
- `folder_id` (선택): 폴더 ID로 필터링

### 5.4 퀴즈 삭제
**DELETE** `/quiz/{quiz_id}`

**Path 파라미터:**
- `quiz_id` (필수): 퀴즈 ID

---

## 6. 키워드 추출 (Keywords) - `/keywords`

### 6.1 텍스트에서 키워드 추출
**POST** `/keywords/`

**JSON 파라미터:**
```json
{
  "text": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않고도 학습할 수 있는 능력을 제공합니다.",
  "max_keywords": 10
}
```

**파라미터 설명:**
- `text` (필수): 키워드를 추출할 텍스트
- `max_keywords` (선택): 최대 키워드 수 (기본값: 10)

### 6.2 파일에서 키워드 추출
**POST** `/keywords/from-file`

**JSON 파라미터:**
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "max_keywords": 10,
  "use_chunks": true
}
```

**다른 예시 (폴더 기반):**
```json
{
  "folder_id": "64f7b8a12345678901234567",
  "max_keywords": 15,
  "use_chunks": false
}
```

**파라미터 설명:**
- `file_id` (선택): 파일 ID
- `folder_id` (선택): 폴더 ID
- `max_keywords` (선택): 최대 키워드 수 (기본값: 10)
- `use_chunks` (선택): 청크 사용 여부 (기본값: true)

**📌 주의사항:**
- `file_id`와 `folder_id` 중 하나는 필수입니다

### 6.3 폴더에서 키워드 추출 (간단 API)
**POST** `/keywords/from-folder`

**Query 파라미터:**
- `folder_id` (필수): 폴더 ID
- `max_keywords` (선택): 최대 키워드 수 (기본값: 10)
- `use_chunks` (선택): 청크 사용 여부 (기본값: true)

---

## 7. 마인드맵 (Mindmap) - `/mindmap`

### 7.1 마인드맵 생성
**POST** `/mindmap/`

**JSON 파라미터:**
```json
{
  "root_keyword": "머신러닝",
  "depth": 3,
  "max_nodes": 20,
  "folder_id": "64f7b8a12345678901234567"
}
```

**파라미터 설명:**
- `root_keyword` (필수): 중심 키워드
- `depth` (선택): 마인드맵 깊이 (기본값: 3)
- `max_nodes` (선택): 최대 노드 수 (기본값: 20)
- `folder_id` (선택): 폴더 ID

---

## 8. 추천 (Recommend) - `/recommend`

### 8.1 키워드 기반 추천
**POST** `/recommend/`

**JSON 파라미터:**
```json
{
  "keywords": ["머신러닝", "딥러닝", "AI"],
  "content_types": ["book", "movie", "youtube_video"],
  "max_items": 10,
  "include_youtube": true,
  "youtube_max_per_keyword": 3,
  "folder_id": "64f7b8a12345678901234567"
}
```

**파라미터 설명:**
- `keywords` (필수): 추천을 위한 키워드 리스트
- `content_types` (선택): 콘텐츠 유형 (기본값: ["book", "movie", "youtube_video"])
- `max_items` (선택): 최대 추천 수 (기본값: 10)
- `include_youtube` (선택): YouTube 포함 여부 (기본값: true)
- `youtube_max_per_keyword` (선택): 키워드당 YouTube 결과 수 (기본값: 3)
- `folder_id` (선택): 폴더 ID

### 8.2 파일 기반 자동 추천 ✅ 수정됨
**POST** `/recommend/from-file`

**JSON 파라미터 (파일 기반):**
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "content_types": ["book", "youtube_video"],
  "max_items": 10,
  "include_youtube": true,
  "youtube_max_per_keyword": 3,
  "max_keywords": 5
}
```

**JSON 파라미터 (폴더 기반):**
```json
{
  "folder_id": "64f7b8a12345678901234567",
  "content_types": ["book", "movie"],
  "max_items": 8,
  "include_youtube": false,
  "max_keywords": 3
}
```

**파라미터 설명:**
- `file_id` (선택): 파일 ID
- `folder_id` (선택): 폴더 ID
- `content_types` (선택): 콘텐츠 유형 (기본값: ["book", "movie", "youtube_video"])
- `max_items` (선택): 최대 추천 수 (기본값: 10)
- `include_youtube` (선택): YouTube 포함 여부 (기본값: true)
- `youtube_max_per_keyword` (선택): 키워드당 YouTube 결과 수 (기본값: 3)
- `max_keywords` (선택): 추출할 최대 키워드 수 (기본값: 5)

**📌 주의사항:**
- `file_id`와 `folder_id` 중 하나는 필수입니다
- 파일에서 자동으로 키워드를 추출하여 추천을 생성합니다

### 8.3 캐시된 추천 목록 조회
**GET** `/recommend/cached`

**Query 파라미터:**
- `folder_id` (선택): 폴더 ID로 필터링
- `limit` (선택): 결과 수 제한 (기본값: 10)

### 8.4 추천 캐시 삭제
**DELETE** `/recommend/cached/{cache_id}`

**Path 파라미터:**
- `cache_id` (필수): 캐시 ID

---

## 9. 기본 정보 조회

### 9.1 루트 엔드포인트
**GET** `/`

응답 예시:
```json
{
  "message": "RAG 백엔드 API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 테스트 시나리오 예시

### 시나리오 1: 기본 문서 업로드 및 질의응답

1. **폴더 생성**
```json
{
  "title": "테스트 폴더",
  "folder_type": "test"
}
```

2. **파일 업로드** (Form Data)
```json
{
  "file": "test_document.pdf",
  "folder_id": "<생성된_폴더_ID>"
}
```

3. **질의응답**
```json
{
  "query": "문서의 주요 내용은 무엇인가요?",
  "folder_id": "<폴더_ID>",
  "top_k": 3
}
```

### 시나리오 2: 종합 분석 워크플로우

1. **파일 업로드** (Form Data)
2. **키워드 추출**
```json
{
  "file_id": "<파일_ID>",
  "max_keywords": 10
}
```

3. **요약 생성**
```json
{
  "folder_id": "<폴더_ID>",
  "summary_type": "detailed"
}
```

4. **퀴즈 생성**
```json
{
  "folder_id": "<폴더_ID>",
  "count": 5,
  "difficulty": "medium"
}
```

5. **자동 추천**
```json
{
  "file_id": "<파일_ID>",
  "max_items": 10
}
```

---

## 주요 수정 사항 및 주의사항

### ✅ 최근 수정 사항
1. **GET /upload/semantic-search**: 한국어 검색어 URL 인코딩 처리 개선
2. **POST /recommend/from-file**: 키워드 추출 로직 수정으로 정상 작동

### ⚠️ 주의사항
1. **인증**: 현재 API에는 인증이 구현되어 있지 않습니다.
2. **CORS**: 모든 도메인에서 접근 가능하도록 설정되어 있습니다.
3. **파일 크기**: 업로드 파일은 최대 10MB까지 지원합니다.
4. **ObjectId**: MongoDB ObjectId는 24자리 16진수 문자열입니다.
5. **환경변수**: YouTube API 기능을 사용하려면 `YOUTUBE_API_KEY`가 필요합니다.
6. **한국어 검색어**: 시맨틱 검색에서 한국어 검색어는 URL 인코딩이 필요합니다.

### 📊 데이터베이스 정보
- **MONGODB_URI**: mongodb+srv://
- **MONGODB_DB_NAME**: 

---

## 에러 응답 형식

모든 API는 다음과 같은 형식의 에러 응답을 반환합니다:

```json
{
  "detail": "에러 메시지"
}
```

일반적인 HTTP 상태 코드:
- `400`: 잘못된 요청 (파라미터 오류 등)
- `404`: 리소스를 찾을 수 없음
- `500`: 서버 내부 오류

---

## API 문서 링크

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc` 