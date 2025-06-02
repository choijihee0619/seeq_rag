#!/bin/bash

# RAG 백엔드 프로젝트 생성 스크립트
# 프로젝트 루트 디렉토리 생성
mkdir -p rag-backend
cd rag-backend

# 디렉토리 구조 생성
mkdir -p config models data_processing ai_processing retrieval api/routers api/chains database utils tests

# .env.example 파일 생성
cat > .env.example << 'EOF'
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB 설정
MONGODB_URI=mongodb://username:password@host:port/database
MONGODB_DB_NAME=rag_database

# API 서버 설정
API_HOST=0.0.0.0
API_PORT=8000

# 로깅 설정
LOG_LEVEL=INFO
EOF

# .gitignore 파일 생성
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 환경 파일
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# 로그
logs/
*.log

# 테스트
.coverage
.pytest_cache/
htmlcov/

# OS
.DS_Store
Thumbs.db
EOF

# requirements.txt 파일 생성
cat > requirements.txt << 'EOF'
# 핵심 의존성
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.6

# LangChain 관련
langchain==0.1.5
langchain-openai==0.0.5
langchain-community==0.0.16
langserve==0.0.39

# OpenAI
openai==1.10.0

# MongoDB
pymongo==4.6.1
motor==3.3.2

# 유틸리티
python-dotenv==1.0.0
aiofiles==23.2.1
pandas==2.1.4
numpy==1.26.3

# 문서 처리
pypdf==3.17.4
python-docx==1.1.0
beautifulsoup4==4.12.3

# 로깅 및 모니터링
loguru==0.7.2

# 테스트
pytest==7.4.4
pytest-asyncio==0.23.3
EOF


# main.py 파일 생성
cat > main.py << 'EOF'
"""
RAG 백엔드 메인 애플리케이션
FastAPI 서버 초기화 및 라우터 설정
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from config.settings import settings
from database.connection import init_db, close_db
from api.routers import query, summary, quiz, keywords, mindmap, recommend
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("RAG 백엔드 서버 시작...")
    await init_db()
    yield
    # 종료 시
    await close_db()
    logger.info("RAG 백엔드 서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="RAG 백엔드 API",
    description="OpenAI GPT-4o-mini와 MongoDB를 활용한 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(summary.router, prefix="/summary", tags=["Summary"])
app.include_router(quiz.router, prefix="/quiz", tags=["Quiz"])
app.include_router(keywords.router, prefix="/keywords", tags=["Keywords"])
app.include_router(mindmap.router, prefix="/mindmap", tags=["Mindmap"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "RAG 백엔드 API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
EOF

# config/__init__.py
touch config/__init__.py

# config/settings.py 파일 생성
cat > config/settings.py << 'EOF'
"""
설정 관리 모듈
환경 변수 로드 및 설정 값 관리
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # OpenAI 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # MongoDB 설정
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "rag_database"
    
    # API 설정
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # 청킹 설정
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # 검색 설정
    DEFAULT_TOP_K: int = 5
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
EOF

# models/__init__.py
touch models/__init__.py

# models/folder.py 파일 생성
cat > models/folder.py << 'EOF'
"""
폴더 모델 정의
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict

class FolderBase(BaseModel):
    """폴더 기본 모델"""
    name: str = Field(..., description="폴더명")
    description: Optional[str] = Field(None, description="폴더 설명")
    metadata: Optional[Dict] = Field(default_factory=dict, description="메타데이터")

class FolderCreate(FolderBase):
    """폴더 생성 모델"""
    pass

class FolderInDB(FolderBase):
    """DB 저장 폴더 모델"""
    id: str = Field(..., alias="_id")
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True
EOF

# models/document.py 파일 생성
cat > models/document.py << 'EOF'
"""
문서 모델 정의
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict

class DocumentBase(BaseModel):
    """문서 기본 모델"""
    folder_id: str = Field(..., description="폴더 ID")
    chunk_id: str = Field(..., description="청크 고유 ID")
    sequence: int = Field(..., description="청크 순서")
    text: str = Field(..., description="텍스트 내용")
    metadata: Optional[Dict] = Field(default_factory=dict, description="메타데이터")

class DocumentCreate(DocumentBase):
    """문서 생성 모델"""
    text_embedding: Optional[List[float]] = Field(None, description="임베딩 벡터")

class DocumentInDB(DocumentBase):
    """DB 저장 문서 모델"""
    id: str = Field(..., alias="_id")
    text_embedding: List[float] = Field(..., description="임베딩 벡터")
    created_at: datetime
    
    class Config:
        populate_by_name = True

class DocumentSearch(BaseModel):
    """문서 검색 결과 모델"""
    document: DocumentInDB
    score: float = Field(..., description="유사도 점수")
EOF

# models/label.py 파일 생성
cat > models/label.py << 'EOF'
"""
라벨 모델 정의
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class LabelBase(BaseModel):
    """라벨 기본 모델"""
    document_id: str = Field(..., description="문서 ID")
    folder_id: str = Field(..., description="폴더 ID")
    main_topic: str = Field(..., description="주요 주제")
    tags: List[str] = Field(default_factory=list, description="태그 목록")
    category: Optional[str] = Field(None, description="카테고리")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")

class LabelCreate(LabelBase):
    """라벨 생성 모델"""
    pass

class LabelInDB(LabelBase):
    """DB 저장 라벨 모델"""
    id: str = Field(..., alias="_id")
    created_at: datetime
    
    class Config:
        populate_by_name = True
EOF

# models/qapair.py 파일 생성
cat > models/qapair.py << 'EOF'
"""
QA 쌍 모델 정의
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class QuestionType(str, Enum):
    """질문 유형"""
    FACTOID = "factoid"
    REASONING = "reasoning"
    SUMMARY = "summary"

class Difficulty(str, Enum):
    """난이도"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QAPairBase(BaseModel):
    """QA 쌍 기본 모델"""
    document_id: str = Field(..., description="문서 ID")
    folder_id: str = Field(..., description="폴더 ID")
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")
    question_type: QuestionType = Field(..., description="질문 유형")
    difficulty: Difficulty = Field(..., description="난이도")
    quiz_options: Optional[List[str]] = Field(None, description="퀴즈 선택지")
    correct_option: Optional[int] = Field(None, description="정답 인덱스")

class QAPairCreate(QAPairBase):
    """QA 쌍 생성 모델"""
    pass

class QAPairInDB(QAPairBase):
    """DB 저장 QA 쌍 모델"""
    id: str = Field(..., alias="_id")
    created_at: datetime
    
    class Config:
        populate_by_name = True
EOF

# models/recommendation.py 파일 생성
cat > models/recommendation.py << 'EOF'
"""
추천 모델 정의
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

class ContentType(str, Enum):
    """콘텐츠 유형"""
    BOOK = "book"
    MOVIE = "movie"
    VIDEO = "video"

class RecommendationBase(BaseModel):
    """추천 기본 모델"""
    keyword: str = Field(..., description="키워드")
    content_type: ContentType = Field(..., description="콘텐츠 유형")
    content_id: str = Field(..., description="콘텐츠 ID")
    title: str = Field(..., description="제목")
    description: Optional[str] = Field(None, description="설명")
    source: str = Field(..., description="출처")
    metadata: Optional[Dict] = Field(default_factory=dict, description="메타데이터")

class RecommendationCreate(RecommendationBase):
    """추천 생성 모델"""
    pass

class RecommendationInDB(RecommendationBase):
    """DB 저장 추천 모델"""
    id: str = Field(..., alias="_id")
    created_at: datetime
    
    class Config:
        populate_by_name = True
EOF

# data_processing/__init__.py
touch data_processing/__init__.py

# data_processing/loader.py 파일 생성
cat > data_processing/loader.py << 'EOF'
"""
문서 로더 모듈
다양한 포맷의 문서를 로드하는 기능
"""
from typing import List, Dict
from pathlib import Path
import pypdf
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """문서 로더 클래스"""
    
    @staticmethod
    async def load_pdf(file_path: Path) -> str:
        """PDF 파일 로드"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PDF 로드 실패: {e}")
            raise
        return text
    
    @staticmethod
    async def load_docx(file_path: Path) -> str:
        """DOCX 파일 로드"""
        text = ""
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"DOCX 로드 실패: {e}")
            raise
        return text
    
    @staticmethod
    async def load_txt(file_path: Path) -> str:
        """텍스트 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"TXT 로드 실패: {e}")
            raise
    
    @staticmethod
    async def load_html(file_path: Path) -> str:
        """HTML 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"HTML 로드 실패: {e}")
            raise
    
    async def load_document(self, file_path: Path) -> Dict[str, str]:
        """파일 확장자에 따라 적절한 로더 선택"""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            text = await self.load_pdf(file_path)
        elif ext == '.docx':
            text = await self.load_docx(file_path)
        elif ext == '.txt':
            text = await self.load_txt(file_path)
        elif ext in ['.html', '.htm']:
            text = await self.load_html(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        
        return {
            "text": text,
            "source": str(file_path),
            "file_type": ext[1:]
        }
EOF

# data_processing/preprocessor.py 파일 생성
cat > data_processing/preprocessor.py << 'EOF'
"""
전처리 모듈
텍스트 클린징 및 정규화
"""
import re
from typing import List
from utils.logger import get_logger

logger = get_logger(__name__)

class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self):
        # 제거할 패턴들
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'),
            'special_chars': re.compile(r'[^가-힣a-zA-Z0-9\\s\\.\\,\\!\\?]'),
            'multiple_spaces': re.compile(r'\\s+'),
            'multiple_newlines': re.compile(r'\\n+')
        }
    
    def remove_html_tags(self, text: str) -> str:
        """HTML 태그 제거"""
        return self.patterns['html_tags'].sub('', text)
    
    def remove_urls(self, text: str) -> str:
        """URL 제거"""
        return self.patterns['urls'].sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """이메일 주소 제거"""
        return self.patterns['emails'].sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = self.patterns['multiple_newlines'].sub('\\n', text)
        return text.strip()
    
    def remove_special_characters(self, text: str) -> str:
        """특수문자 제거 (한글, 영문, 숫자, 기본 문장부호만 유지)"""
        return self.patterns['special_chars'].sub(' ', text)
    
    def preprocess(self, text: str) -> str:
        """전체 전처리 파이프라인"""
        # HTML 태그 제거
        text = self.remove_html_tags(text)
        
        # URL 제거
        text = self.remove_urls(text)
        
        # 이메일 제거
        text = self.remove_emails(text)
        
        # 특수문자 제거
        text = self.remove_special_characters(text)
        
        # 공백 정규화
        text = self.normalize_whitespace(text)
        
        logger.info(f"전처리 완료: {len(text)} 문자")
        return text
EOF

# data_processing/chunker.py 파일 생성
cat > data_processing/chunker.py << 'EOF'
"""
청킹 모듈
문서를 적절한 크기로 분할
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class TextChunker:
    """텍스트 청킹 클래스"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # LangChain 텍스트 분할기 초기화
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\\n\\n", "\\n", ".", "!", "?", " ", ""],
            length_function=len
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """텍스트를 청크로 분할"""
        # 텍스트 분할
        chunks = self.splitter.split_text(text)
        
        # 청크 메타데이터 추가
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "text": chunk,
                "sequence": i,
                "metadata": {
                    "chunk_method": "recursive",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "total_chunks": len(chunks)
                }
            }
            
            # 추가 메타데이터가 있으면 병합
            if metadata:
                chunk_doc["metadata"].update(metadata)
            
            chunk_docs.append(chunk_doc)
        
        logger.info(f"청킹 완료: {len(chunks)}개 청크 생성")
        return chunk_docs
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[Dict]:
        """문장 단위로 청킹"""
        # 간단한 문장 분리 (실제로는 더 정교한 방법 필요)
        sentences = text.split('. ')
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i+sentences_per_chunk]
            chunk_text = '. '.join(chunk_sentences)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            
            chunks.append({
                "text": chunk_text,
                "sequence": i // sentences_per_chunk,
                "metadata": {
                    "chunk_method": "sentences",
                    "sentences_per_chunk": sentences_per_chunk
                }
            })
        
        return chunks
EOF

# data_processing/embedder.py 파일 생성
cat > data_processing/embedder.py << 'EOF'
"""
임베딩 모듈
OpenAI 임베딩 API를 사용한 텍스트 벡터화
"""
from typing import List, Dict
import asyncio
from openai import AsyncOpenAI
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class TextEmbedder:
    """텍스트 임베딩 클래스"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL
    
    async def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    async def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """배치 텍스트 임베딩"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"임베딩 생성 진행: {len(embeddings)}/{len(texts)}")
            except Exception as e:
                logger.error(f"배치 임베딩 실패: {e}")
                raise
        
        return embeddings
    
    async def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """문서 리스트에 임베딩 추가"""
        texts = [doc["text"] for doc in documents]
        embeddings = await self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc["text_embedding"] = embedding
        
        logger.info(f"문서 임베딩 완료: {len(documents)}개")
        return documents
EOF

# ai_processing/__init__.py
touch ai_processing/__init__.py

# ai_processing/llm_client.py 파일 생성
cat > ai_processing/llm_client.py << 'EOF'
"""
LLM 클라이언트 모듈
OpenAI GPT-4o-mini API 인터페이스
"""
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    """LLM 클라이언트 클래스"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """LLM 응답 생성"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 생성 실패: {e}")
            raise
    
    async def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """컨텍스트 기반 응답 생성"""
        prompt = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.

컨텍스트:
{context}

질문: {query}

답변:"""
        
        return await self.generate(prompt, system_prompt)
EOF

# ai_processing/labeler.py 파일 생성
cat > ai_processing/labeler.py << 'EOF'
"""
라벨링 모듈
LLM을 사용한 자동 라벨링 및 키워드 추출
"""
from typing import List, Dict
import json
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class AutoLabeler:
    """자동 라벨링 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    async def extract_labels(self, text: str) -> Dict:
        """텍스트에서 라벨 추출"""
        prompt = f"""다음 텍스트를 분석하여 주요 정보를 추출해주세요.

텍스트:
{text}

다음 JSON 형식으로 응답해주세요:
{{
    "main_topic": "주요 주제 (한 문장)",
    "tags": ["태그1", "태그2", "태그3"],
    "category": "카테고리명",
    "summary": "한 문장 요약"
}}
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # JSON 파싱
            labels = json.loads(response)
            labels["confidence"] = 0.9  # 기본 신뢰도
            
            return labels
        except Exception as e:
            logger.error(f"라벨 추출 실패: {e}")
            # 기본값 반환
            return {
                "main_topic": "알 수 없음",
                "tags": [],
                "category": "기타",
                "summary": text[:100] + "...",
                "confidence": 0.5
            }
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출"""
        prompt = f"""다음 텍스트에서 가장 중요한 키워드를 {max_keywords}개 추출해주세요.

텍스트:
{text}

키워드를 쉼표로 구분하여 나열해주세요:"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            # 키워드 파싱
            keywords = [k.strip() for k in response.split(',')]
            return keywords[:max_keywords]
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
EOF

# ai_processing/qa_generator.py 파일 생성
cat > ai_processing/qa_generator.py << 'EOF'
"""
QA 생성 모듈
LLM을 사용한 질문-답변 쌍 및 퀴즈 생성
"""
from typing import List, Dict
import json
from ai_processing.llm_client import LLMClient
from models.qapair import QuestionType, Difficulty
from utils.logger import get_logger

logger = get_logger(__name__)

class QAGenerator:
    """QA 생성 클래스"""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    async def generate_qa_pairs(
        self,
        text: str,
        num_pairs: int = 3
    ) -> List[Dict]:
        """텍스트에서 QA 쌍 생성"""
        prompt = f"""다음 텍스트를 읽고 {num_pairs}개의 질문-답변 쌍을 생성해주세요.

텍스트:
{text}

다음 JSON 배열 형식으로 응답해주세요:
[
    {{
        "question": "질문 내용",
        "answer": "답변 내용",
        "question_type": "factoid|reasoning|summary 중 하나",
        "difficulty": "easy|medium|hard 중 하나"
    }}
]
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.5,
                max_tokens=800
            )
            
            # JSON 파싱
            qa_pairs = json.loads(response)
            return qa_pairs
        except Exception as e:
            logger.error(f"QA 생성 실패: {e}")
            return []
    
    async def generate_quiz(
        self,
        text: str,
        quiz_type: str = "multiple_choice"
    ) -> Dict:
        """퀴즈 생성"""
        prompt = f"""다음 텍스트를 읽고 객관식 퀴즈를 생성해주세요.

텍스트:
{text}

다음 JSON 형식으로 응답해주세요:
{{
    "question": "퀴즈 질문",
    "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
    "correct_option": 0,
    "explanation": "정답 설명",
    "difficulty": "easy|medium|hard 중 하나"
}}
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=0.5,
                max_tokens=400
            )
            
            # JSON 파싱
            quiz = json.loads(response)
            return quiz
        except Exception as e:
            logger.error(f"퀴즈 생성 실패: {e}")
            return {}
EOF

# retrieval/__init__.py
touch retrieval/__init__.py

# retrieval/vector_search.py 파일 생성
cat > retrieval/vector_search.py << 'EOF'
"""
벡터 검색 모듈
MongoDB 벡터 검색 기능
"""
from typing import List, Dict, Optional
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase
from data_processing.embedder import TextEmbedder
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorSearch:
    """벡터 검색 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.embedder = TextEmbedder()
        self.collection = db.documents
    
    async def create_vector_index(self):
        """벡터 검색 인덱스 생성"""
        try:
            # MongoDB Atlas Search 인덱스 생성
            # 실제 환경에서는 Atlas UI 또는 별도 스크립트로 생성
            await self.collection.create_index([("text_embedding", "2dsphere")])
            logger.info("벡터 인덱스 생성 완료")
        except Exception as e:
            logger.error(f"벡터 인덱스 생성 실패: {e}")
    
    async def search_similar(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """유사도 기반 검색"""
        # 쿼리 임베딩
        query_embedding = await self.embedder.embed_text(query)
        
        # MongoDB 집계 파이프라인
        pipeline = []
        
        # 필터 적용
        if filter_dict:
            pipeline.append({"$match": filter_dict})
        
        # 벡터 유사도 검색 (코사인 유사도)
        # MongoDB 6.0+ 에서는 $vectorSearch 사용 가능
        # 여기서는 간단한 예시로 모든 문서를 가져와서 계산
        documents = []
        cursor = self.collection.find(filter_dict or {})
        
        async for doc in cursor:
            if "text_embedding" in doc:
                # 코사인 유사도 계산
                similarity = self._cosine_similarity(
                    query_embedding,
                    doc["text_embedding"]
                )
                documents.append({
                    "document": doc,
                    "score": similarity
                })
        
        # 유사도 순으로 정렬
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        # 상위 k개 반환
        return documents[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
EOF

# retrieval/hybrid_search.py 파일 생성
cat > retrieval/hybrid_search.py << 'EOF'
"""
하이브리드 검색 모듈
벡터 검색과 키워드 검색을 결합
"""
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from retrieval.vector_search import VectorSearch
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridSearch:
    """하이브리드 검색 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.vector_search = VectorSearch(db)
        self.documents = db.documents
        self.labels = db.labels
    
    async def search(
        self,
        query: str,
        k: int = 5,
        folder_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """하이브리드 검색 실행"""
        # 필터 조건 생성
        filter_dict = {}
        if folder_id:
            filter_dict["folder_id"] = folder_id
        
        # 벡터 검색
        vector_results = await self.vector_search.search_similar(
            query, k=k*2, filter_dict=filter_dict
        )
        
        # 라벨 기반 필터링
        if categories or tags:
            filtered_results = []
            for result in vector_results:
                doc_id = str(result["document"]["_id"])
                
                # 라벨 조회
                label = await self.labels.find_one({"document_id": doc_id})
                
                if label:
                    # 카테고리 필터
                    if categories and label.get("category") not in categories:
                        continue
                    
                    # 태그 필터
                    if tags:
                        doc_tags = label.get("tags", [])
                        if not any(tag in doc_tags for tag in tags):
                            continue
                
                filtered_results.append(result)
            
            vector_results = filtered_results
        
        # 최종 결과 반환
        return vector_results[:k]
    
    async def search_by_keyword(
        self,
        keyword: str,
        k: int = 5
    ) -> List[Dict]:
        """키워드 기반 검색"""
        # 텍스트 검색
        text_filter = {"text": {"$regex": keyword, "$options": "i"}}
        
        documents = []
        cursor = self.documents.find(text_filter).limit(k)
        
        async for doc in cursor:
            documents.append({
                "document": doc,
                "score": 1.0  # 키워드 매칭은 동일한 점수
            })
        
        return documents
EOF

# retrieval/context_builder.py 파일 생성
cat > retrieval/context_builder.py << 'EOF'
"""
컨텍스트 빌더 모듈
검색 결과를 LLM용 컨텍스트로 변환
"""
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

class ContextBuilder:
    """컨텍스트 빌더 클래스"""
    
    def build_context(
        self,
        search_results: List[Dict],
        max_tokens: int = 2000
    ) -> str:
        """검색 결과를 컨텍스트로 변환"""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(search_results):
            doc = result["document"]
            score = result["score"]
            
            # 문서 정보 포맷팅
            doc_context = f"""[문서 {i+1}] (유사도: {score:.3f})
{doc['text']}
"""
            
            # 토큰 수 추정 (대략 4글자 = 1토큰)
            estimated_tokens = len(doc_context) // 4
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            context_parts.append(doc_context)
            current_tokens += estimated_tokens
        
        return "\\n".join(context_parts)
    
    def build_qa_context(
        self,
        qa_pairs: List[Dict]
    ) -> str:
        """QA 쌍을 컨텍스트로 변환"""
        context_parts = []
        
        for qa in qa_pairs:
            qa_context = f"""Q: {qa['question']}
A: {qa['answer']}
"""
            context_parts.append(qa_context)
        
        return "\\n".join(context_parts)
    
    def build_full_context(
        self,
        search_results: List[Dict],
        qa_pairs: List[Dict] = None,
        additional_info: Dict = None
    ) -> str:
        """전체 컨텍스트 생성"""
        parts = []
        
        # 검색 결과 컨텍스트
        if search_results:
            parts.append("=== 관련 문서 ===")
            parts.append(self.build_context(search_results))
        
        # QA 컨텍스트
        if qa_pairs:
            parts.append("\\n=== 관련 질의응답 ===")
            parts.append(self.build_qa_context(qa_pairs))
        
        # 추가 정보
        if additional_info:
            parts.append("\\n=== 추가 정보 ===")
            for key, value in additional_info.items():
                parts.append(f"{key}: {value}")
        
        return "\\n".join(parts)
EOF

# api/__init__.py
touch api/__init__.py

# api/routers/__init__.py
touch api/routers/__init__.py

# api/routers/query.py 파일 생성
cat > api/routers/query.py << 'EOF'
"""
질의응답 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from api.chains.query_chain import QueryChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    """질의 요청 모델"""
    query: str
    folder_id: Optional[str] = None
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    """질의 응답 모델"""
    answer: str
    sources: Optional[List[dict]] = None
    confidence: float

@router.post("/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """질의 처리 엔드포인트"""
    try:
        db = await get_database()
        query_chain = QueryChain(db)
        
        # 질의 처리
        result = await query_chain.process(
            query=request.query,
            folder_id=request.folder_id,
            top_k=request.top_k
        )
        
        # 응답 생성
        response = QueryResponse(
            answer=result["answer"],
            sources=result.get("sources") if request.include_sources else None,
            confidence=result.get("confidence", 0.9)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"질의 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# api/routers/summary.py 파일 생성
cat > api/routers/summary.py << 'EOF'
"""
요약 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.chains.summary_chain import SummaryChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class SummaryRequest(BaseModel):
    """요약 요청 모델"""
    document_ids: Optional[List[str]] = None
    folder_id: Optional[str] = None
    summary_type: str = "brief"  # brief, detailed, bullets

class SummaryResponse(BaseModel):
    """요약 응답 모델"""
    summary: str
    document_count: int
    summary_type: str

@router.post("/", response_model=SummaryResponse)
async def create_summary(request: SummaryRequest):
    """요약 생성 엔드포인트"""
    try:
        db = await get_database()
        summary_chain = SummaryChain(db)
        
        # 요약 생성
        result = await summary_chain.process(
            document_ids=request.document_ids,
            folder_id=request.folder_id,
            summary_type=request.summary_type
        )
        
        return SummaryResponse(
            summary=result["summary"],
            document_count=result["document_count"],
            summary_type=request.summary_type
        )
        
    except Exception as e:
        logger.error(f"요약 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# api/routers/quiz.py 파일 생성
cat > api/routers/quiz.py << 'EOF'
"""
퀴즈 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.chains.quiz_chain import QuizChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class QuizRequest(BaseModel):
    """퀴즈 요청 모델"""
    topic: Optional[str] = None
    folder_id: Optional[str] = None
    difficulty: str = "medium"
    count: int = 5
    quiz_type: str = "multiple_choice"

class QuizItem(BaseModel):
    """퀴즈 항목 모델"""
    question: str
    options: Optional[List[str]] = None
    correct_option: Optional[int] = None
    difficulty: str
    explanation: Optional[str] = None

class QuizResponse(BaseModel):
    """퀴즈 응답 모델"""
    quizzes: List[QuizItem]
    topic: Optional[str]
    total_count: int

@router.post("/", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    """퀴즈 생성 엔드포인트"""
    try:
        db = await get_database()
        quiz_chain = QuizChain(db)
        
        # 퀴즈 생성
        result = await quiz_chain.process(
            topic=request.topic,
            folder_id=request.folder_id,
            difficulty=request.difficulty,
            count=request.count,
            quiz_type=request.quiz_type
        )
        
        # 퀴즈 항목 변환
        quiz_items = []
        for quiz in result["quizzes"]:
            quiz_items.append(QuizItem(
                question=quiz["question"],
                options=quiz.get("options"),
                correct_option=quiz.get("correct_option"),
                difficulty=quiz["difficulty"],
                explanation=quiz.get("explanation")
            ))
        
        return QuizResponse(
            quizzes=quiz_items,
            topic=request.topic,
            total_count=len(quiz_items)
        )
        
    except Exception as e:
        logger.error(f"퀴즈 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# api/routers/keywords.py 파일 생성
cat > api/routers/keywords.py << 'EOF'
"""
키워드 추출 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ai_processing.labeler import AutoLabeler
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class KeywordsRequest(BaseModel):
    """키워드 추출 요청 모델"""
    text: str
    max_keywords: int = 10

class KeywordsResponse(BaseModel):
    """키워드 추출 응답 모델"""
    keywords: List[str]
    count: int

@router.post("/", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest):
    """키워드 추출 엔드포인트"""
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
EOF

# api/routers/mindmap.py 파일 생성
cat > api/routers/mindmap.py << 'EOF'
"""
마인드맵 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class MindmapRequest(BaseModel):
    """마인드맵 요청 모델"""
    root_keyword: str
    depth: int = 3
    max_nodes: int = 20

class MindmapNode(BaseModel):
    """마인드맵 노드 모델"""
    id: str
    label: str
    level: int
    children: List[str] = []

class MindmapEdge(BaseModel):
    """마인드맵 엣지 모델"""
    source: str
    target: str
    weight: float = 1.0

class MindmapResponse(BaseModel):
    """마인드맵 응답 모델"""
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]
    root_id: str

@router.post("/", response_model=MindmapResponse)
async def generate_mindmap(request: MindmapRequest):
    """마인드맵 생성 엔드포인트"""
    try:
        db = await get_database()
        labels_collection = db.labels
        
        # 루트 키워드와 관련된 라벨 검색
        related_labels = await labels_collection.find(
            {"tags": {"$in": [request.root_keyword]}}
        ).limit(request.max_nodes).to_list(None)
        
        # 마인드맵 구조 생성
        nodes = []
        edges = []
        
        # 루트 노드
        root_node = MindmapNode(
            id="root",
            label=request.root_keyword,
            level=0
        )
        nodes.append(root_node)
        
        # 관련 노드 추가
        for i, label in enumerate(related_labels):
            node_id = f"node_{i}"
            
            # 노드 생성
            node = MindmapNode(
                id=node_id,
                label=label["main_topic"],
                level=1
            )
            nodes.append(node)
            
            # 엣지 생성
            edge = MindmapEdge(
                source="root",
                target=node_id,
                weight=label.get("confidence", 0.5)
            )
            edges.append(edge)
        
        return MindmapResponse(
            nodes=nodes,
            edges=edges,
            root_id="root"
        )
        
    except Exception as e:
        logger.error(f"마인드맵 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# api/routers/recommend.py 파일 생성
cat > api/routers/recommend.py << 'EOF'
"""
추천 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.chains.recommend_chain import RecommendChain
from database.connection import get_database
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class RecommendRequest(BaseModel):
    """추천 요청 모델"""
    keywords: List[str]
    content_types: List[str] = ["book", "movie", "video"]
    max_items: int = 10

class RecommendItem(BaseModel):
    """추천 항목 모델"""
    title: str
    content_type: str
    description: Optional[str]
    source: str
    metadata: Dict

class RecommendResponse(BaseModel):
    """추천 응답 모델"""
    recommendations: List[RecommendItem]
    total_count: int

@router.post("/", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """콘텐츠 추천 엔드포인트"""
    try:
        db = await get_database()
        recommend_chain = RecommendChain(db)
        
        # 추천 생성
        result = await recommend_chain.process(
            keywords=request.keywords,
            content_types=request.content_types,
            max_items=request.max_items
        )
        
        # 추천 항목 변환
        recommendations = []
        for item in result["recommendations"]:
            recommendations.append(RecommendItem(
                title=item["title"],
                content_type=item["content_type"],
                description=item.get("description"),
                source=item["source"],
                metadata=item.get("metadata", {})
            ))
        
        return RecommendResponse(
            recommendations=recommendations,
            total_count=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"추천 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# api/chains/__init__.py
touch api/chains/__init__.py

# api/chains/query_chain.py 파일 생성
cat > api/chains/query_chain.py << 'EOF'
"""
질의응답 체인
LangChain을 사용한 RAG 파이프라인
"""
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from retrieval.hybrid_search import HybridSearch
from retrieval.context_builder import ContextBuilder
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryChain:
    """질의응답 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.hybrid_search = HybridSearch(db)
        self.context_builder = ContextBuilder()
        self.llm_client = LLMClient()
        
        # 프롬프트 템플릿
        self.prompt_template = """다음 컨텍스트를 참고하여 사용자의 질문에 답변해주세요.
답변은 정확하고 도움이 되도록 작성하되, 컨텍스트에 없는 내용은 추측하지 마세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    
    async def process(
        self,
        query: str,
        folder_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """질의 처리"""
        try:
            # 1. 관련 문서 검색
            search_results = await self.hybrid_search.search(
                query=query,
                k=top_k,
                folder_id=folder_id
            )
            
            # 2. 컨텍스트 구성
            context = self.context_builder.build_context(search_results)
            
            # 3. LLM 응답 생성
            prompt = self.prompt_template.format(
                context=context,
                question=query
            )
            
            answer = await self.llm_client.generate(prompt)
            
            # 4. 결과 반환
            return {
                "answer": answer,
                "sources": [
                    {
                        "text": result["document"]["text"][:200] + "...",
                        "score": result["score"]
                    }
                    for result in search_results
                ],
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"질의 처리 실패: {e}")
            raise
EOF

# api/chains/summary_chain.py 파일 생성
cat > api/chains/summary_chain.py << 'EOF'
"""
요약 체인
문서 요약 생성
"""
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from ai_processing.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class SummaryChain:
    """요약 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.llm_client = LLMClient()
        self.documents = db.documents
    
    async def process(
        self,
        document_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        summary_type: str = "brief"
    ) -> Dict:
        """요약 처리"""
        try:
            # 문서 조회
            filter_dict = {}
            if document_ids:
                filter_dict["_id"] = {"$in": document_ids}
            elif folder_id:
                filter_dict["folder_id"] = folder_id
            else:
                raise ValueError("document_ids 또는 folder_id 필요")
            
            documents = await self.documents.find(filter_dict).to_list(None)
            
            if not documents:
                return {
                    "summary": "요약할 문서가 없습니다.",
                    "document_count": 0
                }
            
            # 텍스트 결합
            combined_text = "\n\n".join([doc["text"] for doc in documents])
            
            # 프롬프트 선택
            if summary_type == "brief":
                prompt = f"다음 텍스트를 1-2문장으로 간단히 요약해주세요:\n\n{combined_text}"
            elif summary_type == "detailed":
                prompt = f"다음 텍스트를 상세하게 요약해주세요:\n\n{combined_text}"
            else:  # bullets
                prompt = f"다음 텍스트의 핵심 내용을 불릿 포인트로 정리해주세요:\n\n{combined_text}"
            
            # 요약 생성
            summary = await self.llm_client.generate(prompt, max_tokens=500)
            
            return {
                "summary": summary,
                "document_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"요약 처리 실패: {e}")
            raise
EOF

# api/chains/quiz_chain.py 파일 생성
cat > api/chains/quiz_chain.py << 'EOF'
"""
퀴즈 체인
퀴즈 생성 파이프라인
"""
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from ai_processing.qa_generator import QAGenerator
from retrieval.hybrid_search import HybridSearch
from utils.logger import get_logger

logger = get_logger(__name__)

class QuizChain:
    """퀴즈 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.qa_generator = QAGenerator()
        self.hybrid_search = HybridSearch(db)
        self.qapairs = db.qapairs
    
    async def process(
        self,
        topic: Optional[str] = None,
        folder_id: Optional[str] = None,
        difficulty: str = "medium",
        count: int = 5,
        quiz_type: str = "multiple_choice"
    ) -> Dict:
        """퀴즈 처리"""
        try:
            quizzes = []
            
            # 기존 퀴즈 조회
            filter_dict = {"difficulty": difficulty}
            if folder_id:
                filter_dict["folder_id"] = folder_id
            
            existing_quizzes = await self.qapairs.find(
                filter_dict
            ).limit(count).to_list(None)
            
            # 기존 퀴즈가 충분하면 반환
            if len(existing_quizzes) >= count:
                for quiz in existing_quizzes[:count]:
                    quizzes.append({
                        "question": quiz["question"],
                        "options": quiz.get("quiz_options"),
                        "correct_option": quiz.get("correct_option"),
                        "difficulty": quiz["difficulty"],
                        "explanation": quiz.get("answer")
                    })
            else:
                # 새로운 퀴즈 생성 필요
                if topic:
                    # 주제 관련 문서 검색
                    search_results = await self.hybrid_search.search(
                        query=topic,
                        k=3,
                        folder_id=folder_id
                    )
                    
                    # 각 문서에서 퀴즈 생성
                    for result in search_results:
                        text = result["document"]["text"]
                        quiz = await self.qa_generator.generate_quiz(
                            text=text,
                            quiz_type=quiz_type
                        )
                        
                        if quiz:
                            quiz["difficulty"] = difficulty
                            quizzes.append(quiz)
                            
                            if len(quizzes) >= count:
                                break
            
            return {
                "quizzes": quizzes[:count]
            }
            
        except Exception as e:
            logger.error(f"퀴즈 처리 실패: {e}")
            raise
EOF

# api/chains/recommend_chain.py 파일 생성
cat > api/chains/recommend_chain.py << 'EOF'
"""
추천 체인
콘텐츠 추천 생성
"""
from typing import Dict, List
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

class RecommendChain:
    """추천 체인 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.recommendations = db.recommendations
    
    async def process(
        self,
        keywords: List[str],
        content_types: List[str] = ["book", "movie", "video"],
        max_items: int = 10
    ) -> Dict:
        """추천 처리"""
        try:
            recommendations = []
            
            # 키워드별로 추천 검색
            for keyword in keywords:
                filter_dict = {
                    "keyword": keyword,
                    "content_type": {"$in": content_types}
                }
                
                items = await self.recommendations.find(
                    filter_dict
                ).limit(max_items // len(keywords)).to_list(None)
                
                for item in items:
                    recommendations.append({
                        "title": item["title"],
                        "content_type": item["content_type"],
                        "description": item.get("description"),
                        "source": item["source"],
                        "metadata": item.get("metadata", {})
                    })
            
            # 추천이 없는 경우 더미 데이터 생성
            if not recommendations:
                recommendations = [
                    {
                        "title": f"{keyword} 관련 추천 콘텐츠",
                        "content_type": "book",
                        "description": "추천 시스템이 곧 업데이트됩니다.",
                        "source": "internal",
                        "metadata": {}
                    }
                    for keyword in keywords[:3]
                ]
            
            return {
                "recommendations": recommendations[:max_items]
            }
            
        except Exception as e:
            logger.error(f"추천 처리 실패: {e}")
            raise
EOF

# database/__init__.py
touch database/__init__.py

# database/connection.py 파일 생성
cat > database/connection.py << 'EOF'
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
        if self.client:
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
    if not db_connection.db:
        await init_db()
    return db_connection.db
EOF

# database/operations.py 파일 생성
cat > database/operations.py << 'EOF'
"""
데이터베이스 작업 모듈
공통 CRUD 작업
"""
from typing import Dict, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseOperations:
    """데이터베이스 작업 클래스"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    async def insert_one(
        self,
        collection_name: str,
        document: Dict
    ) -> str:
        """단일 문서 삽입"""
        try:
            # 타임스탬프 추가
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.insert_one(document)
            
            logger.info(f"{collection_name}에 문서 삽입: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"문서 삽입 실패: {e}")
            raise
    
    async def insert_many(
        self,
        collection_name: str,
        documents: List[Dict]
    ) -> List[str]:
        """다중 문서 삽입"""
        try:
            # 타임스탬프 추가
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.insert_many(documents)
            
            logger.info(f"{collection_name}에 {len(result.inserted_ids)}개 문서 삽입")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"다중 문서 삽입 실패: {e}")
            raise
    
    async def find_one(
        self,
        collection_name: str,
        filter_dict: Dict
    ) -> Optional[Dict]:
        """단일 문서 조회"""
        try:
            collection = self.db[collection_name]
            document = await collection.find_one(filter_dict)
            return document
            
        except Exception as e:
            logger.error(f"문서 조회 실패: {e}")
            raise
    
    async def find_many(
        self,
        collection_name: str,
        filter_dict: Dict,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict]:
        """다중 문서 조회"""
        try:
            collection = self.db[collection_name]
            cursor = collection.find(filter_dict)
            
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(None)
            return documents
            
        except Exception as e:
            logger.error(f"다중 문서 조회 실패: {e}")
            raise
    
    async def update_one(
        self,
        collection_name: str,
        filter_dict: Dict,
        update_dict: Dict
    ) -> bool:
        """단일 문서 업데이트"""
        try:
            # 업데이트 타임스탬프 추가
            update_dict["$set"] = update_dict.get("$set", {})
            update_dict["$set"]["updated_at"] = datetime.utcnow()
            
            collection = self.db[collection_name]
            result = await collection.update_one(filter_dict, update_dict)
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"문서 업데이트 실패: {e}")
            raise
    
    async def delete_one(
        self,
        collection_name: str,
        filter_dict: Dict
    ) -> bool:
        """단일 문서 삭제"""
        try:
            collection = self.db[collection_name]
            result = await collection.delete_one(filter_dict)
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            raise
EOF

# utils/__init__.py
touch utils/__init__.py

# utils/logger.py 파일 생성
cat > utils/logger.py << 'EOF'
"""
로깅 모듈
애플리케이션 로깅 설정
"""
import sys
from loguru import logger
from config.settings import settings

def setup_logger():
    """로거 설정"""
    # 기본 로거 제거
    logger.remove()
    
    # 콘솔 출력 설정
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 파일 출력 설정
    logger.add(
        "logs/app.log",
        level=settings.LOG_LEVEL,
        rotation="1 day",
        retention="7 days",
        compression="zip"
    )
    
    return logger

def get_logger(name: str):
    """모듈별 로거 반환"""
    return logger.bind(name=name)
EOF

# utils/validators.py 파일 생성
cat > utils/validators.py << 'EOF'
"""
검증 모듈
입력값 검증 유틸리티
"""
from typing import Any, Dict, Optional
import re

def validate_folder_name(name: str) -> bool:
    """폴더명 검증"""
    if not name or len(name) > 100:
        return False
    
    # 특수문자 제한
    pattern = r'^[가-힣a-zA-Z0-9\s_-]+
    return bool(re.match(pattern, name))

def validate_chunk_size(size: int) -> bool:
    """청크 크기 검증"""
    return 100 <= size <= 2000

def validate_top_k(k: int) -> bool:
    """검색 결과 수 검증"""
    return 1 <= k <= 20

def validate_difficulty(difficulty: str) -> bool:
    """난이도 검증"""
    return difficulty in ["easy", "medium", "hard"]

def validate_content_type(content_type: str) -> bool:
    """콘텐츠 타입 검증"""
    return content_type in ["book", "movie", "video"]

def validate_file_extension(filename: str) -> bool:
    """파일 확장자 검증"""
    allowed_extensions = ['.pdf', '.txt', '.docx', '.html', '.htm']
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)
EOF

# tests/__init__.py
touch tests/__init__.py

# tests/test_data_processing.py 파일 생성
cat > tests/test_data_processing.py << 'EOF'
"""
데이터 처리 테스트
"""
import pytest
from data_processing.preprocessor import TextPreprocessor
from data_processing.chunker import TextChunker

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

@pytest.fixture
def chunker():
    return TextChunker(chunk_size=100, chunk_overlap=20)

def test_preprocessor_html_removal(preprocessor):
    """HTML 태그 제거 테스트"""
    text = "<p>테스트 <b>텍스트</b></p>"
    result = preprocessor.remove_html_tags(text)
    assert result == "테스트 텍스트"

def test_preprocessor_url_removal(preprocessor):
    """URL 제거 테스트"""
    text = "방문하세요 https://example.com 여기를"
    result = preprocessor.remove_urls(text)
    assert "https://example.com" not in result

def test_chunker_text_splitting(chunker):
    """텍스트 분할 테스트"""
    text = "안녕하세요. " * 50  # 긴 텍스트
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 120 for chunk in chunks)  # 오버랩 포함
EOF

# tests/test_retrieval.py 파일 생성
cat > tests/test_retrieval.py << 'EOF'
"""
검색 기능 테스트
"""
import pytest
from retrieval.context_builder import ContextBuilder

@pytest.fixture
def context_builder():
    return ContextBuilder()

def test_context_building(context_builder):
    """컨텍스트 생성 테스트"""
    search_results = [
        {
            "document": {"text": "첫 번째 문서"},
            "score": 0.9
        },
        {
            "document": {"text": "두 번째 문서"},
            "score": 0.8
        }
    ]
    
    context = context_builder.build_context(search_results)
    
    assert "첫 번째 문서" in context
    assert "두 번째 문서" in context
    assert "0.9" in context
    assert "0.8" in context
EOF

# tests/test_api.py 파일 생성
cat > tests/test_api.py << 'EOF'
"""
API 엔드포인트 테스트
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "RAG 백엔드 API"

def test_keywords_extraction():
    """키워드 추출 테스트"""
    response = client.post(
        "/keywords/",
        json={
            "text": "인공지능과 머신러닝은 현대 기술의 핵심입니다.",
            "max_keywords": 5
        }
    )
    
    # API 키가 설정되어 있지 않으면 테스트 건너뛰기
    if response.status_code == 500:
        pytest.skip("OpenAI API 키가 설정되지 않음")
    
    assert response.status_code == 200
    data = response.json()
    assert "keywords" in data
    assert isinstance(data["keywords"], list)
EOF

# 실행 권한 부여
chmod +x create_project.sh

echo "프로젝트 생성 완료!"
echo "다음 명령으로 프로젝트를 시작하세요:"
echo "cd rag-backend"
echo "cp .env.example .env"
echo "# .env 파일에 OpenAI API 키와 MongoDB URI 설정"
echo "pip install -r requirements.txt"
echo "python main.py"