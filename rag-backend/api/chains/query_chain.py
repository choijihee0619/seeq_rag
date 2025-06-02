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
                        "text": result.get("chunk", {}).get("text", "")[:200] + "...",
                        "score": result["score"],
                        "filename": result.get("document", {}).get("original_filename", "알 수 없는 파일"),
                        "chunk_id": result.get("chunk", {}).get("chunk_id", "")
                    }
                    for result in search_results
                ],
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"질의 처리 실패: {e}")
            raise
