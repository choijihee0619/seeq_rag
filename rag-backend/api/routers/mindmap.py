"""
마인드맵 API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from database.connection import get_database
from ai_processing.labeler import AutoLabeler
from retrieval.hybrid_search import HybridSearch
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class MindmapRequest(BaseModel):
    """마인드맵 요청 모델"""
    root_keyword: str
    depth: int = 3
    max_nodes: int = 20
    folder_id: Optional[str] = None  # 특정 폴더의 문서만 대상으로 할 때

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
        hybrid_search = HybridSearch(db)
        labeler = AutoLabeler()
        
        # 1. 루트 키워드와 관련된 문서 검색
        search_results = await hybrid_search.search(
            query=request.root_keyword,
            k=min(10, request.max_nodes),
            folder_id=request.folder_id
        )
        
        # 2. 검색된 텍스트들에서 키워드 추출
        combined_text = ""
        for result in search_results[:5]:  # 상위 5개 결과만 사용
            chunk_text = result.get("chunk", {}).get("text", "")
            if chunk_text:
                combined_text += chunk_text + "\n\n"
        
        # 텍스트가 없으면 fallback 마인드맵 생성
        if not combined_text.strip():
            return _generate_fallback_mindmap(request.root_keyword)
        
        # 3. 키워드 추출
        keywords = await labeler.extract_keywords(
            text=combined_text,
            max_keywords=request.max_nodes - 1  # 루트 제외
        )
        
        # 4. 마인드맵 구조 생성
        nodes = []
        edges = []
        
        # 루트 노드
        root_node = MindmapNode(
            id="root",
            label=request.root_keyword,
            level=0
        )
        nodes.append(root_node)
        
        # 5. 관련 키워드 노드들 추가
        for i, keyword in enumerate(keywords):
            if keyword.lower() == request.root_keyword.lower():
                continue  # 루트 키워드와 중복 제거
                
            node_id = f"node_{i}"
            
            # 노드 생성
            node = MindmapNode(
                id=node_id,
                label=keyword,
                level=1
            )
            nodes.append(node)
            
            # 엣지 생성 (루트와 연결)
            edge = MindmapEdge(
                source="root",
                target=node_id,
                weight=1.0 - (i * 0.1)  # 순서에 따라 가중치 감소
            )
            edges.append(edge)
        
        # 6. 2차 관계 생성 (depth > 1인 경우)
        if request.depth > 1 and len(keywords) > 3:
            await _add_secondary_connections(
                nodes, edges, keywords, search_results, labeler, request.depth
            )
        
        logger.info(f"마인드맵 생성 완료: {len(nodes)}개 노드, {len(edges)}개 엣지")
        
        return MindmapResponse(
            nodes=nodes,
            edges=edges,
            root_id="root"
        )
        
    except Exception as e:
        logger.error(f"마인드맵 생성 실패: {e}")
        # 에러 발생 시 기본 마인드맵 반환
        return _generate_fallback_mindmap(request.root_keyword)

async def _add_secondary_connections(
    nodes: List[MindmapNode], 
    edges: List[MindmapEdge], 
    keywords: List[str], 
    search_results: List[Dict],
    labeler: AutoLabeler,
    depth: int
):
    """2차 연결 관계 추가"""
    try:
        # 키워드 간 연관성 분석
        for i, keyword1 in enumerate(keywords[:5]):  # 상위 5개만
            for j, keyword2 in enumerate(keywords[:5]):
                if i >= j:  # 중복 방지
                    continue
                
                # 두 키워드가 함께 나타나는 텍스트 찾기
                co_occurrence_count = 0
                for result in search_results:
                    text = result.get("chunk", {}).get("text", "").lower()
                    if keyword1.lower() in text and keyword2.lower() in text:
                        co_occurrence_count += 1
                
                # 공출현이 2회 이상이면 연결
                if co_occurrence_count >= 2:
                    edge = MindmapEdge(
                        source=f"node_{i}",
                        target=f"node_{j}",
                        weight=min(0.8, co_occurrence_count * 0.2)
                    )
                    edges.append(edge)
                    
                    # 레벨 2 노드로 표시
                    for node in nodes:
                        if node.id == f"node_{j}":
                            node.level = 2
                            break
                    
    except Exception as e:
        logger.warning(f"2차 연결 생성 실패: {e}")

def _generate_fallback_mindmap(root_keyword: str) -> MindmapResponse:
    """기본 마인드맵 생성 (데이터가 없을 때)"""
    
    # 키워드별 기본 관련 개념들
    default_concepts = {
        "데이터베이스": ["테이블", "쿼리", "인덱스", "관계", "정규화"],
        "SQL": ["SELECT", "INSERT", "UPDATE", "DELETE", "JOIN"],
        "머신러닝": ["알고리즘", "훈련", "모델", "데이터", "예측"],
        "프로그래밍": ["변수", "함수", "조건문", "반복문", "객체"],
        "네트워크": ["프로토콜", "IP", "TCP", "HTTP", "라우터"]
    }
    
    # 기본 개념 선택
    concepts = default_concepts.get(
        root_keyword, 
        ["개념1", "개념2", "개념3", "개념4", "개념5"]
    )
    
    nodes = []
    edges = []
    
    # 루트 노드
    root_node = MindmapNode(
        id="root",
        label=root_keyword,
        level=0
    )
    nodes.append(root_node)
    
    # 관련 개념 노드들
    for i, concept in enumerate(concepts):
        node_id = f"node_{i}"
        
        node = MindmapNode(
            id=node_id,
            label=concept,
            level=1
        )
        nodes.append(node)
        
        edge = MindmapEdge(
            source="root",
            target=node_id,
            weight=1.0 - (i * 0.1)
        )
        edges.append(edge)
    
    return MindmapResponse(
        nodes=nodes,
        edges=edges,
        root_id="root"
    )
