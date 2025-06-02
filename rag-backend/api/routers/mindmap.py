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
