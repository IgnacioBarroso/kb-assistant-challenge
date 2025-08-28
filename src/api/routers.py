from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_rag_service
from src.api.schemas import AskRequest, AskResponse
from src.services.rag_service import RAGService

agent_router = APIRouter()

@agent_router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Endpoint principal para realizar consultas al sistema RAG.
    """
    try:
        query_result = await rag_service.query(
            question=request.query, 
            top_k=request.top_k, 
            attach_documents=request.attach_documents
        )
        return AskResponse(**query_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
