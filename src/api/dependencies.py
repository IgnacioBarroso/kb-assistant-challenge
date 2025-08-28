from fastapi import Request
from src.services.rag_service import RAGService

def get_rag_service(request: Request) -> RAGService:
    """
    Función de dependencia de FastAPI para obtener la instancia de RAGService.
    FastAPI se encargará de inyectar el 'request' de la aplicación.
    """
    return request.app.state.rag_service
