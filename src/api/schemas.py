from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str
    top_k: int = 10
    attach_documents: bool = False

class RetrievedDoc(BaseModel):
    """Modelo explícito para un documento recuperado en la respuesta de la API."""
    id: str = Field(description="El ID único del documento en la base de datos vectorial (Qdrant).")
    page_content: str = Field(description="El contenido textual del documento/escena.")
    metadata: Dict[str, Any] = Field(description="Metadatos asociados, como número de escena, personajes, etc.")

class AskResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources_used: List[str] = []
    reasoning: Optional[str] = None
    retrieved_documents: Optional[List[RetrievedDoc]] = None
