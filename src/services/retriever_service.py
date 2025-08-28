from typing import List, Union
from langchain_core.documents import Document

class RetrieverService:
    """Interfaz para servicios de recuperación, ahora con tipado flexible."""
    
    def filter_retrieve(self, must_conditions: list[dict], query_text: Union[str, list[str]] = "") -> List[Document]:
        """
        Define el contrato para la búsqueda filtrada.
        Acepta una lista de condiciones y un texto de búsqueda que puede ser
        un string simple o una lista de keywords.
        """
        raise NotImplementedError
