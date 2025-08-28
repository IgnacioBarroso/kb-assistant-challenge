from typing import List

class DocumentLoaderService:
    """
    Define el contrato (interfaz) para todos los servicios de carga de documentos.
    Cualquier clase que cargue documentos debe heredar de esta y
    implementar el método load_documents.
    """
    def load_documents(self) -> List[dict]:
        """
        Carga documentos desde una fuente y los devuelve como una lista de diccionarios.
        """
        raise NotImplementedError
