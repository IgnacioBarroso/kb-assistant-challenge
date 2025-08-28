import logging
import uuid
from typing import List, Dict, Any, Optional
from itertools import groupby
from src.services.document_loader_service import DocumentLoaderService
from kbac.loaders.matrix_script_loader import MatrixScriptLoader, Document as BaseDocument
from src.settings.config import settings


class MatrixDocumentLoaderService(DocumentLoaderService):
    def __init__(self, source_path: Optional[str] = None):
        if not source_path:
            source_path = settings.MATRIX_SCRIPT_PATH
        assert source_path is not None, "source_path no puede ser None"
        
        self.source_path = str(source_path)
        self.loader = MatrixScriptLoader(source_path=self.source_path)
        self.logger = logging.getLogger("uvicorn")

    def load_documents(self) -> List[dict]:
        self.logger.info("[Matrix RAG] Cargando y agrupando el guion (modo Regex) por escenas...")
        base_documents: List[BaseDocument] = self.loader.load()
        
        dict_documents = [doc.model_dump() for doc in base_documents]

        # --- Lógica mejorada para identificar escenas únicas ---
        # Un 'scene_id' único se genera cada vez que encontramos una nueva línea de 'location'.
        # Esto evita que escenas distintas en la misma ubicación se fusionen.
        scene_id = 0
        current_location = None
        for doc in dict_documents:
            if doc["metadata"]["text_type"] == "location":
                current_location = doc["text"]
                scene_id += 1  # Nueva escena detectada
            
            if current_location:
                doc["metadata"]["location"] = current_location
                doc["metadata"]["scene_id"] = scene_id

        # Filtramos las líneas que no pertenecen a ninguna escena y las propias líneas de ubicación
        docs_with_scene_id = [
            doc for doc in dict_documents 
            if "scene_id" in doc["metadata"] and doc["metadata"]["text_type"] != "location"
        ]

        # Agrupamos por el 'scene_id' único en lugar de por el nombre de la ubicación.
        # Es crucial ordenar antes de agrupar.
        grouped_by_scene = groupby(sorted(docs_with_scene_id, key=lambda d: d["metadata"]["scene_id"]), 
                                   key=lambda doc: doc["metadata"]["scene_id"])
        
        scene_chunks = []
        for scene_id, docs_in_scene_iter in grouped_by_scene:
            docs_in_scene = list(docs_in_scene_iter)
            # La ubicación es la misma para todos los documentos de una escena.
            location = docs_in_scene[0]["metadata"]["location"]
            
            scene_content = self._format_scene_content(docs_in_scene, location)
            # Pasamos el scene_id como scene_number para que sea el número de escena real.
            scene_metadata = self._aggregate_metadata(docs_in_scene, location, scene_id)
            
            scene_chunks.append({
                "text": scene_content,
                "page_content": scene_content,
                "metadata": scene_metadata
            })

        self.logger.info(f"[Matrix RAG] {len(scene_chunks)} chunks basados en escenas generados.")
        return scene_chunks

    def _format_scene_content(self, docs: List[dict], location: str) -> str:
        full_text = f"Location: {location}\n"
        current_char = None
        for doc in docs:
            text_type = doc["metadata"].get('text_type')
            if text_type == 'character':
                current_char = doc['text']
            elif text_type == 'dialog':
                full_text += f"{current_char or 'Unknown'}: {doc['text']}\n"
            elif text_type == 'description':
                full_text += f"Scene description: {doc['text']}\n"
        return full_text.strip()

    def _aggregate_metadata(self, docs: List[dict], location: str, scene_number: int) -> Dict[str, Any]:
        # --- Extraer personajes del texto de las líneas de tipo 'character' ---
        # La lógica anterior buscaba un metadato 'character' que no existía.
        characters = sorted(list(set(
            doc["text"].upper() for doc in docs if doc["metadata"].get("text_type") == "character"
        )))
        page_numbers = sorted(list(set(doc["metadata"].get("page_number") for doc in docs)))
        return {
            "scene_number": scene_number,
            "location": location,
            "characters": characters,
            "page_start": min(page_numbers) if page_numbers else -1,
            "page_end": max(page_numbers) if page_numbers else -1,
            "_id": str(uuid.uuid4()),
            "_collection_name": settings.QDRANT_COLLECTION
        }