import uuid
import logging
from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from pydantic import SecretStr
from src.settings.config import settings
from src.services.retriever_service import RetrieverService

class QdrantRetrieverService(RetrieverService):
    def __init__(self):
        self.logger = logging.getLogger("uvicorn")
        try:
            self.embedding_dims = settings.QDRANT_EMBEDDING_DIMS
            self.collection_name = settings.QDRANT_COLLECTION
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            api_key = settings.OPENAI_API_KEY
            if not isinstance(api_key, SecretStr):
                api_key = SecretStr(api_key)
            
            self.embeddings = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                dimensions=self.embedding_dims,
                api_key=api_key
            )
            self.vector_store = None
            self._ensure_collection()
            self.logger.info("[Init] QdrantRetrieverService inicializado correctamente.")
        except Exception as e:
            self.logger.error(f"[Init] Error inicializando QdrantRetrieverService: {e}", exc_info=True)
            raise

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dims, distance=Distance.COSINE),
            )
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def index_documents(self, documents: List[dict]):
        self.logger.info(f"[Matrix RAG] Indexando {len(documents)} documentos en Qdrant...")
        docs = [Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})) for doc in documents]
        uuids = [doc.get("metadata", {}).get("_id", str(uuid.uuid4())) for doc in documents]
        if self.vector_store:
            self.vector_store.add_documents(documents=docs, ids=uuids)
            self.logger.info(f"[Matrix RAG] Indexado completo.")
        else:
            self.logger.error("[Matrix RAG] vector_store no inicializado.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        if not self.vector_store:
            return []
        
        # --- Usar el cliente directamente para obtener IDs ---
        # Esto nos da control para añadir el ID de Qdrant a los metadatos para trazabilidad.
        query_embedding = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        docs = []
        for hit in hits:
            docs.append(self._create_document_from_point(hit))
        return docs

    def is_initialized(self) -> bool:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.vectors_count > 0 if hasattr(info, "vectors_count") and info.vectors_count is not None else False
        except Exception:
            return False

    def filter_retrieve(self, must_conditions: list[dict], query_text: Union[str, list[str]] = "") -> List[Document]:
        """
        Recupera TODOS los documentos que cumplen con filtros de metadatos y contienen TODAS las frases/keywords.
        """
        if not self.is_initialized():
            self.logger.warning("filter_retrieve llamado pero la colección parece vacía.")
            return []
    
        qdrant_filter = None
        if must_conditions:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=f"metadata.{cond['key']}", match=MatchValue(value=cond['value']))
                    for cond in must_conditions
                ]
            )
    
        found_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        
        # --- LÓGICA DE KEYWORDS ---
        keywords_to_check = []
        if isinstance(query_text, str) and query_text.strip():
            keywords_to_check.append(query_text.lower())
        elif isinstance(query_text, list):
            keywords_to_check = [kw.lower() for kw in query_text if kw and isinstance(kw, str)]

        # Si no hay keywords, devolvemos todos los documentos que coinciden con el filtro.
        if not keywords_to_check:
            return [self._create_document_from_point(p) for p in found_points]

        results = []
        for point in found_points:
            if not point.payload:
                continue
            
            page_content = point.payload.get("page_content", "")
            page_content_lower = page_content.lower()
            
            # El documento debe contener TODAS las frases/keywords como subcadenas
            if all(kw in page_content_lower for kw in keywords_to_check):
                results.append(self._create_document_from_point(point))
                
        self.logger.info(f"filter_retrieve encontró {len(results)} documentos para condiciones={must_conditions} y keywords={keywords_to_check}")
        return results

    def _create_document_from_point(self, point) -> Document:
        """Crea un objeto Document a partir de un Qdrant ScoredPoint o PointStruct."""
        metadata = point.payload.get("metadata", {}) if point.payload else {}
        metadata["qdrant_id"] = point.id
        page_content = point.payload.get("page_content", "") if point.payload else ""
        return Document(page_content=page_content, metadata=metadata)
