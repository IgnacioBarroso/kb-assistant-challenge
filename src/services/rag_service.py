import logging
from src.services.document_loader_service import DocumentLoaderService
from src.services.qdrant_retriever_service import QdrantRetrieverService
from src.api.schemas import RetrievedDoc

class RAGService:
    def __init__(self, loader: DocumentLoaderService, retriever: QdrantRetrieverService, generator):
        self.logger = logging.getLogger("uvicorn")
        self.loader = loader
        self.retriever = retriever
        self.generator = generator

    def index(self) -> None:
        """Carga e indexa los documentos de forma síncrona."""
        try:
            if self.retriever.is_initialized():
                self.logger.info("La colección ya contiene documentos. Se omite el indexado.")
                return
            
            # Sin 'await'
            documents = self.loader.load_documents()
            
            self.logger.info(f"{len(documents)} documentos de escenas cargados para indexar.")
            self.retriever.index_documents(documents)
            self.logger.info("Documentos de escenas indexados correctamente en Qdrant.")
        except Exception as e:
            self.logger.error(f"[RAGService.index] Error al indexar documentos: {e}", exc_info=True)
            raise

    def _get_query_type(self, query: str) -> str:
        query_lower = query.lower()
        quantitative_indicators = ["how many times", "how many", "count", "list all", "find every"]
        if any(indicator in query_lower for indicator in quantitative_indicators):
            return 'QUANTITATIVE'
        return 'QUALITATIVE'

    async def query(self, question: str, top_k: int = 10, attach_documents: bool = True):
        self.logger.info(f"--- Iniciando nueva consulta: '{question}' ---")
        try:
            query_type = self._get_query_type(question)
            retrieved_docs = []

            if query_type == 'QUANTITATIVE':
                self.logger.info("[Strategy] QUANTITATIVE: Extracting entities...")
                extraction_result = await self.generator.extraction_agent.run(question)
                entities = extraction_result.output
                
                must_conditions = []
                if hasattr(entities, 'character') and entities.character:
                    must_conditions.append({'key': 'characters', 'value': entities.character.upper()})
                keywords = getattr(entities, 'keywords', [])

                # --- ESTRATEGIA DE FALLBACK CUANTITATIVA ---
                # Intento A: Alta precisión (personaje + keywords)
                self.logger.info(f"[Quantitative Fallback A] Searching with character filter and keywords: {keywords}")
                retrieved_docs = self.retriever.filter_retrieve(must_conditions=must_conditions, query_text=keywords)

                # Intento B: Si A falla, buscar solo por personaje (si existe)
                if not retrieved_docs and must_conditions:
                    self.logger.info("[Quantitative Fallback B] Attempt A failed. Searching with character filter only.")
                    retrieved_docs = self.retriever.filter_retrieve(must_conditions=must_conditions, query_text="")

                # Intento C: Si todo lo anterior falla, hacer una búsqueda semántica general
                if not retrieved_docs:
                    self.logger.info("[Quantitative Fallback C] Attempts A & B failed. Performing general semantic search.")
                    retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            else:
                self.logger.info("[Strategy] QUALITATIVE: Similarity search.")
                retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            # La deduplicación ahora se basa en el page_content de los objetos Document
            unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())
            self.logger.info(f"Contexto final: {len(unique_docs)} documentos únicos para el generador.")
            
            # El generador recibe objetos Document y devuelve un objeto MatrixResponse
            result = await self.generator.generate_response(question, unique_docs)
            result_data = result.model_dump()

            # --- MEJORA DE FIABILIDAD: Rellenar campos si el LLM no lo hizo ---
            if not result_data.get("sources_used"):
                result_data["sources_used"] = [
                    f"Scene {doc.metadata.get('scene_number', 'N/A')}: {doc.metadata.get('location', 'Unknown')}"
                    for doc in unique_docs
                ]
            
            if not result_data.get("reasoning"):
                result_data["reasoning"] = (
                    f"The answer was synthesized from {len(unique_docs)} documents retrieved "
                    f"via a {query_type.lower()} search strategy."
                )

            # --- TRANSFORMACIÓN A ESQUEMA DE SALIDA ---
            if attach_documents:
                # Convertir LangChain Documents a nuestro schema RetrievedDoc
                response_docs = []
                for doc in unique_docs:
                    response_docs.append(RetrievedDoc(
                        id=doc.metadata.get("qdrant_id", "ID not found"),
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    ))
                result_data['retrieved_documents'] = response_docs
            else:
                result_data['retrieved_documents'] = []

            self.logger.info("--- Consulta finalizada con éxito ---")
            return result_data

        except Exception as e:
            self.logger.error(f"[RAGService.query] Error fatal en la orquestación: {e}", exc_info=True)
            raise
