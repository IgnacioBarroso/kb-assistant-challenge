import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routers import agent_router
from src.services.rag_service import RAGService
from src.services.qdrant_retriever_service import QdrantRetrieverService
from src.services.implementations.matrix_generator_service import MatrixGeneratorService
from src.services.implementations.matrix_document_loader_service import MatrixDocumentLoaderService

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Inicializando servicios...")
    try:
        # Usamos el único y correcto loader
        loader = MatrixDocumentLoaderService()
        retriever = QdrantRetrieverService()
        generator = MatrixGeneratorService()
        rag_service_instance = RAGService(loader=loader, retriever=retriever, generator=generator)
        
        app.state.rag_service = rag_service_instance
        
        logger.info("[Startup] Iniciando indexación de documentos...")
        rag_service_instance.index()
        logger.info("[Startup] Indexación completada. La aplicación está lista.")
        
    except Exception as e:
        logger.error(f"[Startup] Error fatal durante la inicialización: {e}", exc_info=True)
        raise

    yield
    logger.info("[Shutdown] La aplicación se está cerrando.")

app = FastAPI(
    title="Matrix Agentic RAG API",
    description="Una API para hacer preguntas sobre el guion de The Matrix.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(agent_router)
