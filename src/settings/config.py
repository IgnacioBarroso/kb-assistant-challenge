"""
Configuración centralizada para la solución alternativa.
Carga variables sensibles desde un archivo .env en BASE_DIR usando dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    # Qdrant (hardcodeado)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "matrix_collection"
    QDRANT_EMBEDDING_DIMS: int = 256  # text-embedding-3-large (coincide con el ejemplo)

    # OpenAI (solo la API key es sensible)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Para pydantic-ai (requiere el prefijo 'openai:')
    OPENAI_PYDANTIC_MODEL: str = "openai:gpt-4.1-nano"  # gpt-4.1-nano
    # Para LangChain/OpenAI (solo el nombre del modelo)
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Documentos
    MATRIX_SCRIPT_PATH: str = "resources/movie-scripts/the-matrix-1999.pdf"

    # Otros
    DEBUG: bool = False
    APP_PORT: int = 8000

settings = Settings()
