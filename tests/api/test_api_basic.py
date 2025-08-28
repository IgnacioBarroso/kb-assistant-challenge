import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.dependencies import get_rag_service # Importamos la función de dependencia

# --- Lista de Preguntas para los Tests ---
BASIC_QUERIES = [
    "Under what circumstances does Neo see a white rabbit?",
    "How did Trinity and Neo first meet?",
    "Why is there no sunlight in the future?",
    "Who needs solar power to survive?",
    "Why do the Agents want to capture Morpheus?",
    "Describe the Nebuchadnezzar.",
    "What is Nebuchadnezzar's crew made up of?",
]

# --- Tests Parametrizados ---

@pytest.mark.parametrize("user_question", BASIC_QUERIES)
def test_basic_matrix_qa(client: TestClient, user_question: str):
    """
    Test parametrizado que prueba varias preguntas básicas contra el endpoint /ask.
    """
    # Aseguramos que no haya sobreescrituras de dependencias activas
    app.dependency_overrides.clear()
    
    response = client.post("/ask", json={"query": user_question, "top_k": 8, "attach_documents": False})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] is not None
    assert data["query"] == user_question
