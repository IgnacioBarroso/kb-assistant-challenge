import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import os
import json

@pytest.fixture(scope="session", autouse=True)
def clean_response_tests():
    """
    Limpia el archivo response_tests.json antes de ejecutar la suite de tests.
    """
    response_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../response_tests.json'))
    if os.path.exists(response_file):
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

# Fixture global para cliente de pruebas FastAPI
# Permite usar `client` en cualquier test de la suite
@pytest.fixture(scope="session")
def client():
    """
    Cliente de pruebas para la API principal.
    Úsalo pasando `client` como argumento en tus tests.
    """
    with TestClient(app) as test_client:
        yield test_client
