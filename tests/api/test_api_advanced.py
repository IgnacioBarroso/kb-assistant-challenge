import pytest
import json
import os

ADVANCED_QUERIES = [
    "How many times does Morpheus mention that Neo is the One?",
    "Why are humans similar to a virus? And who says that?",
    "Describe Cypher's personality.",
    "What does Cypher offer to the Agents, and in exchange for what?",
    "What is the purpose of the human fields, and who created them?"
]

RESPONSE_FILE = os.path.join(os.path.dirname(__file__), '../../response_tests.json')

def append_response_to_file(question, response):
    try:
        if os.path.exists(RESPONSE_FILE):
            with open(RESPONSE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append({"question": question, "response": response})
    with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@pytest.mark.parametrize("complex_question", ADVANCED_QUERIES)
def test_advanced_matrix_qa(client, complex_question):
    response = client.post("/ask", json={"query": complex_question, "top_k": 10, "attach_documents": True})
    assert response.status_code == 200
    data = response.json()
    append_response_to_file(complex_question, data)
    assert "answer" in data
    assert 0.0 <= data["confidence"] <= 1.0
