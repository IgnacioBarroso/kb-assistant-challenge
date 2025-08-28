import pytest
from unittest.mock import MagicMock, patch

with patch.dict("sys.modules", {
    'qdrant_client': MagicMock(),
    'langchain_openai': MagicMock(),
    'langchain_qdrant': MagicMock(),
    'langchain_core.documents': MagicMock(),
}):
    from src.services.qdrant_retriever_service import QdrantRetrieverService

class RetrieverComponentSuite:
    @pytest.fixture
    def example_docs(self):
        return [
            {"text": "Sample doc for retriever", "metadata": {"source": "unit"}},
            {"text": "Another retriever test doc", "metadata": {"source": "unit"}}
        ]

    def test_indexing(self, example_docs):
        svc = QdrantRetrieverService.__new__(QdrantRetrieverService)
        svc.vector_store = MagicMock()
        svc.index_documents(example_docs)
        assert svc.vector_store.add_documents.called

    def test_semantic_search(self):
        svc = QdrantRetrieverService.__new__(QdrantRetrieverService)
        svc.vector_store = MagicMock()
        svc.similarity_search("matrix", k=2)
        svc.vector_store.similarity_search_with_score.assert_called_once_with("matrix", k=2)

    def test_keyword_search(self, example_docs):
        svc = QdrantRetrieverService.__new__(QdrantRetrieverService)
        # keyword_search no está implementado, debe devolver lista vacía y loguear warning
        results = svc.keyword_search(["Sample", "retriever"])
        assert results == []

def test_retriever_init_error(monkeypatch, caplog):
    from src.services import qdrant_retriever_service
    monkeypatch.setattr(qdrant_retriever_service, "QdrantClient", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception):
        qdrant_retriever_service.QdrantRetrieverService()
    assert "Error inicializando QdrantRetrieverService" in caplog.text
