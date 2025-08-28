import pytest
from unittest.mock import MagicMock, AsyncMock
from src.services.rag_service import RAGService

class RagPipelineSuite:
    @pytest.fixture
    def doc_samples(self):
        return [
            {"text": "Pipeline doc 1", "metadata": {"source": "core"}},
            {"text": "Pipeline doc 2", "metadata": {"source": "core"}}
        ]

    @pytest.fixture
    def fake_loader(self, doc_samples):
        loader = MagicMock()
        loader.load_documents.return_value = doc_samples
        return loader

    @pytest.fixture
    def fake_retriever(self):
        return MagicMock()

    @pytest.fixture
    def fake_retriever_with_docs(self, doc_samples):
        retriever = MagicMock()
        retriever.retrieve.return_value = doc_samples
        retriever.keyword_search.return_value = doc_samples
        return retriever

    @pytest.fixture
    def test_pipeline_init(self, fake_loader, fake_retriever):
        fake_generator = MagicMock()
        rag = RAGService(
            loader=fake_loader,
            retriever=fake_retriever,
            generator=fake_generator
        )
        assert rag.loader == fake_loader
        assert rag.retriever == fake_retriever
        assert rag.generator == fake_generator

    def test_pipeline_indexing(self, fake_loader, fake_retriever, doc_samples):
        fake_generator = MagicMock()
        fake_retriever.is_initialized.return_value = False
        rag = RAGService(
            loader=fake_loader,
            retriever=fake_retriever,
            generator=fake_generator
        )
        rag.index()
        fake_loader.load_documents.assert_called_once()
        fake_retriever.index_documents.assert_called_once_with(doc_samples)

    @pytest.mark.asyncio
    async def test_pipeline_query(self, fake_loader, fake_retriever_with_docs, doc_samples):
        fake_generator = MagicMock()
        fake_generator.generate_response = AsyncMock(return_value={
            "query": "pipeline test",
            "answer": "Pipeline response",
            "confidence": 0.99,
            "sources_used": ["core1", "core2"],
            "reasoning": "",
            "retrieved_documents": doc_samples
        })
        rag = RAGService(
            loader=fake_loader,
            retriever=fake_retriever_with_docs,
            generator=fake_generator
        )
        result = await rag.query("pipeline test", top_k=3, attach_documents=True)
        fake_generator.generate_response.assert_awaited()
        assert result["query"] == "pipeline test"
        assert result["answer"] == "Pipeline response"
        assert result["confidence"] == 0.99
        assert result.get("retrieved_documents") is not None
