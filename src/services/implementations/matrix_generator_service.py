import logging
from typing import Optional, List, Any

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from src.services.generator_service import GeneratorService
from pydantic_ai import Agent

# --------------------------------------------------------------------------
# Modelos Pydantic (Con una pequeña mejora)
# --------------------------------------------------------------------------

class FilteredContext(BaseModel):
    relevant_sentences: List[str] = Field(description="A list of exact sentences from the context that directly answer or are relevant to the user's query.")

class CountingExtraction(BaseModel):
    character: Optional[str] = Field(None, description="The character speaking or mentioned, if any.")
    keywords: List[str] = Field(description="The specific keywords, phrases, or objects to count.")

class MatrixResponse(BaseModel):
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0. This field is ALWAYS required.")
    sources_used: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    retrieved_documents: Optional[Any] = None

# --------------------------------------------------------------------------
# Servicio Generador Principal
# --------------------------------------------------------------------------

class MatrixGeneratorService(GeneratorService):
    def __init__(self, model_name: str = "openai:gpt-4.1-nano"):
        self.model_name = model_name
        self.logger = logging.getLogger("uvicorn")

        # --- Agentes Especializados ---

        self.advanced_agent = Agent(
            model_name,
            output_type=MatrixResponse,
            system_prompt=self._get_advanced_system_prompt()
        )
        
        self.extraction_agent = Agent(
            model_name,
            output_type=CountingExtraction,
            system_prompt="You are an entity extraction expert. From the user's query, extract the character and the specific keywords/objects to be counted. For 'How many times does Morpheus mention 'the One'?', you must extract character='Morpheus' and keywords=['the One']. For 'How many cars appear?', you must extract character=None and keywords=['car', 'cars']."
        )

        self.filter_agent = Agent(
            model_name,
            output_type=FilteredContext,
            system_prompt="You are an AI text filter. Your job is to read a large context and extract only the exact sentences that are directly relevant to the user's query. Do not paraphrase, do not answer the query, just extract the verbatim sentences."
        )

    async def generate_response(self, query: str, context: list) -> MatrixResponse:
        normalized_context = self._normalize_context(context)
        query_type = self._get_query_type(query)

        if query_type == 'QUANTITATIVE':
            return await self._handle_quantitative_query(query, normalized_context)
        else:
            return await self._handle_qualitative_query(query, normalized_context)

    async def _handle_qualitative_query(self, query: str, context: List[Document]) -> MatrixResponse:
        if not context:
            return MatrixResponse(query=query, answer="I could not find any relevant context to answer this question.", confidence=0.0)

        prompt = f"User Query: {query}\n\nScript Context:\n{self._format_context(context)}"
        result = await self.advanced_agent.run(prompt)
        return result.output

    async def _handle_quantitative_query(self, query: str, context: List[Document]) -> MatrixResponse:
        if not context:
            return MatrixResponse(
                query=query,
                answer="No relevant context was found to perform the count.",
                confidence=0.0,
                reasoning="The retrieval step did not return any documents matching the extracted entities."
            )

        prompt = f"User Query: '{query}'\n\nScript Context:\n{self._format_context(context)}"
        
        counting_agent = Agent(
            self.model_name,
            output_type=MatrixResponse,
            system_prompt=self._get_counting_system_prompt()
        )
        result = await counting_agent.run(prompt)
        output = result.output
        
        if not output.reasoning:
            output.reasoning = f"A specialized counting agent analyzed {len(context)} pre-filtered script scenes to find the occurrences."
        
        return output

    def _get_query_type(self, query: str) -> str:
        query_lower = query.lower()
        quantitative_indicators = ["how many times", "how many", "count", "list all", "find every"]
        if any(indicator in query_lower for indicator in quantitative_indicators):
            return 'QUANTITATIVE'
        return 'QUALITATIVE'

    def _normalize_context(self, context: list) -> List[Document]:
        normalized = []
        for doc in context:
            if isinstance(doc, Document):
                normalized.append(doc)
            elif isinstance(doc, dict):
                page_content = doc.get('page_content') or doc.get('text') or ''
                metadata = doc.get('metadata', {})
                normalized.append(Document(page_content=page_content, metadata=metadata))
        return normalized

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No context provided."
        return "\n\n".join(
            f"--- Document (Scene: {doc.metadata.get('scene_number', 'N/A')}, Location: {doc.metadata.get('location', 'Unknown')}) ---\n{doc.page_content}"
            for doc in docs
        )

    def _get_advanced_system_prompt(self) -> str:
        """Prompt para el agente que responde preguntas cualitativas."""
        return """You are a world-class expert on the script of 'The Matrix'. Your task is to answer questions with depth and insight, based ONLY on the provided script context.

**CRITICAL RULES:**
1.  **Strictly Contextual:** Your entire response must be derived solely from the provided script excerpts. Do not use any external knowledge.
2.  **Cite Evidence:** Weave quotes or specific descriptions from the context directly into your answer to support your claims.
3.  **Synthesize:** If multiple documents are provided, synthesize them into a coherent, well-written answer.
4.  **Handle Insufficient Context:** If the context is not sufficient, you MUST state that clearly. For example: "Based on the provided context, there is no information about X."
5.  **Schema Adherence:** You MUST populate all fields of the `MatrixResponse` model. The `answer` should be a complete paragraph. The `reasoning` should explain your conclusion. The `confidence` score (from 0.0 to 1.0) is ALWAYS REQUIRED and reflects how well the context supports your answer.
"""

    def _get_counting_system_prompt(self) -> str:
        """Prompt altamente directivo para el agente de conteo."""
        return """You are a meticulous and precise counting machine. Your ONLY purpose is to analyze the provided script excerpts and count occurrences based on the user's query. You must follow these rules without deviation.

**OPERATIONAL DIRECTIVES:**
1.  **Identify Target:** Read the user query to understand exactly what to count.
2.  **List All Evidence First:** Before the final count, you MUST create a bulleted list of every single instance found in the context. Each bullet point must contain a direct quote of the sentence where the item was found.
3.  **State Final Count:** After the evidence list, provide the final summary. It MUST be in the format: "Based on the provided context, the final count is X."
4.  **Handle Zero Occurrences:** If no instances are found, your answer MUST be: "Based on the provided context, there are 0 occurrences."
5.  **Final Answer Structure:** The `answer` field of your response MUST contain BOTH the bulleted list of evidence AND the final count summary.
6.  **Confidence Score:** The `confidence` field is MANDATORY. Set it to 1.0 if you can perform the count (even if the result is 0). Set it to 0.5 if the context is ambiguous.
"""