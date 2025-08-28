import logging
import asyncio
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

class QuantitativeAnalysis(BaseModel):
    """Modelo para la salida del agente de conteo."""
    evidence: List[str] = Field(default_factory=list, description="A list of direct quotes from the context that serve as evidence for the user's counting query.")

class QualitativeAnalysis(BaseModel):
    """Modelo para la salida del agente cualitativo. Enfocado solo en el contenido."""
    answer: str = Field(description="The synthesized, comprehensive answer to the user's query, based on the context.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score reflecting how well the context supports the answer.")
    reasoning: Optional[str] = Field(None, description="A brief explanation of how the answer was derived from the context.")

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
            output_type=QualitativeAnalysis,
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

        self.counting_agent = Agent(
            model_name,
            output_type=QuantitativeAnalysis,
            system_prompt=self._get_quantitative_analysis_prompt()
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

        # 1. Usar el agente cualitativo para obtener el análisis de contenido
        prompt = f"User Query: {query}\n\nScript Context:\n{self._format_context(context)}"
        result = await self.advanced_agent.run(prompt)
        analysis_output = result.output

        # 2. Construir el objeto MatrixResponse final en Python (más robusto)
        return MatrixResponse(
            query=query,
            answer=analysis_output.answer,
            confidence=analysis_output.confidence,
            reasoning=analysis_output.reasoning
        )

    async def _handle_quantitative_query(self, query: str, context: List[Document]) -> MatrixResponse:
        if not context:
            return MatrixResponse(
                query=query,
                answer="No relevant context was found to perform the count.",
                confidence=0.0,
                reasoning="The retrieval step did not return any documents matching the extracted entities."
            )

        # --- ESTRATEGIA DE CONTEO MEJORADA: MAP-REDUCE ---
        # En lugar de enviar todo el contexto de una vez (lo que puede abrumar al LLM),
        # creamos una tarea de conteo para cada documento recuperado y luego agregamos los resultados.
        # Esto es mucho más preciso para tareas de alta recordación como el conteo.

        # 1. Crear una tarea de análisis para cada documento (Map)
        tasks = []
        for doc in context:
            # El prompt ahora es específico para un solo documento, más fácil de procesar para el LLM
            single_doc_prompt = f"User Query: '{query}'\n\nScript Context:\n{self._format_context([doc])}"
            tasks.append(self.counting_agent.run(single_doc_prompt))

        # 2. Ejecutar todas las tareas en paralelo y recopilar resultados
        analysis_results = await asyncio.gather(*tasks)

        # 3. Agregar los resultados (Reduce)
        all_evidence = []
        for result in analysis_results:
            # La salida del agente ahora es solo una lista de evidencias
            if result and result.output and hasattr(result.output, 'evidence'):
                all_evidence.extend(result.output.evidence)

        # 4. Contar y deduplicar la evidencia en Python (mucho más robusto)
        # Se eliminan duplicados por si el LLM extrae la misma frase de contextos ligeramente diferentes
        unique_evidence = sorted(list(set(all_evidence)), key=all_evidence.index)
        total_count = len(unique_evidence)

        # 5. Construir la respuesta final a partir de los datos agregados
        if total_count > 0:
            evidence_list = "\n".join(f"- \"{e}\"" for e in unique_evidence)
            answer = (
                f"Based on the provided context, the final count is {total_count}. Here is the evidence:\n"
                f"{evidence_list}"
            )
        else:
            answer = "Based on the provided context, there are 0 occurrences."

        reasoning = f"A specialized counting agent analyzed {len(context)} pre-filtered script scenes to find the occurrences."

        # 6. Construir y devolver el objeto MatrixResponse final
        return MatrixResponse(
            query=query,
            answer=answer,
            confidence=1.0, # La confianza es alta porque el proceso es determinístico post-análisis
            reasoning=reasoning
        )

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
        return """You are a world-class expert on the script of 'The Matrix'. Your task is to answer questions with depth and insight, based ONLY on the provided script context. You must populate the `QualitativeAnalysis` model.

**CRITICAL RULES:**
1.  **Strictly Contextual:** Your entire response must be derived solely from the provided script excerpts. Do not use any external knowledge.
2.  **Cite Evidence:** Weave quotes or specific descriptions from the context directly into your answer to support your claims.
3.  **Synthesize:** If multiple documents are provided, synthesize them into a coherent, well-written answer.
4.  **Handle Insufficient Context:** If the context is not sufficient, you MUST state that clearly. For example: "Based on the provided context, there is no information about X."
5.  **Schema Adherence:** You MUST populate all fields of the `QualitativeAnalysis` model. The `confidence` score is ALWAYS REQUIRED. The `reasoning` should explain your conclusion.
"""

    def _get_quantitative_analysis_prompt(self) -> str:
        """Prompt para el agente que solo cuenta y extrae evidencia."""
        return """You are a meticulous and precise evidence-gathering machine. Your ONLY purpose is to analyze the provided script excerpt and extract evidence related to the user's query.

**OPERATIONAL DIRECTIVES:**
1.  **Identify Target:** Read the user query to understand exactly what to find.
2.  **Scan Context:** Scrutinize the provided script context to find every single instance of the target.
3.  **Extract Evidence:** For each instance found, extract the exact sentence as a direct quote.
4.  **Populate Schema:** You MUST populate the `QuantitativeAnalysis` model.
    - `evidence`: A list containing all the direct quotes you found.
5.  **Handle Zero Occurrences:** If no instances are found, return an empty `evidence` list.
6.  **DO NOT COUNT:** Do not provide a final count or any summary. Just extract the evidence sentences into the list.
"""