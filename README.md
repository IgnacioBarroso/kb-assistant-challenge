# Matrix RAG API

This project implements a modular and scalable Retrieval-Augmented Generation (RAG) system to answer complex questions about *The Matrix* script. It uses FastAPI for the API, Qdrant as the vector database, and OpenAI for embeddings and language models.

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- Docker (for containerized setup)
- Make

First, install Make if you don't have it:

```bash
# On Debian/Ubuntu
sudo apt install make
```

Install Docker following the official [Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

Clone the repository and enter the project folder:

```bash
git clone <REPO_URL>
cd <REPO_NAME>
```

Create a `.env` file in the root with your OpenAI key:

```env
OPENAI_API_KEY=your_openai_key
```

(Optional) Adjust parameters in `src/settings/config.py` if needed (ports, model, etc).

---

## Usage

### Dev Container (Recommended)

This project includes a reproducible, preconfigured development environment using Dev Container. All dependencies, extensions and settings are consistent and compatible with the current solution.

1. Open the repository in VS Code:

    ```bash
    code .
    ```
2. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension if you don't have it.
3. Reopen the project in the container:
    - Press `F1`, type `Dev Containers`, and select Reopen in Container.
4. Wait for the container to build (first time may take a while; installs Python 3.12, dependencies, extensions and Oh My Zsh).
5. You can now work, run notebooks and launch the API from the container.

Open a terminal in VS Code (inside the container) and run:

```bash
uvicorn src.api.main:app --reload
```

To run the tests:

```bash
pytest
```

You can use notebooks, install new dependencies (edit `requirements.txt`) and everything will work reproducibly.

#### Advantages

- Python 3.12+ environment ready and isolated
- VS Code dependencies and extensions preinstalled
- Notebook and interactive development support
- Configuration aligned with the solution and unified requirements

### Docker Compose

```bash
make run-server
# This starts the API and Qdrant in containers
```

### Local (requires Python 3.12+, Qdrant and dependencies installed)

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

---

## How it works

When the API starts:

- Loads and splits the PDF script (`resources/movie-scripts/the-matrix-1999.pdf`).
- Indexes the chunks in Qdrant (only the first time).

You can test the system from Swagger UI:

- Open [http://localhost:8000/docs](http://localhost:8000/docs)
- Use the POST `/ask` endpoint with a body like:

    ```json
    {
        "query": "Why are humans similar to a virus? And who says that?",
        "top_k": 8,
        "attach_documents": true
    }
    ```

- Or from the console with curl:

    ```bash
    curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"query": "How many times does Morpheus mention that Neo is the One?", "top_k": 10, "attach_documents": true}'
    ```

---

## Running tests

To run all tests (API and internal components):

```bash
pytest
```

This will execute both endpoint tests (`tests/api/`) and internal logic tests (`tests/core/`).

To run only a specific suite:

```bash
pytest tests/api/
pytest tests/core/
```

Make sure all dependencies are installed and the API is up before running tests.

---

## Internal RAG pipeline flow

1. The user sends a question to the `/ask` endpoint.
2. The system detects if the question is simple or complex (count, coverage, reasoning, etc).
3. For count/coverage, it uses exhaustive retrieval (`keyword_search`); otherwise, semantic retrieval.
4. The LLM orchestrator answers directly or decomposes and synthesizes the response as needed.
5. Returns a structured response with:
    - `answer`: generated answer
    - `confidence`: confidence
    - `sources_used`: used chunks
    - `reasoning`: reasoning
    - `retrieved_documents`: (optional) full chunks

---

## Key folders and files

- `src/api/`
    - `main.py`: exposes the FastAPI app
    - `app.py`: initializes services and lifecycle
    - `routers.py`: defines the `/ask` endpoint and uses the RAG pipeline
    - `schemas.py`: Pydantic models for request/response
- `src/services/`
    - `document_loader_service.py`: loads and splits the PDF
    - `qdrant_retriever_service.py`: indexes and retrieves chunks using Qdrant
    - `rag_service.py`: orchestrates loader, retriever and LLM agent
- `src/settings/config.py`: centralized configuration

---

## System capabilities

- Automatically indexes The Matrix script in Qdrant on startup
- Retrieves relevant context for any question using OpenAI embeddings
- Orchestrates LLM agents for simple and complex queries:
    - Decomposes complex questions into subqueries
    - Synthesizes final answers with reasoning and sources
- Returns structured answers with confidence, sources and reasoning
- All configuration is centralized and secure
---

## Example usage of `/ask` endpoint

Request:

```json
POST /ask
{
    "query": "Why are humans similar to a virus? And who says that?",
    "top_k": 8,
    "attach_documents": true
}
```

Response:

```json
{
    "query": "Why are humans similar to a virus? And who says that?",
    "answer": "...generated by the LLM...",
    "confidence": 0.92,
    "sources_used": ["chunk_23", "chunk_45"],
    "reasoning": "The answer is based on the chunks where Agent Smith compares humans to a virus...",
    "retrieved_documents": [ ... ]
}
```

---

## Extensibility and best practices

- Modular: each service is decoupled and testable
- Security: sensitive keys and paths in `.env`
- Scalable: easy to swap LLM model, retriever or loader
- OpenAPI documentation generated automatically

---

## Limitations and considerations

- The system depends on the quality of chunking and context coverage
- Decomposition and synthesis of complex queries depends on LLM and prompt quality
- If relevant context is not retrieved, the answer may be incomplete

---

## Current status and next steps

- The pipeline is ready to answer complex, reasoned questions about the indexed script
- Adjust chunking and retrieval to maximize relevant context coverage
- Admin and monitoring endpoints can be added

See: [notebooks/03-llm-agents/01-llm-agents.ipynb](notebooks/03-llm-agents/01-llm-agents.ipynb)
### Advanced Capability - MCP Server
For advanced reasoning and agent orchestration (see Part 2 requirements), use an MCP server. The environment includes Pydantic-AI with built-in MCP protocol support.
See: [https://ai.pydantic.dev/mcp/](https://ai.pydantic.dev/mcp/)
#### System evaluation

## Jupyter
Jupyter is preconfigured inside the Dev Container.
Explore examples in the [notebooks/](notebooks/) directory.
When opening a notebook, select the kernel: Python Environments -> Python 3.12 (Global Env)
---

## Custom Python library
A local Python package named kbac (KB Assistant Challenge) is included. It contains utility functions for the project. Extend this library as needed. Example usage: [notebooks/01-loaders/01-matrix-script-loader.ipynb](notebooks/01-loaders/01-matrix-script-loader.ipynb).
After modifying this library, you don't need to rebuild the container, but restart the notebook if using it in Jupyter.
---

## Python dependencies
Install additional Python libraries by adding them to requirements.txt.
Rebuild the container afterwards (F1 + Rebuild Container).
---

## Environment variables
Define environment variables (such as OPENAI_API_KEY) in a .env file at the project root. These will be loaded automatically inside the Dev Container.
Example .env file:
```env
OPENAI_API_KEY=your-key-here
MY_CUSTOM_VAR=some-value
```

    Example .env file:

    ```env
    OPENAI_API_KEY=your-key-here
    MY_CUSTOM_VAR=some-value
    ```
