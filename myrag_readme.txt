# InsightEngine: Production-Grade RAG AI Application

InsightEngine is a modern, high-performance Document Analytics and Retrieval-Augmented Generation (RAG) system. It features an interactive Streamlit UI, a robust FastAPI backend service, and uses Qdrant as its high-speed vector database.

---

## Key Features

- **Interactive QA Session**: Real-time conversation interface showing sources, page numbers, and passage relevance scores.
- **Dynamic Retrieval Strategies**:
  - **Dense Vector**: Standard vector similarity retrieval.
  - **MMR (Maximal Marginal Relevance) Diversity**: Deduplicates matching passages.
  - **Lexical + Dense Hybrid**: Combines keyword search and semantic matching.
  - **Multi-Query Expansion**: Multi-query rewriting via LLM.
  - **Cross-Encoder Rerank**: Reranks top matches for optimal relevance.
- **Corrective Retrieval-Augmented Generation (CRAG)**: Automatically falls back to DuckDuckGo search if document match confidence falls below threshold.
- **Asynchronous Ingestion**: Ingests, chunks, and indexes PDFs in the background without blocking users.
- **Strategy Evaluator**: Compare multiple search strategies side-by-side on any test query.

---

## Project Structure

- `server.py`: FastAPI endpoints for ingestion, query coordination, and database wiping.
- `dashboard.py`: Streamlit-based user interface.
- `pipeline.py`: RAGCoordinator, custom LangChain retriever, and query expansion logic.
- `parser.py`: PDF parser and embedding generators.
- `vector_db.py`: Qdrant storage implementation.
- `run_app.bat`: Batch file to launch uvicorn server and streamlit dashboard simultaneously.

---

## Dependencies & Requirements

Install the following requirements to run the application:

```txt
# Streamlit UI Framework
streamlit>=1.30.0

# LangChain Ecosystem & LLM Integrations
langchain>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.20
langchain-text-splitters>=0.0.1
langchain-groq>=0.1.0

# Vector Database & Retrieval Engine
qdrant-client>=1.8.0
sentence-transformers>=2.5.0
pypdf>=4.0.0
beautifulsoup4>=4.12.0

# Asynchronous Background Processing Engine
fastapi>=0.110.0
uvicorn>=0.28.0
httpx>=0.25.0

# Core Utilities & Data Validation
pydantic>=2.6.0
python-dotenv>=1.0.0
numpy>=1.26.0
torch>=2.0.0
```

---

## Setup & Installation

1. **Environment Setup**:
   Ensure you have Python 3.10+ installed. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   You can install all requirements manually or parse them directly from this README file:
   ```bash
   pip install streamlit langchain langchain-community langchain-core langchain-text-splitters langchain-groq qdrant-client sentence-transformers pypdf beautifulsoup4 fastapi uvicorn httpx pydantic python-dotenv numpy torch
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Application**:
   Use the helper batch file on Windows:
   ```cmd
   run_app.bat
   ```
   Or manually run the uvicorn API and Streamlit UI:
   ```bash
   # Terminal 1 - API
   uvicorn server:app --host 127.0.0.1 --port 8000
   
   # Terminal 2 - Streamlit UI
   streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
   ```
