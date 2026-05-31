# Production RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with Hybrid Search and Cohere reranking, served via a FastAPI REST API.

Given a PDF document, you can ask any question and get two answers side by side — one from a standard RetrievalQA chain and one from a direct LLM call with Cohere reranking — so you can compare quality.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI `gpt-3.5-turbo` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS (dense retrieval) |
| Keyword search | BM25 (sparse retrieval) |
| Hybrid search | LangChain `EnsembleRetriever` (BM25 30% + FAISS 70%) |
| Reranking | Cohere `rerank-english-v3.0` |
| Orchestration | LangChain |
| API server | FastAPI + Uvicorn |
| Testing | Pytest + Starlette TestClient |

---

## Setup

**1. Clone the repo and create a virtual environment**

```bash
git clone <repo-url>
cd production-rag-pipeline
python -m venv rag_venv
source rag_venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API keys**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
```

---

## How to Run

**API server**

```bash
source rag_venv/bin/activate
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Interactive CLI script** (no server needed)

```bash
cd src
python basic_rag.py
```

---

## API Endpoints

### GET /health
Check if the server is running and whether a document is loaded.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "gpt-3.5-turbo",
  "document_loaded": false,
  "loaded_file": null
}
```

---

### POST /upload
Upload a PDF to be indexed. Must be called before `/query`.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/BERT_(language_model).pdf"
```

Response:
```json
{
  "message": "✓ 'BERT_(language_model).pdf' loaded successfully.",
  "pages": 12,
  "chunks": 47
}
```

---

### POST /query
Ask a question. Returns two answers — one without reranking, one with Cohere reranking.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BERT?"}'
```

Response:
```json
{
  "question": "What is BERT?",
  "retrieval_qa": {
    "answer": "BERT is a transformer-based language model developed by Google...",
    "source_pages": [1, 2]
  },
  "reranked_llm": {
    "answer": "BERT stands for Bidirectional Encoder Representations from Transformers...",
    "source_pages": [1]
  }
}
```

---

## Tests

```bash
source rag_venv/bin/activate
PYTHONPATH=src pytest tests/ -v
```

```
tests/test_api.py::test_health               PASSED
tests/test_api.py::test_query_empty_string   PASSED
tests/test_api.py::test_query_real_question  PASSED

3 passed
```

> Note: `test_query_real_question` makes real API calls — requires a valid `.env` file.

---

## Precision@3

Evaluated manually on the question *"What is BERT?"* using `src/reranking.py`.

| Strategy | Precision@3 |
|---|---|
| Without reranking (ensemble top-3) | 2 / 3 |
| With Cohere reranking | 3 / 3 |

Reranking improves chunk relevance by re-scoring the top-10 retrieved candidates and selecting the 3 most relevant — giving the LLM a tighter, more focused context window.