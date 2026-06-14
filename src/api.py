# ============================================================
#  FastAPI server — RAG Pipeline REST API
#
#  Endpoints:
#    POST /upload      — ingest a PDF document
#    POST /query       — ask a question, get two answers
#    GET  /health      — liveness check + pipeline status
#
#  Run locally:
#    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
# ============================================================

from contextlib import asynccontextmanager
from basic_rag import RAGPipeline
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from config import LLM_MODEL


load_dotenv()

# ── Lifespan: startup / shutdown ──────────────────────────────

# The pipeline is initialised once when the server starts and shared
# across all requests.  This avoids re-loading models on every call.
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀  Initialising RAG pipeline...")
    pipeline = RAGPipeline()          # loads models & clients (no PDF yet)
    print("✓  Pipeline ready. POST a PDF to /upload to load a document.")
    yield
    # Teardown — nothing to explicitly close for this stack
    print("🛑  Server shutting down.")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Production RAG Pipeline",
    description=(
        "Hybrid Search (BM25 + FAISS) with Cohere reranking. "
        "Upload a PDF, then query it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class ApproachResult(BaseModel):
    answer: str
    source_pages: list

class QueryResponse(BaseModel):
    question: str
    retrieval_qa: ApproachResult     # RetrievalQA chain (no reranking)
    reranked_llm: ApproachResult     # Direct LLM call after Cohere reranking


# ── Endpoints ─────────────────────────────────────────────────

@app.post(
    "/upload",
    summary="Upload a PDF document",
    description="Ingests a PDF: loads pages, chunks them, and builds the retrieval index.",
)
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF upload and build the retrieval index from it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet.")

    try:
        meta = pipeline.load_pdf(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {
        "message"  : f"✓ '{file.filename}' loaded successfully.",
        "pages"    : meta["pages"],
        "chunks"   : meta["chunks"],
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Returns two answers side by side: "
        "(1) RetrievalQA without reranking, "
        "(2) Direct LLM call with Cohere reranking."
    ),
)
async def query_pipeline(req: QueryRequest):
    """Answer a question using the loaded document."""
    if not pipeline or not pipeline.document_loaded:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. POST a PDF to /upload first.",
        )

    try:
        result = pipeline.query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return result


@app.get(
    "/health",
    summary="Health check",
    description="Returns server status and whether a document is currently loaded.",
)
async def health():
    """Liveness + readiness check."""
    return {
        "status"         : "ok",
        "model"          : LLM_MODEL,
        "document_loaded": pipeline.document_loaded if pipeline else False,
        "loaded_file"    : pipeline.loaded_filename if pipeline else None,
    }
