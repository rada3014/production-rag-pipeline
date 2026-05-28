# ============================================================
#  Production RAG Pipeline — Hybrid Search + Cohere Reranking
#
#  Two answer strategies side by side:
#    1. RetrievalQA chain  (no reranking)
#    2. Direct LLM call    (with Cohere reranking)
#
#  Usage:
#    As a script  → python basic_rag.py         (interactive CLI)
#    As a module  → from basic_rag import RAGPipeline  (used by api.py)
# ============================================================

import os
import re
import tempfile

import cohere
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import (
    CHAIN_TYPE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PDF_PATH,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────

ENSEMBLE_WEIGHTS = [0.3, 0.7]   # [BM25 weight, Dense weight]
RERANK_TOP_N     = 3
RETRIEVER_K      = 10

PROMPT_TEMPLATE = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# ── Helpers ───────────────────────────────────────────────────

def clean(text: str) -> str:
    """Collapse whitespace and strip newlines from a chunk of text."""
    return re.sub(r"[ \t]+", " ", text).replace("\n", " ").strip()


def print_section(title: str) -> None:
    """Print a clearly visible section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Pipeline class ────────────────────────────────────────────

class RAGPipeline:
    """
    Encapsulates the full Hybrid Search + Cohere Reranking RAG pipeline.

    Lifecycle:
      1. Instantiate once (initialises models & API clients).
      2. Call load_pdf() to ingest a document.
      3. Call query() for every question.

    Works both as a module (imported by api.py) and as a script
    (see the __main__ block below).
    """

    def __init__(self) -> None:
        # Shared, stateless components — created once, reused across calls
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm        = ChatOpenAI(model=LLM_MODEL)
        self.co         = cohere.Client()

        # Document-specific state — set after load_pdf()
        self.ensemble_retriever: EnsembleRetriever | None   = None
        self.document_loaded: bool                          = False
        self.loaded_filename: str | None                    = None

    # ── Document loading ──────────────────────────────────────

    def load_pdf(self, source: str | bytes, filename: str = "") -> dict:
        """
        Ingest a PDF from a file path (str) or raw bytes.

        - str   → used by the CLI script (PDF_PATH from config)
        - bytes → used by the API (/upload endpoint)

        Builds FAISS + BM25 retrievers and wires up the ensemble.
        Replaces any previously loaded document.
        """
        # Resolve file path
        # Declare pdf_path as str so Pylance doesn't widen it to str|bytes
        pdf_path: str
        if isinstance(source, bytes):
            # API upload: write bytes to a temp file (PyPDFLoader needs a path)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(source)
            tmp.close()
            pdf_path = str(tmp.name)
            should_cleanup = True
        else:
            # Script: use the path directly (source is str here)
            pdf_path = source
            should_cleanup = False

        try:
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()
        finally:
            if should_cleanup:
                os.unlink(pdf_path)   # always remove the temp file

        # Chunk the document
        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(pages)

        # Dense retriever — semantic similarity via FAISS + OpenAI embeddings
        vector_store    = FAISS.from_documents(chunks, self.embeddings)
        dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )

        # Sparse retriever — keyword-based BM25
        bm25_retriever   = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = RETRIEVER_K

        # Ensemble: fuses both retrieval signals
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=ENSEMBLE_WEIGHTS,
        )

        self.document_loaded = True
        self.loaded_filename = filename or os.path.basename(pdf_path)

        return {
            "filename": self.loaded_filename,
            "pages"   : len(pages),
            "chunks"  : len(chunks),
        }

    # ── Querying ──────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Answer a question using both pipeline approaches.

        Returns:
          retrieval_qa  — RetrievalQA chain answer (no reranking)
          reranked_llm  — Direct LLM answer after Cohere reranking
        """
        if not self.document_loaded or self.ensemble_retriever is None:
            raise RuntimeError("No document loaded. Call load_pdf() first.")

        # Retrieve candidate documents from the hybrid ensemble
        retrieved = self.ensemble_retriever.invoke(question)

        # ── Approach 1 : RetrievalQA (no reranking) ──────────
        #    LangChain builds the prompt internally and queries the LLM.
        qa_chain  = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=CHAIN_TYPE,
            retriever=self.ensemble_retriever,   # retriever, not retrieved docs
            return_source_documents=True,
        )
        qa_result    = qa_chain.invoke(question)
        source_pages = sorted({
            doc.metadata.get("page", "?")
            for doc in qa_result.get("source_documents", [])
        })

        # ── Approach 2 : Direct LLM with Cohere Reranking ────
        #  Step 1 — clean texts and send to Cohere reranker
        docs_text      = [clean(doc.page_content) for doc in retrieved]
        rerank_results = self.co.rerank(
            query=question,
            documents=docs_text,
            top_n=RERANK_TOP_N,
            model="rerank-english-v3.0",
        )

        #  Step 2 — recover Document objects via index (preserves metadata)
        top_docs = [retrieved[r.index] for r in rerank_results.results]

        #  Step 3 — build prompt with tighter context and call LLM directly
        context         = "\n\n---\n\n".join(clean(d.page_content) for d in top_docs)
        prompt          = PROMPT_TEMPLATE.format(context=context, question=question)
        reranked_answer = self.llm.invoke(prompt)
        reranked_pages  = sorted({
            doc.metadata.get("page", "?") for doc in top_docs
        })

        return {
            "question"    : question,
            "retrieval_qa": {
                "answer"      : qa_result["result"],
                "source_pages": source_pages,
            },
            "reranked_llm": {
                "answer"      : reranked_answer.content,
                "source_pages": reranked_pages,
            },
        }


# ── Interactive script ────────────────────────────────────────
# Only runs when executed directly: python basic_rag.py
# When imported by api.py, this block is skipped entirely.

if __name__ == "__main__":

    print_section("Loading & Chunking Document")
    pipeline = RAGPipeline()
    meta = pipeline.load_pdf(PDF_PATH)
    print(f"  ✓ Loaded '{meta['filename']}'  "
          f"({meta['pages']} pages → {meta['chunks']} chunks)")

    print_section("Query")
    q = input("  Ask a question (e.g. 'What is the main topic?'): ").strip()

    result = pipeline.query(q)

    # ── Print Approach 1 result
    print_section("Approach 1 — RetrievalQA (no reranking)")
    print(f"\n  Answer:\n  {result['retrieval_qa']['answer']}")
    if result['retrieval_qa']['source_pages']:
        print(f"\n  Source page(s): {result['retrieval_qa']['source_pages']}")

    # ── Print Approach 2 result
    print_section("Approach 2 — Direct LLM with Cohere Reranking")
    print(f"\n  Answer:\n  {result['reranked_llm']['answer']}")
    if result['reranked_llm']['source_pages']:
        print(f"\n  Source page(s): {result['reranked_llm']['source_pages']}")

    print_section("Done")
