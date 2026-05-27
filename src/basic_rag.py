# ============================================================
#  Production RAG Pipeline — Hybrid Search + Cohere Reranking
#  Demonstrates two answer strategies side by side:
#    1. RetrievalQA chain  (no reranking)
#    2. Direct LLM call    (with Cohere reranking)
# ============================================================

import re

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

# ── Helpers ──────────────────────────────────────────────────

def clean(text: str) -> str:
    """Collapse whitespace and strip newlines from a chunk of text."""
    return re.sub(r"[ \t]+", " ", text).replace("\n", " ").strip()


def print_section(title: str) -> None:
    """Print a clearly visible section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Document Loading & Chunking ───────────────────────────────

print_section("Loading & Chunking Document")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"  Loaded {len(pages)} page(s) from: {PDF_PATH}")

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = text_splitter.split_documents(pages)
print(f"  Split into {len(chunks)} chunk(s)  "
      f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")


# ── Retriever Setup ───────────────────────────────────────────

print_section("Building Retrievers")

# Dense retriever — semantic similarity via FAISS + OpenAI embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(chunks, embeddings)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
print("  ✓ Dense retriever (FAISS) ready")

# Sparse retriever — keyword-based BM25
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10
print("  ✓ Sparse retriever (BM25) ready")

# Ensemble retriever — combines dense (70 %) + sparse (30 %)
# Higher weight on dense for semantic queries; tune as needed.
ENSEMBLE_WEIGHTS = [0.3, 0.7]   # [BM25, Dense]
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=ENSEMBLE_WEIGHTS,
)
print(f"  ✓ Ensemble retriever ready  (weights BM25={ENSEMBLE_WEIGHTS[0]}, Dense={ENSEMBLE_WEIGHTS[1]})")

# Cohere client — used for reranking later
co = cohere.Client()


# ── Query ─────────────────────────────────────────────────────

print_section("Query")
q = input("  Ask a question (e.g. 'What is the main topic?'): ").strip()

# Retrieve candidate documents from the hybrid ensemble
retrieved = ensemble_retriever.invoke(q)
print(f"\n  Retrieved {len(retrieved)} candidate doc(s) from ensemble")


# ── Approach 1 : RetrievalQA (no reranking) ───────────────────
#
#  Uses the ensemble retriever directly inside a LangChain chain.
#  The LLM receives all retrieved chunks and answers from them.
#  No reranking step — faster but may include less relevant chunks.

print_section("Approach 1 — RetrievalQA (no reranking)")

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=ensemble_retriever,   # pass the retriever, not the retrieved docs
    return_source_documents=True,
)

qa_result = qa_chain.invoke(q)
print(f"\n  Answer:\n  {qa_result['result']}")

# Optionally surface source pages so the answer can be traced back
source_pages = sorted({
    doc.metadata.get("page", "?")
    for doc in qa_result.get("source_documents", [])
})
if source_pages:
    print(f"\n  Source page(s): {source_pages}")


# ── Approach 2 : Direct LLM with Cohere Reranking ─────────────
#
#  1. Clean the retrieved chunk texts for the reranker.
#  2. Cohere rerank selects the top-N most relevant chunks.
#  3. Those top chunks are injected into a prompt and sent directly
#     to the LLM — giving a tighter, more focused context window.

print_section("Approach 2 — Direct LLM with Cohere Reranking")

# Step 1 — prepare clean text for the reranker
docs_text = [clean(doc.page_content) for doc in retrieved]

# Step 2 — rerank; top_n controls how many chunks reach the LLM
rerank_results = co.rerank(
    query=q,
    documents=docs_text,
    top_n=10,
    model="rerank-english-v3.0",
)

# Step 3 — map reranked indices back to the original Document objects
#           so we preserve metadata (page numbers, source, etc.)
top_docs = [retrieved[r.index] for r in rerank_results.results]
print(f"  Reranked to top {len(top_docs)} doc(s) "
      f"(from {len(retrieved)} candidates)")

# Step 4 — build prompt and call LLM directly
PROMPT_TEMPLATE = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

context = "\n\n---\n\n".join(clean(doc.page_content) for doc in top_docs)
prompt  = PROMPT_TEMPLATE.format(context=context, question=q)

llm = ChatOpenAI(model=LLM_MODEL)
reranked_answer = llm.invoke(prompt)

print(f"\n  Answer:\n  {reranked_answer.content}")

reranked_pages = sorted({
    doc.metadata.get("page", "?")
    for doc in top_docs
})
if reranked_pages:
    print(f"\n  Source page(s): {reranked_pages}")

print_section("Done")
