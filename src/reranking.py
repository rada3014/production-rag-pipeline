from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, QUESTIONS
import cohere
import re

load_dotenv(find_dotenv())

# ── Setup ─────────────────────────────────────────────────────

loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = text_splitter.split_documents(pages)

embeddings     = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store   = FAISS.from_documents(chunks, embeddings)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store.as_retriever()],
    weights=[0.3, 0.7],
)

co = cohere.Client()

def clean(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).replace('\n', ' ').strip()


# ── Evaluation loop ───────────────────────────────────────────

all_scores = []   # list of (question, without_score, with_score)

for i, q in enumerate(QUESTIONS, 1):

    print(f"\n{'='*60}")
    print(f"  Question {i}/{len(QUESTIONS)}: {q}")
    print(f"{'='*60}")

    retrieved = ensemble_retriever.invoke(q)
    docs_text = [clean(doc.page_content) for doc in retrieved]

    # WITHOUT reranking: top 3 by ensemble score
    print("\n[WITHOUT RERANKING] Top 3 chunks:")
    for j, text in enumerate(docs_text[:3], 1):
        print(f"\n  Chunk {j}:\n  {text[:300]}...")
        print("  " + "-"*56)

    without_score = int(input("\n  How many of these 3 chunks contain the answer? (0/1/2/3): ").strip())

    # WITH reranking: top 3 by Cohere score
    reranked = co.rerank(query=q, documents=docs_text, top_n=3, model='rerank-english-v3.0')

    print("\n[WITH RERANKING] Top 3 chunks:")
    for j, r in enumerate(reranked.results, 1):
        print(f"\n  Chunk {j} (relevance score: {r.relevance_score:.4f}):")
        print(f"  {docs_text[r.index][:300]}...")
        print("  " + "-"*56)

    with_score = int(input("\n  How many of these 3 chunks contain the answer? (0/1/2/3): ").strip())

    all_scores.append((q, without_score, with_score))

    print(f"\n  Precision@3  →  Without: {without_score}/3  |  With: {with_score}/3")


# ── Final summary ─────────────────────────────────────────────

total_without = sum(r[1] for r in all_scores)
total_with    = sum(r[2] for r in all_scores)
max_possible  = len(QUESTIONS) * 3

print(f"\n\n{'='*60}")
print("  PRECISION@3 REPORT")
print(f"{'='*60}\n")

print(f"  {'Question':<45} {'Without':>8}  {'With':>6}")
print(f"  {'-'*45} {'-'*8}  {'-'*6}")
for q, wo, wi in all_scores:
    short_q = (q[:42] + "...") if len(q) > 45 else q
    print(f"  {short_q:<45} {wo}/3       {wi}/3")

print(f"\n  {'OVERALL':<45} {total_without}/{max_possible}   {total_with}/{max_possible}")
print(f"  {'Precision@3':<45} {total_without/max_possible:.2f}     {total_with/max_possible:.2f}")
print(f"\n{'='*60}")