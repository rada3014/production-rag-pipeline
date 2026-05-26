from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP,LLM_MODEL, CHAIN_TYPE
from config import  QUESTIONS,EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import re 
import cohere

load_dotenv(find_dotenv())

# Load a PDF file
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

def clean(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).replace('\n', ' ').strip()


#load the text splitter
def text_splitter_strategy2(pages):
    #load the text splitter
    text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
    chunked_text = text_splitter.split_documents(pages)
    return chunked_text

chunked_text = text_splitter_strategy2(pages)


# 2. Initialize Retriever
bm25_retriever = BM25Retriever.from_documents(chunked_text)

# 3. Configure search settings
bm25_retriever.k = 10

#generating text embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = FAISS.from_documents(chunked_text, embeddings)

config =[0.3, 0.7]

configs_scores = {f'Weights: {name}': [] for  name in config}

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store.as_retriever()],weights=config)

co = cohere.Client()
scores = {"without_rerank": 0, "with_rerank": 0}

for q in QUESTIONS[:1]:

    print("\n" + "="*60)
    print(f"QUESTION: {q}")
    print("="*60)

    # Retrieve pool of 10
    retrieved = ensemble_retriever.invoke(q)
    docs_text = [clean(doc.page_content) for doc in retrieved]

    # --- WITHOUT RE-RANK: top 3 by ensemble score ---
    print("\n[WITHOUT RE-RANK] Top 3 chunks:")
    for i, text in enumerate(docs_text[:3], 1):
        print(f"\n--- Chunk {i} ---\n{text}\n" + "-"*60)

    relevant = input("How many of these 3 contain the answer? (0/1/2/3): ").strip()
    scores["without_rerank"] = int(relevant)

    # --- WITH RE-RANK: top 3 by Cohere score ---
    results = co.rerank(query=q, documents=docs_text, top_n=3, model='rerank-english-v3.0')

    print("\n[WITH RE-RANK] Top 3 chunks:")
    for i, r in enumerate(results.results, 1):
        print(f"\n--- Chunk {i} (score: {r.relevance_score:.4f}) ---\n{docs_text[r.index]}\n" + "-"*60)

    relevant = input("How many of these 3 contain the answer? (0/1/2/3): ").strip()
    scores["with_rerank"] = int(relevant)

    # --- Result ---
    print("\n" + "="*60)
    print(f"Precision@3 WITHOUT rerank : {scores['without_rerank']}/3")
    print(f"Precision@3 WITH rerank    : {scores['with_rerank']}/3")
    print("="*60)

# # I want to print the retrieved documents in a more readable format, showing the chunk number, page number, and a snippet of the text. Here's how you can do that:
# for i, doc in enumerate(retrieved, 1):
#     text: str = clean(doc.page_content if hasattr(doc, 'page_content') else str(doc))
#     print(f"\n--- Chunk {i} (page {doc.metadata.get('page', '?')}, {len(text)} chars) ---")
#     print(text)
#     print("-"*60)

