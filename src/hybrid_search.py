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
bm25_retriever.k = 3 # Retrieve top 3 results

#generating text embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = FAISS.from_documents(chunked_text, embeddings)

configs =[[0.3, 0.7], [0.7, 0.3]]

configs_scores = {f'Weights: {name}': [] for  name in configs}


for config,name in zip(configs,configs_scores):
     
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store.as_retriever()],weights=config)
    #Retrieval QA
    qa =RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=LLM_MODEL),
        chain_type=CHAIN_TYPE,
        retriever=ensemble_retriever,
        return_source_documents=True # Option to return source documents
    )
    print(f"\n  [{name}] ")
    for q in QUESTIONS[:1]:

        
        print("\n" + "="*60)
        print(f"QUESTION: {q}")
        print("="*60)
        retrieved = ensemble_retriever.invoke(q)
        answer = qa.invoke(q)

        for i, doc in enumerate(retrieved, 1):
            text: str = clean(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            print(f"\n--- Chunk {i} (page {doc.metadata.get('page', '?')}, {len(text)} chars) ---")
            print(text)
            print("-"*60)
        print(f"\n  LLM Response: {answer['result']}\n")

        # Rate relevance 1-5 for this model
        while True:
            print(f"  Relevant? [{name}] (1=yes / 0=no): ", end="")
            rating = input().strip()
            if rating in ("0", "1"):
                configs_scores[name].append(int(rating))
                break
            print("  Please enter 0 or 1.")


# Final summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"  {'Weights':<45} {'Correct/In-Correct':>6}")
print("-"*60)
for name in configs_scores:
    score = sum(configs_scores[name])
    total = len(configs_scores[name])
    print(f"  {name:<45} {score}/{total} correct")


best = max(configs_scores, key=lambda k: sum(configs_scores[k]))
print(f"\n  Best model by avg relevance: {best}")