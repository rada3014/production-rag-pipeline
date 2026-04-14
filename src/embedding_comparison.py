from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, LLM_MODEL, CHAIN_TYPE,TOP_N,QUESTIONS
from nltk.tokenize import sent_tokenize
import re
import time

def clean(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).replace('\n', ' ').strip()

load_dotenv()

# Load a PDF file
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
for page in pages:
    page.page_content = clean(page.page_content)



def text_splitter(pages):
    #load the text splitter
    text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

    chunked_text = text_splitter.split_documents(pages)
    return chunked_text


chunked_text= text_splitter(pages)
#print(chunked_text)    
# chunked_sentences = [i.page_content for i in chunked_text]
# print(chunked_sentences)
# print(chunked_sentences[0])
# print(len(chunked_sentences[0]))
# print(chunked_sentences[1])
# print(len(chunked_sentences[1]))

#generating text embeddings
#open AI Embeddings
embeddings1 = OpenAIEmbeddings(model=EMBEDDING_MODEL)
embeddings2 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embeddings3 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

text = "This is a test document."

embed_result1 = embeddings1.embed_query(text)
embed_result2 = embeddings2.embed_query(text)
embed_result3 = embeddings3.embed_query(text)

embed_result = [embed_result1, embed_result2,embed_result3]


st = time.time()
vector_store1 = FAISS.from_documents(chunked_text, embeddings1)
index_time1 = time.time() - st
st = time.time()
vector_store2 = FAISS.from_documents(chunked_text, embeddings2)
index_time2 = time.time() - st
st = time.time()
vector_store3 = FAISS.from_documents(chunked_text, embeddings3)
index_time3 = time.time() - st

index_times = [index_time1, index_time2, index_time3]
strategy_names = [
    "Strategy 1 - Open AI Embeddings",
    "Strategy 2 - Sentence Transformer - all-MiniLM-L6-v2",
    "Strategy 3 - Sentence Transformer - all-mpnet-base-v2",
]
retrievers = [
    vector_store1.as_retriever(search_kwargs={"k": TOP_N}),
    vector_store2.as_retriever(search_kwargs={"k": TOP_N}),
    vector_store3.as_retriever(search_kwargs={"k": TOP_N}),
]


#Retrieval QA
qa1 =RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=vector_store1.as_retriever(search_kwargs={"k": TOP_N}),
    return_source_documents=True # Option to return source documents
)

qa2 =RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=vector_store2.as_retriever(search_kwargs={"k": TOP_N}),
    return_source_documents=True # Option to return source documents
)

qa3 =RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=vector_store3.as_retriever(search_kwargs={"k": TOP_N}),
    return_source_documents=True # Option to return source documents
)
costs = ["$0.0002/1K tokens", "$0 (free, local)", "$0 (free, local)"]
relevance_scores = {name: [] for name in strategy_names}

for q in QUESTIONS:
    print("\n" + "="*70)
    print(f"QUESTION: {q}")
    print("="*70)
    query_times = []
    for r in retrievers:
        st = time.time()
        r.invoke(q)
        query_times.append(time.time() - st)

    retrieved = [r.invoke(q) for r in retrievers]
    answers = [qa.invoke(q) for qa in [qa1, qa2, qa3]]

    for name, docs, answer, it, qt, em in zip(strategy_names, retrieved, answers, index_times, query_times, embed_result):
        print(f"\n  [{name}] | Index: {it:.2f}s | Query: {qt:.2f}s | Dims: {len(em)}")
        for i, doc in enumerate(docs, 1):
            text: str = clean(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            print(f"    Chunk {i} ({len(text)} chars): {text[:200]}...")
        print(f"\n  LLM Response: {answer['result']}\n")

        # Rate relevance 1-5 for this model
        while True:
            print(f"  Rate retrieved chunks relevance for [{name}] (1-5): ", end="")
            rating = input().strip()
            if rating in ("1", "2", "3", "4", "5"):
                relevance_scores[name].append(int(rating))
                break
            print("  Please enter a number between 1 and 5.")

# Final summary table
print("\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)
print(f"  {'Model':<45} {'Dims':>6} {'Index(s)':>10} {'Query(s)':>10} {'Avg Rel':>9} {'Cost/1K':>18}")
print("-"*90)
for name, it, em, cost in zip(strategy_names, index_times, embed_result, costs):
    avg_qt = sum(
        query_times[i] for i, n in enumerate(strategy_names) if n == name
    ) / len(QUESTIONS) if QUESTIONS else 0
    avg_rel = sum(relevance_scores[name]) / len(relevance_scores[name]) if relevance_scores[name] else 0
    print(f"  {name:<45} {len(em):>6} {it:>10.2f} {avg_qt:>10.3f} {avg_rel:>9.2f} {cost:>18}")

best = max(relevance_scores, key=lambda k: sum(relevance_scores[k]))
print(f"\n  Best model by avg relevance: {best}")