from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, LLM_MODEL, CHAIN_TYPE
import cohere
import re
load_dotenv()

def clean(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).replace('\n', ' ').strip()


# Load a PDF file
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

co = cohere.Client()

#load the text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunked_text = text_splitter.split_documents(pages)

#generating text embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# This embeds all chunks AND stores them correctly in one call
vector_store = FAISS.from_documents(chunked_text, embeddings)

# 2. Initialize the BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunked_text)

# 3. Configure search settings
bm25_retriever.k = 10 # Retrieve top 3 results

config = [0.3, 0.7]

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store.as_retriever()],weights=config)

q=input('Please ask something (like :What is the main topic of this document?)')

retrieved = ensemble_retriever.invoke(q)

docs_text = [clean(doc.page_content) for doc in retrieved]

results = co.rerank(query=q, documents=docs_text, top_n=3, model='rerank-english-v3.0')

#Retrieval QA
qa =RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=results,
    return_source_documents=True # Option to return source documents
)


answer = qa.invoke(q)
print(answer['result'])