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

load_dotenv()

# Load a PDF file
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

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
bm25_retriever.k = 3 # Retrieve top 3 results

config = [0.3, 0.7]
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store.as_retriever()],weights=config)
#Retrieval QA
qa =RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=LLM_MODEL),
    chain_type=CHAIN_TYPE,
    retriever=ensemble_retriever,
    return_source_documents=True # Option to return source documents
)

q=input('Please ask something (like :What is the main topic of this document?)')
answer = qa.invoke(q)
print(answer['result'])