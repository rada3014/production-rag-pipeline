from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
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

def clean(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).replace('\n', ' ').strip()

load_dotenv()

# Load a PDF file
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
for page in pages:
    page.page_content = clean(page.page_content)


def text_splitter_strategy1(pages):
    #load the text splitter
    text_splitter = CharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 50,
    separator='',
    strip_whitespace=False
)
    chunked_text = text_splitter.split_documents(pages)
    return chunked_text

def text_splitter_strategy2(pages):
    #load the text splitter
    text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
    chunked_text = text_splitter.split_documents(pages)
    return chunked_text

def text_splitter_strategy3(pages):
    all_text = " ".join([i.page_content for i in pages])
    chunks = []
    current_chunk = ""

    for sentence in sent_tokenize(all_text):
        if len(current_chunk) + len(sentence) <= 500:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    chunked_text = chunks
    return chunked_text

    

#generating text embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

chunked_text1 = text_splitter_strategy1(pages)
chunked_text2 = text_splitter_strategy2(pages)
chunked_text3 = text_splitter_strategy3(pages)



# This embeds all chunks AND stores them correctly in one call
vector_store1 = FAISS.from_documents(chunked_text1, embeddings)
vector_store2 = FAISS.from_documents(chunked_text2, embeddings)
vector_store3 = FAISS.from_texts(chunked_text3, embeddings)


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


strategy_names = [
    "Strategy 1 - CharacterTextSplitter",
    "Strategy 2 - RecursiveCharacterSplitter",
    "Strategy 3 - Sentence-Aware",
]
retrievers = [
    vector_store1.as_retriever(search_kwargs={"k": TOP_N}),
    vector_store2.as_retriever(search_kwargs={"k": TOP_N}),
    vector_store3.as_retriever(search_kwargs={"k": TOP_N}),
]

win_count = {name: 0 for name in strategy_names}

for q in QUESTIONS:
    print("\n" + "="*70)
    print(f"QUESTION: {q}")
    print("="*70)

    retrieved = [r.invoke(q) for r in retrievers]
    answers = [qa.invoke(q) for qa in [qa1, qa2, qa3]]

    for name, docs, answer in zip(strategy_names, retrieved, answers):
        print(f"\n  [{name}]")
        for i, doc in enumerate(docs, 1):
            text: str = clean(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            print(f"    Chunk {i} ({len(text)} chars): {text[:200]}...")
        print("\n")
        print(f"  LLM Response: {answer['result']}")
        print()

    # prompt user to pick best strategy for this question
    print("  Which strategy had the most relevant chunks? (1/2/3): ", end="")
    choice = input().strip()
    if choice in ("1", "2", "3"):
        win_count[strategy_names[int(choice)-1]] += 1

print("\n" + "="*70)
print("RELEVANCE SUMMARY (user votes)")
print("="*70)
for name, wins in win_count.items():
    bar = "#" * wins
    print(f"  {name}: {wins}/5 questions  {bar}")

best = max(win_count, key=lambda k: win_count[k])
print(f"\n  Best strategy: {best}")