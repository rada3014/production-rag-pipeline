PDF_PATH = "/Users/apple/Desktop/stuff/Projects/production-rag-pipeline/data/BERT_(language_model).pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "gpt-3.5-turbo"
CHAIN_TYPE = "stuff"

TOP_N = 3

QUESTIONS = [
    "What is BERT ?",
    "Who Developed BERT ?",
    "What are the advantages of BERT over others ?",
    "Before BERT what was used ? ",
    "What are some of the limitations of BERT ?"

]