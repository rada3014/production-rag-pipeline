
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from tqdm import tqdm
from dotenv import load_dotenv
from basic_rag import RAGPipeline
from config import PDF_PATH

load_dotenv()

EVAL_PATH    = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_baseline.json")

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

pipeline = RAGPipeline()
pipeline.load_pdf(PDF_PATH)

answers = []
for i in tqdm(data):
    result = pipeline.query(i["question"])
    answers.append({
        "question"        : i["question"],
        "ground_truth"    : i["ground_truth"],
        "pipeline_answer" : result["reranked_llm"]["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
    })

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(answers, f, indent=4)
