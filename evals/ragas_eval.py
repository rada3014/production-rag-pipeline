import json
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

load_dotenv()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

METRICS = [faithfulness, answer_relevancy, context_precision]


def load_baseline_as_hf_dataset(INPUT_JSON_PATH):
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)


    rows = []

    for item in data:

        row = {
            "question": item["question"],
            "answer": item["pipeline_answer"],
            "contexts": item["retrieved_chunks"],
            "ground_truth": item["ground_truth"],
        }

        # Make sure contexts is always list[str]
        if isinstance(row["contexts"], str):
            row["contexts"] = [row["contexts"]]

        rows.append(row)


    dataset = Dataset.from_list(rows)

    return dataset

def main(INPUT_JSON_PATH,OUTPUT_JSON_PATH):
    
    dataset = load_baseline_as_hf_dataset(INPUT_JSON_PATH)

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()

    print("\nRAGAS row-level scores:")
    print(df)

    # Print average scores
    metric_cols = [
        col for col in df.columns
        if col in ["faithfulness", "answer_relevancy", "context_precision"]
    ]

    print("\nAverage scores:")
    avg_scores = {}

    for col in metric_cols:
        avg_scores[col] = float(df[col].mean())
        print(f"{col}: {avg_scores[col]:.4f}")

    targets = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
    }

    print("\nTarget check:")
    target_check = {}

    for metric, target in targets.items():
        score = avg_scores.get(metric)

        if score is None:
            target_check[metric] = {
                "score": None,
                "target": target,
                "passed": False,
                "note": "Metric column not found in result dataframe."
            }
            print(f"{metric}: NOT FOUND")
        else:
            passed = score >= target
            target_check[metric] = {
                "score": score,
                "target": target,
                "passed": passed,
            }
            print(
                f"{metric}: {score:.4f} "
                f"{'>=' if passed else '<'} {target} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    output = {
        "summary": avg_scores,
        "targets": target_check,
        "rows": df.to_dict(orient="records"),
    }

    Path("evals").mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved scores to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    INPUT_JSON_PATH = "results_baseline.json"
    OUTPUT_JSON_PATH = "ragas_scores_baseline.json"
    main(INPUT_JSON_PATH,OUTPUT_JSON_PATH)