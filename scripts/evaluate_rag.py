
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
from supportops.rag import generate_support_draft
from supportops.semantic_retrieval import build_semantic_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "support_cases.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "rag_evaluation.csv"

def main() -> None:
    if not TRAINING_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}"
        )
    if not CASES_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation cases not found: {CASES_PATH}"
        )
    with CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not cases:
        raise ValueError("The evaluation case list is empty.")
    print("Loading training data...")
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    print("Building semantic retriever...")
    retriever = build_semantic_retriever(training_data)
    records = []
    for case in cases:
        print(f"Evaluating {case['case_id']}...")
        started_at = perf_counter()
        try:
            result = generate_support_draft(
                customer_message=case["customer_message"],
                retriever=retriever,
                available_facts=case["available_facts"],
                top_k=3,
            )

            latency_seconds = round(perf_counter() - started_at, 3)
            examples = result["retrieved_examples"]

            records.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "customer_message": case["customer_message"],
                    "available_facts": case["available_facts"],
                    "draft": result["draft"],
                    "retrieved_intents": json.dumps(
                        [example["intent"] for example in examples]
                    ),
                    "retrieved_examples": json.dumps(
                        examples, ensure_ascii=False
                    ),
                    "latency_seconds": latency_seconds,
                    "review_focus": case["review_focus"],
                    "relevance_1_to_5": "",
                    "grounding_1_to_5": "",
                    "unsupported_claims": "",
                    "safe_when_facts_missing": "",
                    "review_notes": "",
                    "error": "",
                }
            )
        except Exception as exc:
            latency_seconds = round(perf_counter() - started_at, 3)

            records.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "customer_message": case["customer_message"],
                    "available_facts": case["available_facts"],
                    "draft": "",
                    "retrieved_intents": "[]",
                    "retrieved_examples": "[]",
                    "latency_seconds": latency_seconds,
                    "review_focus": case["review_focus"],
                    "relevance_1_to_5": "",
                    "grounding_1_to_5": "",
                    "unsupported_claims": "",
                    "safe_when_facts_missing": "",
                    "review_notes": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  Failed: {type(exc).__name__}: {exc}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(REPORT_PATH, index=False)
    successful = sum(not record["error"] for record in records)
    print(f"\nEvaluation complete: {successful}/{len(records)} cases succeeded.")
    print(f"Report saved to: {REPORT_PATH}")
    successful_latencies = [
        record["latency_seconds"]
        for record in records
        if not record["error"]
    ]
    if successful_latencies:
        print(
            "Mean latency: "
            f"{sum(successful_latencies) / len(successful_latencies):.2f}s"
        )
        print(f"Max latency: {max(successful_latencies):.2f}s")

if __name__ == "__main__":
    main()