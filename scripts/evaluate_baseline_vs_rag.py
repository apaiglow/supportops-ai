
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import pandas as pd
from supportops.llm import generate_response
from supportops.prompts import build_support_prompt
from supportops.rag import generate_support_draft
from supportops.semantic_retrieval import build_semantic_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "rag_evaluation"
DEFAULT_MODEL = os.getenv("SUPPORTOPS_LLM_MODEL", "qwen2.5:1.5b")
TEMPERATURE = 0.2
TIMEOUT = 120
SCENARIOS = [
    {
        "scenario_id": "refund_missing_info",
        "customer_message": "Where is my refund? I have been waiting for it.",
        "available_facts": "No account or order records are available.",
    },
    {
        "scenario_id": "order_verified_facts",
        "customer_message": "Has my order shipped? Can you give me the tracking number?",
        "available_facts": (
            "Order status: shipped. Tracking number: TRK12345."
        ),
    },
    {
        "scenario_id": "cancellation_policy_unknown",
        "customer_message": (
            "I want to cancel my order. Can you confirm whether "
            "I will receive a full refund?"
        ),
        "available_facts": (
            "No account or order records are available. "
            "No cancellation or refund policy has been provided."
        ),
    },
]

def save_results(
    rows: list[dict],
    retrieved_rows: list[dict],
    scores_template: list[dict],
) -> None:
    """Save progress after each generation so partial results survive errors."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "baseline_vs_rag.csv", index=False
    )
    pd.DataFrame(retrieved_rows).to_csv(
        OUTPUT_DIR / "retrieved_examples.csv", index=False
    )
    pd.DataFrame(scores_template).to_csv(
        OUTPUT_DIR / "manual_scores.csv", index=False
    )

def run_generation(
    scenario: dict,
    mode: str,
    retriever: dict,
    model: str,
    top_k: int,
) -> tuple[dict, list[dict]]:
    """Run one generation and return its record and retrieved examples."""
    started = time.perf_counter()
    retrieved_examples = []

    try:
        if mode == "baseline":
            prompt = build_support_prompt(
                customer_message=scenario["customer_message"],
                available_facts=scenario["available_facts"],
                retrieved_examples=[],
            )
            draft = generate_response(
                prompt=prompt,
                model=model,
                temperature=TEMPERATURE,
                timeout=TIMEOUT,
            )
        else:
            result = generate_support_draft(
                customer_message=scenario["customer_message"],
                retriever=retriever,
                available_facts=scenario["available_facts"],
                top_k=top_k,
                model=model,
                temperature=TEMPERATURE,
                timeout=TIMEOUT,
            )
            draft = result["draft"]
            retrieved_examples = result["retrieved_examples"]

    except Exception as exc:
        draft = ""
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = ""

    latency = time.perf_counter() - started

    record = {
        "scenario_id": scenario["scenario_id"],
        "mode": mode,
        "model": model,
        "temperature": TEMPERATURE,
        "top_k": top_k if mode == "rag" else 0,
        "customer_message": scenario["customer_message"],
        "verified_facts": scenario["available_facts"],
        "latency_seconds": round(latency, 3),
        "draft": draft,
        "error": error,
    }

    retrieval_records = [
        {
            "scenario_id": scenario["scenario_id"],
            **example,
        }
        for example in retrieved_examples
    ]

    return record, retrieval_records

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline generation with semantic RAG."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training split not found: {TRAIN_PATH}")

    print(f"Training split: {TRAIN_PATH}")
    print(f"Model: {args.model}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("Building semantic retriever from training data only...")

    train = pd.read_csv(TRAIN_PATH)
    retriever = build_semantic_retriever(train)

    print(f"Training examples indexed: {len(retriever['examples']):,}")
    print("Starting real-model comparison. This may take several minutes.")

    rows = []
    retrieved_rows = []
    score_template = []

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario['scenario_id']}")

        for mode in ("baseline", "rag"):
            print(f"  Running {mode}...")
            record, examples = run_generation(
                scenario=scenario,
                mode=mode,
                retriever=retriever,
                model=args.model,
                top_k=args.top_k,
            )
            rows.append(record)
            retrieved_rows.extend(examples)

            score_template.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "mode": mode,
                    "relevance_1_to_5": "",
                    "clarity_1_to_5": "",
                    "actionability_1_to_5": "",
                    "grounding_1_to_5": "",
                    "unsupported_claims": "",
                    "notes": "",
                }
            )

            save_results(rows, retrieved_rows, score_template)

            if record["error"]:
                print(f"    ERROR: {record['error']}")
            else:
                print(f"    Latency: {record['latency_seconds']} seconds")
                print(f"    Draft: {record['draft']}")

    print("\nEvaluation run finished.")
    print(f"Results saved under: {OUTPUT_DIR}")
    print("Review baseline_vs_rag.csv and complete manual_scores.csv.")


if __name__ == "__main__":
    main()
