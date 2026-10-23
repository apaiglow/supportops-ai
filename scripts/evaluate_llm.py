from pathlib import Path
from time import perf_counter
import pandas as pd
from supportops.llm import generate_response
from supportops.prompts import build_support_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "llm_quality_evaluation.csv"


TEST_CASES = [
    {
        "name": "Refund status",
        "message": "Has my refund been processed?",
        "facts": "No account or order records are available.",
    },
    {
        "name": "Order tracking",
        "message": "Where is my order?",
        "facts": "No tracking information or order status is available.",
    },
    {
        "name": "Order cancellation",
        "message": "I want to cancel my order.",
        "facts": "The order number and shipment status are unknown.",
    },
    {
        "name": "Known order facts",
        "message": "What is the status of my order?",
        "facts": "Order status: shipped. Tracking number: TRK12345.",
    },
    {
        "name": "Unsupported refund claim",
        "message": "Please confirm that my refund is approved.",
        "facts": "No refund decision or account records are available.",
    },
]

def main() -> None:
    records = []

    for case in TEST_CASES:
        prompt = build_support_prompt(
            customer_message=case["message"],
            available_facts=case["facts"],
        )

        start_time = perf_counter()

        try:
            response = generate_response(prompt)
            elapsed_seconds = perf_counter() - start_time
            error = ""
        except Exception as exc:
            elapsed_seconds = perf_counter() - start_time
            response = ""
            error = f"{type(exc).__name__}: {exc}"

        records.append(
            {
                "case_name": case["name"],
                "customer_message": case["message"],
                "available_facts": case["facts"],
                "latency_seconds": round(elapsed_seconds, 2),
                "response": response,
                "error": error,
                "instruction_following_score": "",
                "factual_restraint_score": "",
                "helpfulness_score": "",
                "review_notes": "",
            }
        )

        print(f"\nCase: {case['name']}")
        print(f"Latency: {elapsed_seconds:.2f} seconds")

        if error:
            print(f"Error: {error}")
        else:
            print(f"Response: {response}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(records)
    report.to_csv(REPORT_PATH, index=False)

    successful = report[report["error"] == ""]

    print("\n" + "=" * 60)
    print("LLM EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(report)}")
    print(f"Successful responses: {len(successful)}")
    print(f"Failed requests: {len(report) - len(successful)}")

    if not successful.empty:
        print(
            "Mean latency: "
            f"{successful['latency_seconds'].mean():.2f} seconds"
        )
        print(
            "Median latency: "
            f"{successful['latency_seconds'].median():.2f} seconds"
        )

    print(f"Saved evaluation report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()