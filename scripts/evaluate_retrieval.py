
from pathlib import Path
import pandas as pd
from supportops.retrieval import build_retriever, search_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    required_columns = {"instruction_clean", "intent"}
    missing = required_columns - set(validation.columns)
    if missing:
        raise ValueError(
            f"Validation data is missing columns: {sorted(missing)}"
        )
    retriever = build_retriever(train)
    relevant_counts = train["intent"].value_counts().to_dict()
    query_records = []
    for query_row in validation.itertuples(index = False):
        query = query_row.instruction_clean
        true_intent = query_row.intent
        results = search_examples(retriever, query, top_k = MAX_K)
        matching_ranks = [rank for rank, intent in zip(results["rank"], results["intent"])
            if intent == true_intent
        ]
        first_relevant_rank = (matching_ranks[0] if matching_ranks else None)
        total_relevant = relevant_counts.get(true_intent, 0)
        record = {
            "query": query,
            "true_intent": true_intent,
            "first_relevant_rank": first_relevant_rank,
            "relevant_examples_in_train": total_relevant,
        }
        for k in K_VALUES:
            relevant_in_top_k = sum(rank <= k for rank in matching_ranks)
            record[f"hit_rate_at_{k}"] = int(relevant_in_top_k > 0)
            record[f"recall_at_{k}"] = (
                relevant_in_top_k / total_relevant
                if total_relevant > 0
                else 0.0
            )
            record[f"mrr_at_{k}"] = (
                1.0 / first_relevant_rank
                if first_relevant_rank is not None
                and first_relevant_rank <= k
                else 0.0
            )
        query_records.append(record)
    per_query = pd.DataFrame(query_records)
    metric_records = []
    for k in K_VALUES:
        metric_records.append(
            {
                "k": k,
                "hit_rate": per_query[f"hit_rate_at_{k}"].mean(),
                "recall": per_query[f"recall_at_{k}"].mean(),
                "mrr": per_query[f"mrr_at_{k}"].mean(),
            }
        )
    metrics = pd.DataFrame(metric_records)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "retrieval_metrics.csv"
    query_path = REPORT_DIR / "retrieval_query_results.csv"
    metrics.to_csv(metrics_path, index=False)
    per_query.to_csv(query_path, index=False)
    print(f"Training examples indexed: {len(retriever['examples']):,}")
    print(f"Validation queries evaluated: {len(validation):,}")
    print("\nRETRIEVAL EVALUATION")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved aggregate metrics to: {metrics_path}")
    print(f"Saved per-query results to: {query_path}")

if __name__ == "__main__":
    main()
