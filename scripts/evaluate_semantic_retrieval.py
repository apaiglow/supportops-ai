
from pathlib import Path
import numpy as np
import pandas as pd
from supportops.semantic_retrieval import build_semantic_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
METRICS_PATH = REPORTS_DIR / "semantic_retrieval_metrics.csv"
QUERY_RESULTS_PATH = REPORTS_DIR / "semantic_retrieval_query_results.csv"
INPUT_COLUMN = "instruction_clean"
TARGET_COLUMN = "intent"
TOP_K = 10
EVALUATION_K_VALUES = [1, 3, 5, 10]

def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    retriever = build_semantic_retriever(train)
    training_examples = retriever["examples"]
    training_embeddings = retriever["embeddings"]
    model = retriever["model"]
    print(f"Training examples indexed: {len(training_examples):,}")
    print(f"Validation queries: {len(validation):,}")
    print(f"Embedding dimensions: {training_embeddings.shape[1]}")
    validation_queries = (validation[INPUT_COLUMN].fillna("").astype(str).tolist())
    validation_embeddings = model.encode(validation_queries, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    relevant_counts = (training_examples[TARGET_COLUMN].value_counts().to_dict())
    query_results = []
    for query_index, (query, true_intent) in enumerate(
        zip(validation_queries, validation[TARGET_COLUMN].tolist())
    ):
        scores = validation_embeddings[query_index] @ training_embeddings.T
        ranked_indices = np.argsort(-scores, kind="stable")[:TOP_K]
        retrieved_intents = (training_examples.iloc[ranked_indices][TARGET_COLUMN].tolist())
        retrieved_scores = scores[ranked_indices]
        relevant_count = int(relevant_counts.get(true_intent, 0))
        matching_ranks = [
            rank
            for rank, intent in enumerate(retrieved_intents, start=1)
            if intent == true_intent
        ]
        first_relevant_rank = (
            matching_ranks[0] if matching_ranks else np.nan
        )
        result = {
            "query_index": query_index,
            "query": query,
            "true_intent": true_intent,
            "relevant_count": relevant_count,
            "first_relevant_rank": first_relevant_rank,
        }
        for rank in range(1, TOP_K + 1):
            result[f"top_{rank}_intent"] = retrieved_intents[rank - 1]
            result[f"top_{rank}_similarity"] = float(
                retrieved_scores[rank - 1]
            )
            result[f"top_{rank}_relevant"] = (
                retrieved_intents[rank - 1] == true_intent
            )
        for k in EVALUATION_K_VALUES:
            hits = sum(
                intent == true_intent
                for intent in retrieved_intents[:k]
            )
            result[f"hit_rate_at_{k}"] = int(hits > 0)
            result[f"recall_at_{k}"] = hits / relevant_count
            reciprocal_rank = (
                1 / first_relevant_rank
                if pd.notna(first_relevant_rank)
                and first_relevant_rank <= k
                else 0.0
            )
            result[f"mrr_at_{k}"] = reciprocal_rank
        query_results.append(result)
    query_results_df = pd.DataFrame(query_results)
    metric_rows = []

    for k in EVALUATION_K_VALUES:
        metric_rows.append(
            {
                "k": k,
                "hit_rate": query_results_df[
                    f"hit_rate_at_{k}"
                ].mean(),
                "recall": query_results_df[
                    f"recall_at_{k}"
                ].mean(),
                "mrr": query_results_df[
                    f"mrr_at_{k}"
                ].mean(),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    query_results_df.to_csv(QUERY_RESULTS_PATH, index=False)
    print("\nSEMANTIC RETRIEVAL METRICS")
    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "hit_rate": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "mrr": "{:.4f}".format,
            },
        )
    )
    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved query results to: {QUERY_RESULTS_PATH}")

if __name__ == "__main__":
    main()