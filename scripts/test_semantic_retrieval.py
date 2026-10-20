
from pathlib import Path
import pandas as pd
from supportops.semantic_retrieval import (build_semantic_retriever, search_semantic_examples)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_QUERIES = [
    {
        "query": "order cancelation",
        "expected_intent": "cancel_order",
    },
    {
        "query": "I need help editing my order",
        "expected_intent": "change_order",
    },
    {
        "query": "Can you help me contact somebody?",
        "expected_intent": "contact_human_agent",
    },
    {
        "query": "What are your delivery time estimates?",
        "expected_intent": "delivery_period",
    },
    {
        "query": "I want to get my money back",
        "expected_intent": "get_refund",
    },
    {
        "query": "I would like to change my account type",
        "expected_intent": "switch_account",
    }
]


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )
    train = pd.read_csv(TRAIN_PATH)
    print("=" * 70)
    print("SupportOps AI — Semantic Retrieval Test")
    print("=" * 70)
    print(f"Training examples loaded : {len(train):,}")
    print("\nBuilding semantic index. The first run may download the model.")
    retriever = build_semantic_retriever(train)
    print(f"\nIndexed examples : {len(retriever['examples']):,}")
    print(f"Embedding matrix shape : {retriever['embeddings'].shape}")
    assert len(retriever["examples"]) == len(retriever["embeddings"])
    assert retriever["embeddings"].shape[1] == 384
    print("\nRunning test queries...\n")
    for case in TEST_QUERIES:
        query = case["query"]
        expected_intent = case["expected_intent"]
        results = search_semantic_examples(retriever, query, top_k=5)
        print("-" * 70)
        print(f"Query: {query}")
        print(f"Expected intent: {expected_intent}")
        if results.empty:
            print("No results returned.")
            continue
        retrieved_intents = results["intent"].tolist()
        hit = expected_intent in retrieved_intents
        print(f"Expected intent in top 5: {hit}")
        for row in results.itertuples(index=False):
            print(
                f"\nRank {row.rank} | "
                f"Similarity: {row.similarity:.4f}"
            )
            print(f"Intent: {row.intent}")
            print(f"Instruction: {row.instruction_clean}")
            print(f"Response: {row.response}")
    print("\nSemantic retrieval smoke test completed.")

if __name__ == "__main__":
    main()
