
from pathlib import Path
import pandas as pd
from supportops.retrieval import build_retriever, search_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

def main():
    train = pd.read_csv(TRAIN_PATH)
    retriever = build_retriever(train)
    query = "My package has not arrived yet."
    results = search_examples(retriever, query, top_k=5)
    print(f"Training examples indexed: {len(retriever['examples']):,}")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")
    print()
    if results.empty:
        print("No lexical matches found.")
    else:
        print(results.to_string(index=False))

    try:
        search_examples(retriever, "", top_k=5)
    except ValueError:
        print("\nEmpty-query validation: passed")
    else:
        raise AssertionError("An empty query should raise ValueError.")

    try:
        search_examples(retriever, query, top_k=0)
    except ValueError:
        print("Invalid top_k validation: passed")
    else:
        raise AssertionError("top_k=0 should raise ValueError.")
    unknown_results = search_examples(retriever, "zzzxxyyqqv", top_k=5)
    assert unknown_results.empty
    print("Unknown-vocabulary query: passed")

if __name__ == "__main__":
    main()