
from pathlib import Path
import pandas as pd
from supportops.retrieval import build_retriever, search_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

def main():
    train = pd.read_csv(TRAIN_PATH)
    retriever = build_retriever(train)
    print("=" * 65)
    print("SupportOps AI — Past Example Search")
    print("=" * 65)
    print("Type a customer message to search for similar examples.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("Customer message: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break
        if not query:
            print("Please enter a non-empty message.\n")
            continue
        results = search_examples(retriever, query, top_k=5)
        if results.empty:
            print(
                "\nNo lexical matches found. "
                "Try rephrasing the message.\n"
            )
            continue
        print(f"\nTop {len(results)} retrieved examples:\n")
        for row in results.itertuples(index=False):
            print(f"Rank: {row.rank}")
            print(f"Similarity: {row.similarity:.4f}")
            print(f"Intent: {row.intent}")
            print(f"Past instruction: {row.instruction_clean}")
            print(f"Associated response: {row.response}")
            print("-" * 65)
        print()

if __name__ == "__main__":
    main()
