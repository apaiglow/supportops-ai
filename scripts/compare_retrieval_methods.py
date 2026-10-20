
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
LEXICAL_PATH = REPORTS_DIR / "retrieval_metrics.csv"
SEMANTIC_PATH = REPORTS_DIR / "semantic_retrieval_metrics.csv"
OUTPUT_PATH = REPORTS_DIR / "retrieval_method_comparison.csv"

def main():
    for path in [LEXICAL_PATH, SEMANTIC_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")
    lexical = pd.read_csv(LEXICAL_PATH)
    semantic = pd.read_csv(SEMANTIC_PATH)
    required_columns = {"k", "hit_rate", "recall", "mrr"}
    for name, report in [("Lexical", lexical), ("Semantic", semantic),]:
        missing = required_columns - set(report.columns)
        if missing:
            raise ValueError(
                f"{name} report is missing columns: {sorted(missing)}"
            )
    comparison = lexical.merge(semantic, on="k", how="inner", suffixes=("_lexical", "_semantic"), validate="one_to_one")
    if comparison.empty:
        raise ValueError("No matching k values were found in the reports.")
    for metric in ["hit_rate", "recall", "mrr"]:
        comparison[f"{metric}_change"] = (
            comparison[f"{metric}_semantic"]
            - comparison[f"{metric}_lexical"]
        )
    comparison = comparison.sort_values("k").reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    display_columns = ["k", "hit_rate_lexical", "hit_rate_semantic", "hit_rate_change", "recall_lexical",
        "recall_semantic", "recall_change", "mrr_lexical", "mrr_semantic", "mrr_change"]
    print("\nLEXICAL VS SEMANTIC RETRIEVAL")
    print(
        comparison[display_columns].to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in display_columns
                if column != "k"
            },
        )
    )
    print(f"\nSaved comparison to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
