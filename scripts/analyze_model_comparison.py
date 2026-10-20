from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / 'reports' / 'model_comparison_per_intent.csv'

def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f'Comparison report not found : {REPORT_PATH}'
            'Run scripts/compare_classifiers.py first'
        )
    results = pd.read_csv(REPORT_PATH)
    required_columns = {"model", "intent", "precision", "recall", "f1-score", "support"}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            f"Comparison report is missing columns : {sorted(missing_columns)}"
        )
    summary_labels = {"accuracy", "macro avg", "weighted avg"}
    results = results[~results["intent"].isin(summary_labels)]
    f1_comparison = results.pivot(index="intent", columns="model", values="f1-score")
    expected_models = {"logistic_regression", "linear_svc"}
    if not expected_models.issubset(f1_comparison.columns):
        raise ValueError(
            f"Expected model results for {sorted(expected_models)}; "
            f"found {list(f1_comparison.columns)}"
        )
    f1_comparison["f1_change"] = (f1_comparison["linear_svc"] - f1_comparison["logistic_regression"])
    f1_comparison = f1_comparison.sort_values("f1_change")
    print("\nINTENTS WITH THE LARGEST F1 REGRESSIONS")
    print(f1_comparison.head(10).to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nINTENTS WITH THE LARGEST F1 IMPROVEMENTS")
    print(
        f1_comparison.tail(10)
        .sort_values("f1_change", ascending=False)
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print("\nSUMMARY")
    print(f"Intents compared: {len(f1_comparison)}")
    print(
        "Intents improved : "
        f"{(f1_comparison['f1_change'] > 0).sum()}"
    )
    print(
        "Intents unchanged : "
        f"{(f1_comparison['f1_change'] == 0).sum()}"
    )
    print(
        "Intents regressed: "
        f"{(f1_comparison['f1_change'] < 0).sum()}"
    )
    output_path = PROJECT_ROOT / "reports" / "per_intent_f1_comparison.csv"
    f1_comparison.to_csv(output_path)
    print(f"\nSaved per-intent comparison to: {output_path}")

if __name__ == "__main__":
    main()