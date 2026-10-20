from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
CLASSIFICATION_REPORT = REPORT_DIR / "classification_report.csv"
CONFUSION_MATRIX = REPORT_DIR / "confusion_matrix.csv"
VALIDATION_ERRORS = REPORT_DIR / "validation_errors.csv"

def main():

    required_files = [CLASSIFICATION_REPORT, CONFUSION_MATRIX, VALIDATION_ERRORS]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f'Required report is missing : {path}'
                'Run scripts/evaluate_classifier.py first'
            )
    report = pd.read_csv(CLASSIFICATION_REPORT, index_col = 0)
    confusion = pd.read_csv(CONFUSION_MATRIX, index_col = 0)
    errors = pd.read_csv(VALIDATION_ERRORS)
    summary_rows = {'accuracy', 'macro avg', 'weighted avg'}
    intent_report = report.loc[~report.index.isin(summary_rows)].copy()
    print('Baseline Error Analysis')
    print('\nOverall Validation Metrics')
    for metric in ['accuracy', 'macro avg', 'weighted avg']:
        if metric in report.index:
            print(f'\n{metric}')
            print(report.loc[metric].to_string())
    print('\n10 Intents with the lowest F1')
    print(intent_report.sort_values('f1-score')
          [['precision', 'recall', 'f1-score', 'support']].head(10)
          .to_string(float_format = lambda value : f'{value:.3f}'))

    off_diagonal = confusion.copy()
    for label in off_diagonal.index:
        if label in off_diagonal.columns:
            off_diagonal.loc[label, label] = 0
    confusion_pairs = (off_diagonal.stack().rename("count").reset_index())
    confusion_pairs.columns = ["actual_intent", "predicted_intent", "count"]
    confusion_pairs = confusion_pairs[confusion_pairs["count"] > 0].sort_values("count", ascending=False)

    print("\nTOP 15 CONFUSION PAIRS")
    if confusion_pairs.empty:
        print("No off-diagonal errors found.")
    else:
        print(confusion_pairs.head(15).to_string(index = False))

    print("\nVALIDATION ERROR SUMMARY")
    print(f"Total misclassified rows: {len(errors):,}")
    if not errors.empty:
        print("\nMOST COMMON ACTUAL → PREDICTED ERROR PAIRS")
        error_pairs = (
            errors.groupby(["actual_intent", "predicted_intent"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(error_pairs.head(10).to_string(index=False))
        print("\nEXAMPLE MISCLASSIFICATIONS")
        print(errors.head(10).to_string(index=False))


if __name__ == "__main__":
    main()