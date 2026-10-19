from pathlib import Path
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from supportops.classifier import build_intent_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
INPUT_COLUMN = "instruction_clean"
TARGET_COLUMN = "intent"

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train[INPUT_COLUMN]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[INPUT_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    model = build_intent_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(y_validation, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_validation, predictions, average="weighted", zero_division=0)
    print("Validation Metrics")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Weighted-F1 : {weighted_f1:.4f}")
    report = classification_report(y_validation, predictions, labels=sorted(model.classes_), output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    print("\nPER-INTENT CLASSIFICATION REPORT")
    print(report_df.to_string(float_format=lambda value: f"{value:.3f}"))
    labels = sorted(model.classes_)
    matrix = confusion_matrix(y_validation, predictions, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    errors_df = pd.DataFrame(
        {
            "instruction": X_validation.to_numpy(),
            "actual_intent": y_validation.to_numpy(),
            "predicted_intent": predictions,
        }
    )
    errors_df = errors_df[errors_df["actual_intent"] != errors_df["predicted_intent"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "classification_report.csv")
    confusion_df.to_csv(REPORT_DIR / "confusion_matrix.csv")
    errors_df.to_csv(REPORT_DIR / "validation_errors.csv", index=False)

    print(f"\nMisclassified validation rows: {len(errors_df):,}")
    print(f"Reports saved to: {REPORT_DIR}")

    print("\nSAMPLE MISCLASSIFICATIONS")
    if errors_df.empty:
        print("No validation errors found.")
    else:
        print(errors_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()