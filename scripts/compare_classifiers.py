from pathlib import Path
from time import perf_counter
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from supportops.classifier import (build_intent_classifier, build_linear_svc_classifier)
from supportops.preprocessing import build_tfidf_vectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
INPUT_COLUMN = "instruction_clean"
TARGET_COLUMN = "intent"

def main():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    validation = pd.read_csv(DATA_DIR / 'validation.csv')
    X_train = train[INPUT_COLUMN]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[INPUT_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    models = {
        'logistic_regression' : build_intent_classifier(),
        'linear_svc' : build_linear_svc_classifier()
        }
    summary_rows = []
    class_reports = []
    for model_name, model in models.items():
        print(f'\nTraining {model_name}...')
        start_time = perf_counter()
        model.fit(X_train, y_train)
        training_seconds = perf_counter() - start_time
        predictions = model.predict(X_validation)
        accuracy = accuracy_score(y_validation, predictions)
        macro_f1 = f1_score(y_validation, predictions, average = 'macro', zero_division = 0)
        weighted_f1 = f1_score(y_validation, predictions, average = 'weighted', zero_division = 0)

        summary_rows.append({
                "model" : model_name,
                "accuracy" : accuracy,
                "macro_f1" : macro_f1,
                "weighted_f1" : weighted_f1,
                "training_seconds" : training_seconds,
            })
        report = classification_report(y_validation, predictions, labels=sorted(model.classes_), output_dict=True, zero_division=0)
        per_intent = pd.DataFrame(report).transpose()
        per_intent.index.name = "intent"
        per_intent = per_intent.reset_index()
        per_intent.insert(0, "model", model_name)
        class_reports.append(per_intent)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")
        print(f"Training seconds: {training_seconds:.2f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    combined_reports = pd.concat(class_reports, ignore_index=True)
    combined_reports.to_csv(REPORT_DIR / "model_comparison_per_intent.csv", index=False,)
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(REPORT_DIR / "model_comparison.csv")
    print(REPORT_DIR / "model_comparison_per_intent.csv")

if __name__ == "__main__":
    main()