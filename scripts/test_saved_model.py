from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "intent_classifier.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}. "
            "Run scripts/train_classifier.py first."
        )

    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(DATA_PATH)

    examples = validation["instruction_clean"].head(5)
    predictions = model.predict(examples)

    print("SAVED MODEL RELOAD CHECK PASSED")
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Recognized intents: {len(model.classes_)}")

    for text, prediction in zip(examples, predictions):
        print(f"\nInput:      {text}")
        print(f"Predicted:  {prediction}")


if __name__ == "__main__":
    main()