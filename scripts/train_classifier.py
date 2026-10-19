from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from supportops.classifier import build_intent_classifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_COLUMN = 'instruction_clean'
TARGET_COLUMN = 'intent'

def main():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    validation = pd.read_csv(DATA_DIR / 'validation.csv')

    X_train = train[INPUT_COLUMN]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[INPUT_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    model = build_intent_classifier()
    model.fit(X_train, y_train)

    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "intent_classifier.joblib"
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")

    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)
    train_accuracy = accuracy_score(y_train, train_predictions)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    print('Intent Classifier Trained')
    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Intent classes:  {len(model.classes_)}")
    print(f"Training accuracy:   {train_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")

    print("\nSAMPLE PREDICTIONS")
    for text, actual, predicted in zip(
        X_validation.head(10),
        y_validation.head(10),
        validation_predictions[:10],
    ):
        print(f"\nInput:     {text}")
        print(f"Actual:    {actual}")
        print(f"Predicted: {predicted}")


if __name__ == "__main__":
    main()