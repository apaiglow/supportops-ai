
from pathlib import Path
import joblib
import pandas as pd
from supportops.classifier import build_linear_svc_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
INPUT_COLUMN = "instruction_clean"
TARGET_COLUMN = "intent"

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    X_train = train[INPUT_COLUMN]
    y_train = train[TARGET_COLUMN]
    model = build_linear_svc_classifier()
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Training examples : {len(train):,}")
    print(f"Number of intents : {y_train.nunique()}")
    print(f"Saved model to : {MODEL_PATH}")

if __name__ == "__main__":
    main()
