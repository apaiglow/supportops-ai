
# SupportOps AI — Intent Classification Baseline

## Objective

Predict the intent of a customer support message from its cleaned instruction text.

## Dataset and splits

- Dataset: Bitext Customer Support Training Dataset.
- Splits: pre-created train, validation, and test CSV files.
- Split strategy: group by `instruction_clean` so identical cleaned
  instructions cannot cross split boundaries.
- Input: `instruction_clean`.
- Target: `intent`.
- The test split is reserved for final evaluation.

## Model

A scikit-learn Pipeline containing:

1. TfidfVectorizer
   - lowercase = true
   - ngram_range = (1, 2)
   - min_df = 2
   - sublinear_tf = true
2. LogisticRegression
   - max_iter = 1000
   - random_state = 42

The vectorizer is fitted only on the training data.

## Evaluation

Validation metrics:
- Accuracy
- Macro-F1
- Weighted-F1
- Per-intent precision, recall, and F1
- Confusion matrix
- Misclassified examples

Generated reports:
- `reports/classification_report.csv`
- `reports/confusion_matrix.csv`
- `reports/validation_errors.csv`

## Known data and model limitations

- Two training instructions produced zero TF-IDF vectors:
  `contactingassistant` and `makepurchase`.
- These examples were not silently removed.
- The dataset contains repeated instruction text with multiple responses.
- The dataset contains synthetic/template-like messages and placeholders.
- Validation performance does not establish production readiness.
- The held-out test split has not yet been used for final evaluation.

## Reproducibility

Run from the project root:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_classifier.py
PYTHONPATH=src .venv/bin/python scripts/evaluate_classifier.py
PYTHONPATH=src .venv/bin/python scripts/test_saved_model.py
```
