# SupportOps AI — Classifier Comparison and Model Selection

## 1. Objective

The objective of Day 4 was to compare two classical machine-learning classifiers for intent classification:

- Logistic Regression
- Linear Support Vector Classification (Linear SVC)

Both models use the same TF-IDF text representation and training/validation splits. This keeps the comparison focused on the classifier rather than changing several variables simultaneously.

## 2. Experimental Setup

**Task:** Predict the customer-support intent from the cleaned instruction text.

**Input feature:** `instruction_clean`
**Target label:** `intent`
**Text representation:** TF-IDF with lowercase normalization, unigram and bigram features, `min_df=2`, and sublinear term frequency.
**Data:** The existing training and validation splits, grouped by instruction to reduce leakage from repeated instructions.
**Evaluation metrics:**
- Accuracy
- Macro-F1
- Weighted-F1
- Training time

Macro-F1 is particularly useful because it gives each intent equal weight, while weighted-F1 accounts for the number of examples belonging to each intent.

## 3. Overall Results

On this validation split, Linear SVC improved accuracy by 0.0025 and macro-F1 by 0.0026. Its measured training time was also lower in this run.

These timing results are specific to the current environment and run; they should not be interpreted as a universal speed comparison.

## 4. Per-Intent Analysis

The per-intent F1 comparison found:

- 15 intents improved with Linear SVC.
- 10 intents had unchanged F1.
- 2 intents regressed.
- The largest reported F1 regression was approximately 0.0031.

This suggests that the overall improvement was not accompanied by a large reported F1 decline for any single intent. However, the aggregate comparison alone does not establish that every class is equally reliable. The per-intent report and confusion matrix remain useful for investigating difficult cases.

## 5. Model-Selection Decision

**Selected candidate: Linear SVC with TF-IDF.**

Linear SVC is the current candidate because it achieved higher validation accuracy and F1 scores than Logistic Regression in this experiment, while also training faster in the measured run.

This is a provisional model-selection decision based on validation data. The held-out test set has not been used to make this decision and should remain untouched until final evaluation.

## 6. Reproducibility and Saved Model

The reusable classifier builder is defined in `src/supportops/classifier.py`. The selected-model training script is `scripts/train_selected_model.py`.

Run from the project root:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_selected_model.py
```

The script trains the selected pipeline on the training split and saves the fitted TF-IDF vectorizer and Linear SVC model together at:

`models/intent_classifier.joblib`

The saved artifact is intended for local development and inference testing. It is ignored by Git; the code required to reproduce it is version-controlled instead.

To test the saved model:

```bash
PYTHONPATH=src .venv/bin/python scripts/test_saved_model.py
```

The comparison can be reproduced with:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_classifiers.py
```

## 7. Limitations and Next Steps

1. The results describe one train/validation split and should not be assumed to represent all possible customer-support messages.
2. The dataset contains templated instructions, spelling variations, and repeated instruction texts. Performance on this dataset may overestimate performance on naturally written, real-world support requests.
3. The two classifiers were compared with the same TF-IDF configuration. Other representations or models may perform differently.
4. High intent-classification scores do not establish that generated support responses are factually correct or safe.
5. The next step is to complete the classifier evaluation workflow, including final held-out test evaluation at the appropriate stage, and then proceed to retrieval and evidence-grounded response generation.

The classification model is only one component of SupportOps AI. The full system will also require retrieval evaluation, response-quality checks, uncertainty handling, and human review for cases where confidence or evidence is insufficient.
