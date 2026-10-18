
# SupportOps AI — Dataset Audit

## 1. Dataset

Name: Bitext Customer Support Training Dataset

Local path: `data/raw/bitext_support.csv`

License: CDLA-Sharing-1.0, as identified during the initial dataset review. Recheck the accompanying license and attribution requirements before distributing derived artifacts.

Intended initial uses:
- Intent and category classification.
- Retrieval experiments over customer instructions.
- Response-drafting experiments using example responses.

## 2. Initial Structure

Rows: 26,872

Columns:
- `flag`
- `instruction`
- `category`
- `intent`
- `response`

All five columns were reported as string/object data types.

Missing values: 0

Exact duplicate rows: 0

Unique instructions: 24,635

Unique categories: 11

Unique intents: 27

Unique responses: 26,870

Duplicate instruction occurrences beyond the first: 2,237

## 3. Initial Interpretation

The dataset has useful supervised labels and example responses, making it a candidate for a multi-component NLP application.

The absence of missing values and exact duplicate rows does not guarantee that the data is clean or that evaluation will be reliable.

Repeated instructions require further investigation:
- Do repeated instructions have the same category and intent?
- Do their responses differ?
- Are repeated messages template variants?
- Could related examples appear across training and test sets?

The `flag` column also requires inspection before deciding whether it has predictive or operational value.

## 4. Data Quality and Evaluation Risks

### Synthetic data

The dataset is described as synthetic. Results on it cannot establish real-world performance on naturally occurring support conversations.

### Duplicate and template leakage

Random row-level splitting may place repeated or near-identical instructions in both training and test sets.

We must inspect duplicates and consider grouping related examples before finalizing the evaluation strategy.

### Response quality

Nearly all response strings are unique, but uniqueness is not evidence of factual correctness, usefulness, or suitability for retrieval-augmented generation.

Generated answers require separate evaluation.

### Label distribution

The counts of 11 categories and 27 intents are known. Their individual frequencies and any rare or imbalanced classes still need to be inspected.

## 5. Initial Decision

Keep the raw dataset unchanged and excluded from version control.

Proceed with this dataset as the initial candidate. Reconsider its suitability only if subsequent inspection reveals material problems with labels, provenance, leakage, or response quality.

## 6. Outstanding Questions

- What does `flag` represent?
- How are categories and intents distributed?
- Do duplicate instructions have conflicting labels or responses?
- Are near-duplicate templates present?
- What split strategy best estimates generalization?
- Which records can legitimately serve as retrieval evidence?

## 7. Audit Status

Initial structural audit: complete.

Detailed label, duplicate, and leakage investigation: pending.