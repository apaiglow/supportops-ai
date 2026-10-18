# SupportOps AI — Data Preparation and Splitting

## 1. Dataset Overview

SupportOps AI uses the Bitext Customer Support Training Dataset.

- Original file: `data/raw/bitext_support.csv`
- Original rows: 26,872
- Columns: `flags`, `instruction`, `category`, `intent`, `response`
- Categories: 11
- Intents: 27
- Unique instruction texts: 24,635

The dataset contains repeated instruction texts and multiple responses for some repeated instructions. The data also includes synthetic-looking templates, placeholders such as `{{Order Number}}`, and occasional spelling errors.

## 2. Schema and Data Validation

The data pipeline checks that the required columns exist and that the required text fields contain no null or blank values.

Invalid records are rejected with a `ValueError`; they are not silently dropped.

The pipeline preserves the original rows and original instruction text.

## 3. Text Normalization

A new column, `instruction_clean`, is created from `instruction`.

Normalization:
- Removes leading and trailing whitespace.
- Collapses repeated whitespace into a single space.
- Does not lowercase text, remove punctuation, correct spelling, remove stopwords, or replace placeholders.

The original `instruction` column is retained for traceability.

## 4. Splitting Strategy

The dataset is split into training, validation, and test sets using `GroupShuffleSplit`.

The grouping key is `instruction_clean`, ensuring that identical normalized instructions remain within a single split.

The intended proportions are approximately:
- Training: 70%
- Validation: 15%
- Test: 15%

The split sizes may differ from these exact proportions because groups contain different numbers of rows.

A fixed random seed is used to make the splitting procedure reproducible.

## 5. Verification

The splitting script checks that:
- The total number of rows is preserved.
- No original row index occurs in multiple splits.
- No normalized instruction occurs in multiple splits.
- All original intents and categories are represented in every split.

The script stops with an error if a required verification check fails.

## 6. Limitations

This strategy prevents exact normalized-instruction leakage but does not prevent semantically similar instructions from appearing in different splits.

The split is group-aware but not explicitly stratified. Label proportions may therefore vary across splits, even when every intent and category is represented.

Repeated instructions may have different responses. Consequently, intent classification and response-generation evaluation require different considerations.

The `flags` column is retained but is not used as a model feature until its meaning is understood.

## 7. Output Files

The processed splits are written to:
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

The original raw dataset is not modified.

## 8. Reproducibility

The splitting script uses a fixed random seed and deterministic group assignment. Re-running it with the same input data, code, library versions, and environment should reproduce the same split assignments.

Dataset licensing and attribution requirements must be checked before distributing the dataset or derived artifacts.