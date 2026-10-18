from pathlib import Path
import pandas as pd

DATA_PATH = Path('data/raw/bitext_support.csv')

df = pd.read_csv(DATA_PATH)


print("\n--- DATASET SHAPE ---")
print(f"Rows : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string(index=False))

print("\n--- MISSING VALUES ---")
print(df.isna().sum().sort_values(ascending=False))

print("\n--- EXACT DUPLICATE ROWS ---")
print(df.duplicated().sum())

for column in ["instruction", "category", "intent", "response"]:
    if column in df.columns:
        print(f"\n--- {column.upper()} ---")
        print(f"Unique values: {df[column].nunique():,}")
        print(df[column].value_counts().head(10).to_string())

if "instruction" in df.columns:
    print("\n--- DUPLICATE INSTRUCTIONS ---")
    print(df["instruction"].duplicated().sum())

if {"instruction", "response"}.issubset(df.columns):
    print("\n--- SAMPLE INSTRUCTION / RESPONSE PAIRS ---")

    sample = df.sample(n=min(5, len(df)), random_state=42)

    for _, row in sample.iterrows():
        print("\nINSTRUCTION:", row["instruction"])
        print("RESPONSE:", row["response"])
