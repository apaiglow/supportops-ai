from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'bitext_support.csv'

df = pd.read_csv(DATA_PATH)

required_columns = {'flags', 'instruction', 'category', 'intent', 'response'}

missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f'Missing required columns : {sorted(missing_columns)}')
print(f'Rows : {len(df):,}')
print(f'Columns : {list(df.columns)}')

for column in ['flags', 'category', 'intent']:
    print(f'\n---{column} : value counts ---')
    print(df[column].value_counts(dropna = False).to_string())
print('\n--- Examples by category ---')
for category in sorted(df['category'].dropna().unique()):
    example = df.loc[df['category'] == category, 'instruction'].iloc[0]
    print(f'{category} : {example[:180]}')

    
print("\n--- Duplicate instruction analysis ---")

duplicate_mask = df["instruction"].duplicated(keep=False)
duplicates = df.loc[duplicate_mask].copy()

print(f"Rows belonging to repeated instructions: {len(duplicates):,}")
print(
    "Distinct repeated instruction texts: "
    f"{duplicates['instruction'].nunique():,}"
)

if not duplicates.empty:
    groups = duplicates.groupby("instruction", dropna=False)

    intent_counts = groups["intent"].nunique(dropna=False)
    category_counts = groups["category"].nunique(dropna=False)
    response_counts = groups["response"].nunique(dropna=False)

    print(
        "Repeated instructions with multiple intents: "
        f"{int((intent_counts > 1).sum()):,}"
    )
    print(
        "Repeated instructions with multiple categories: "
        f"{int((category_counts > 1).sum()):,}"
    )
    print(
        "Repeated instructions with multiple responses: "
        f"{int((response_counts > 1).sum()):,}"
    )

    print("\n--- Sample duplicate groups ---")

    examples = (
        duplicates.groupby("instruction", dropna=False)
        .filter(lambda group: len(group) > 1)
        .groupby("instruction", dropna=False)
        .head(3)
        .head(15)
    )

    print(
        examples[
            ["instruction", "category", "intent", "response"]
        ].to_string(index=False)
    )
