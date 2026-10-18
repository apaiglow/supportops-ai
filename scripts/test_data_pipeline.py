from pathlib import Path
import pandas as pd
from supportops.data import clean_text, prepare_dataset, validate_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'bitext_support.csv'

def main():
    df = pd.read_csv(DATA_PATH)
    prepared = prepare_dataset(df)
    assert len(prepared) == len(df)
    assert "instruction_clean" in prepared.columns
    assert prepared["instruction"].equals(df["instruction"])

    assert clean_text("  need   help\tplease  ") == "need help please"

    invalid_df = df.head(3).copy()
    invalid_df.loc[invalid_df.index[0], "instruction"] = "   "

    try:
        validate_data(invalid_df)
    except ValueError:
        pass
    else:
        raise AssertionError("Blank instructions should fail validation.")

    print("All data pipeline checks passed.")
    print(f"Rows preserved: {len(prepared):,}")
    print(f"Cleaned instruction example: {prepared['instruction_clean'].iloc[0]}")


if __name__ == "__main__":
    main()
