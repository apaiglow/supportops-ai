from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

SPLIT_FILES = {'train' : DATA_DIR / 'train.csv', 
               'validation' : DATA_DIR / 'validation.csv',
               'test' : DATA_DIR / 'test.csv'}
REQUIRED_COLUMNS = {'instruction', 'instruction_clean', 'intent', 'category', 'response'}

def main():

    splits = {}
    for name, path in SPLIT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing split file : {path}')
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f'{name} is missing columns {sorted(missing)}')
        if df['instruction_clean'].isna().any():
            raise ValueError(f'{name} contains null target labels')
        splits[name] = df
    all_intents = set(splits['train']['intent'].unique())
    print('SPLIT SIZES')
    for name, df in splits.items():
        print(f'{name:10} rows = {len(df):6,}'
              f'intents = {df['intent'].nunique():2}'
              f'categories={df['category'].nunique():2}')
    print('\nINTENT COUNTS BY SPLIT')
    intent_counts = pd.DataFrame({
        name : df['intent'].value_counts() for name, df in splits.items()
    }).fillna(0).astype(int)
    print(intent_counts.to_string())
    print('\nINTENT COVERAGE')
    for name, df in splits.items():
        missing_intents = all_intents - set(df['intent'].unique())
        print(
            f'{name:10} missing training intents : '
            f'{sorted(missing_intents)}'
        )
    print('\nEXAMPLE INPUT/TARGET PAIRS')
    print(splits['train'][['instruction_clean', 'intent']].head(5).to_string(index = False))

if __name__ == '__main__':
    main()