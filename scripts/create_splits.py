from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from supportops.data import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'bitext_support.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
RANDOM_STATE = 27

def split_by_group(df, test_size, random_state):

    splitter = GroupShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    train_indices, holdout_indices = next(splitter.split(df, groups = df['instruction_clean']))
    return (df.iloc[train_indices].copy(), df.iloc[holdout_indices].copy())

def verify_splits(original, train, validation, test):

    splits = {'train' : train, 'validation' : validation, 'test' : test}
    row_indices = [set(split.index) for split in splits.values()]
    assert len(train) + len(validation) + len(test) == len(original)
    assert row_indices[0].isdisjoint(row_indices[1])
    assert row_indices[0].isdisjoint(row_indices[2])
    assert row_indices[1].isdisjoint(row_indices[2])

    instruction_groups = {name : set(split['instruction_clean']) for name, split in splits.items()}
    names = list(instruction_groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = (instruction_groups[names[i]] & instruction_groups[names[j]])
            assert not overlap, (f'Instruction leakage between {names[i]} and {names[j]}')

    expected_intents = set(original['intent'].unique())
    expected_categories = set(original['category'].unique())
    for name, split in splits.items():
        missing_intents = expected_intents - set(split['intent'].unique())
        missing_categories = (expected_categories - set(split['category'].unique()))
        if missing_intents or missing_categories:
            raise ValueError (
                f'{name} is missing labels.'
                f'Intents : {sorted(missing_intents)}'
                f'Categories : {sorted(missing_categories)}'
            )

def main():
    df = pd.read_csv(DATA_PATH)
    prepared = prepare_dataset(df)
    train, holdout = split_by_group(prepared, test_size = 0.30, random_state = RANDOM_STATE)
    validation, test = split_by_group(holdout, test_size = 0.50, random_state = RANDOM_STATE + 1)
    verify_splits(prepared, train, validation, test)
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    for name, split in {'train' : train, 'validation' : validation, 'test' : test}.items():
        output_path = OUTPUT_DIR / f'{name}.csv'
        split.to_csv(output_path, index = False)
    print('All split verification checks passed.')
    print(f'Total rows : {len(prepared):,}')
    for name, split in {'train' : train, 'validation' : validation, 'test' : test}.items():
        percentage = len(split) / len(prepared) * 100
        print(
            f'{name.capitalize():10}'
            f'{len(split):6,} rows ({percentage:5.2f}%) | '
            f'{split['intent'].nunique()} intents | '
            f'{split['category'].nunique()} categories'
        )
    print('\nInstruction-group overlap : 0')
    print(f'Saved splits to : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}')

if __name__ == '__main__':
    main()