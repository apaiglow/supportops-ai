import pandas as pd

REQUIRED_COLUMNS = {'flags', 'instruction', 'category', 'intent', 'response'}
TEXT_COLUMNS = {'instruction', 'category', 'intent', 'response'}

def validate_schema(df : pd.DataFrame) -> None:
    '''Raise an error if required dataset columns are missing.'''
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f'Missing required columns : {sorted(missing_columns)}')

def validate_data(df : pd.DataFrame) -> None:
    '''Valide schema and required text values without dropping rows.'''
    validate_schema(df)
    invalid_counts = {}
    for column in TEXT_COLUMNS:
        values = df[column].astype('string')
        null_mask = values.isna()
        blank_mask = values.str.strip().eq("").fillna(False)
        invalid_mask = null_mask | blank_mask
        invalid_counts[column] = int(invalid_mask.sum())
    invalid_counts = {
        column : count
        for column, count in invalid_counts.items()
        if count > 0
    }
    if invalid_counts:
        raise ValueError(f'Found null or blank values in required columns : {invalid_counts}')

def clean_text(text : str) -> str:
    '''Trim leading or trailing whitespace and collapse repeated whitespace.'''
    return " ".join(text.split())

def prepare_dataset(df : pd.DataFrame) -> pd.DataFrame:
    '''Validate a dataset and add a normalized instruction column.'''
    validate_data(df)
    prepared = df.copy()
    prepared['instruction_clean'] = prepared['instruction'].map(clean_text)
    return prepared
