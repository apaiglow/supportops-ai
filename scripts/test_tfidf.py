from pathlib import Path
import pandas as pd
from supportops.preprocessing import build_tfidf_vectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

def main():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    validation = pd.read_csv(DATA_DIR / 'validation.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train['instruction_clean'])
    X_validation = vectorizer.transform(validation['instruction_clean'])
    X_test = vectorizer.transform(test['instruction_clean'])

    assert X_train.shape[1] == X_validation.shape[1]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_validation.shape[1] == X_test.shape[1]

    train_nnz = X_train.getnnz(axis=1)
    validation_nnz = X_validation.getnnz(axis=1)
    test_nnz = X_test.getnnz(axis=1)

    print("\nZERO-VECTOR DIAGNOSTICS")
    print(f"Training zero vectors :   {(train_nnz == 0).sum():,}")
    print(f"Validation zero vectors : {(validation_nnz == 0).sum():,}")
    print(f"Test zero vectors : {(test_nnz == 0).sum():,}")

    zero_train_indices = (train_nnz == 0).nonzero()[0][:5]

    print("\nEXAMPLES OF ZERO-VECTOR TRAINING INSTRUCTIONS")
    for index in zero_train_indices:
        print(f"Row {index}: {train['instruction_clean'].iloc[index]!r}")

    print("\nZERO-VECTOR ROW DETAILS")

    for index in (train_nnz == 0).nonzero()[0]:
        row = train.iloc[index]

        print(f"\nInstruction: {row['instruction_clean']!r}")
        print(f"Intent:      {row['intent']}")
        print(f"Category:    {row['category']}")
    
    feature_names = vectorizer.get_feature_names_out()
    print('TF-IDF Checks Passed')
    print(f"Training matrix : {X_train.shape}")
    print(f"Validation matrix : {X_validation.shape}")
    print(f"Test matrix : {X_test.shape}")
    print(f'Vocabulary size : {len(feature_names):,}')
    print('\nTop TF-IDF Features for one training examples')
    row = X_train[0]
    top_features = sorted(zip(row.indices, row.data), key = lambda item : item[1], reverse = True)[:10]
    for feature_index, weight in top_features:
        print(f'{feature_names[feature_index]:25} {weight:.4f}')
    print('\nExample input : ')
    print(train['instruction_clean'].iloc[0])

if __name__ == '__main__':
    main()