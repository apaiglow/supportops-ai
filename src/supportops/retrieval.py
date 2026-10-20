from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {'instruction_clean', 'intent', 'response'}

def build_retriever(training_examples : pd.DataFrame) -> dict:
    missing_columns = REQUIRED_COLUMNS - set(training_examples.columns)
    if missing_columns:
        raise ValueError(
            f'Training data is missing required columns : {sorted(missing_columns)}'
        )
    examples = training_examples[['instruction_clean', 'intent', 'response']].copy()
    examples = examples.dropna(subset = ['instruction_clean', 'intent', 'response'])
    examples['instruction_clean'] = (examples['instruction_clean'].astype(str).str.strip())
    examples = examples[examples['instruction_clean'] != ''].reset_index(drop = True)
    if examples.empty:
        raise ValueError('No valid training examples remain for retrieval.')
    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 2), min_df = 1, sublinear_tf = True)
    instruction_matrix = vectorizer.fit_transform(examples['instruction_clean'])
    return {
        'vectorizer' : vectorizer,
        'instruction_matrix' : instruction_matrix,
        'examples' : examples
    }

def search_examples(retriever : dict, query : str, top_k : int = 5) -> pd.DataFrame:
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Query must be a non-empty string.')
    if top_k < 1:
        raise ValueError('top_k must be at least 1.')
    vectorizer = retriever['vectorizer']
    instruction_matrix = retriever['instruction_matrix']
    examples = retriever['examples']
    query_vector = vectorizer.transform([query.strip()])
    if query_vector.nnz == 0:
        return pd.DataFrame(columns = ['rank', 'similarity', 'instruction_clean', 'intent', 'response'])
    scores = cosine_similarity(query_vector, instruction_matrix).ravel()
    ranked_indices = np.argsort(-scores, kind = 'stable')[:top_k]
    results = examples.iloc[ranked_indices].copy()
    results['similarity'] = scores[ranked_indices]
    results = results[results['similarity'] > 0].copy()
    results.insert(0, 'rank', range(1, len(results) + 1))
    return results[['rank', 'similarity', 'instruction_clean', 'intent', 'response']].reset_index(drop = True)