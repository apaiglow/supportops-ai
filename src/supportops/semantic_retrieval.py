from __future__ import annotations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REQUIRED_COLUMNS = {'instruction_clean', 'intent', 'response'}
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def build_semantic_retriever(training_examples : pd.DataFrame) -> dict:
    missing_columns = REQUIRED_COLUMNS - set(training_examples.columns)
    if missing_columns:
        raise ValueError(
            f'Training data is missing required columns : '
            f'{sorted(missing_columns)}'
        )
    examples = training_examples[['instruction_clean', 'intent', 'response']].copy()
    examples = examples.dropna(subset = ['instruction_clean', 'intent', 'response'])
    examples['instruction_clean'] = (examples['instruction_clean'].astype(str).str.strip())
    examples = examples[examples['instruction_clean'] != ''].reset_index(drop = True)
    if examples.empty:
        raise ValueError('No valid training examples remain for semantic retrieval.')
    model = SentenceTransformer(MODEL_NAME, device = 'cpu')
    embeddings = model.encode(
        examples['instruction_clean'].tolist(), batch_size = 32, show_progress_bar = True,
        convert_to_numpy = True, normalize_embeddings = True
    )
    return {
        "model": model,
        "embeddings": embeddings,
        "examples": examples,
    }

def search_semantic_examples(retriever : dict, query : str, top_k : int = 5) -> pd.DataFrame:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    model = retriever["model"]
    embeddings = retriever["embeddings"]
    examples = retriever["examples"]
    query_embedding = model.encode([query.strip()], convert_to_numpy = True, normalize_embeddings = True)
    scores = (query_embedding @ embeddings.T).ravel()
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
    results = examples.iloc[ranked_indices].copy()
    results["similarity"] = scores[ranked_indices]
    results.insert(0, "rank", range(1, len(results) + 1))
    return results[["rank", "similarity", "instruction_clean", "intent", "response"]].reset_index(drop=True)
