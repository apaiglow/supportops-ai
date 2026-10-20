# Semantic Retrieval: Evaluation and Comparison

## 1. Objective

The objective of Day 6 was to implement semantic retrieval and compare it against the lexical TF-IDF retrieval baseline developed on Day 5.

Both methods retrieve previous customer-support examples from the training split. Retrieved examples contain an instruction, intent label, and associated response.

The purpose of retrieval is to provide relevant examples that can support downstream customer-support workflows.

## 2. Retrieval Methods

### 2.1 Lexical retrieval

Lexical retrieval uses a TF-IDF vectorizer with the following configuration:

- Lowercase text.
- Unigrams and bigrams (`ngram_range=(1, 2)`).
- Minimum document frequency of 1 for the retrieval index.
- Sublinear term-frequency scaling.
- Cosine similarity for ranking.

The vectorizer is fitted on training instructions, and validation queries are transformed using the fitted vectorizer.

### 2.2 Semantic retrieval

Semantic retrieval uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` model.

Each instruction is encoded into a 384-dimensional sentence embedding. Embeddings are normalized, and cosine similarity is calculated using dot products.

The index contains training examples only. Validation instructions are encoded separately for evaluation.

### 2.3 Experimental consistency

Both methods use the same training and validation splits, the same intent labels for relevance judgments, and the same evaluation cutoffs: k = 1, 3, 5, and 10.

The held-out test split was not used for model selection.

## 3. Evaluation Metrics

- **Hit Rate@k:** Fraction of validation queries with at least one matching-intent example in the top k results.
- **Recall@k:** Fraction of all training examples with the query's intent that appear in the top k, averaged across validation queries.
- **MRR@k:** Mean reciprocal rank of the first matching-intent result, with zero contribution when no matching intent appears in the top k.

Relevance is defined using intent-label equality. These metrics measure intent retrieval performance under that proxy; they do not directly establish that a retrieved response is factually correct, safe, or useful.

## 4. Aggregate Results

| k | Lexical Hit Rate | Semantic Hit Rate | Hit Rate Change | Lexical Recall | Semantic Recall | Recall Change | Lexical MRR | Semantic MRR | MRR Change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9713 | 0.9913 | +0.0200 | 0.0014 | 0.0014 | 0.0000 | 0.9713 | 0.9913 | +0.0200 |
| 3 | 0.9920 | 0.9963 | +0.0042 | 0.0041 | 0.0043 | +0.0001 | 0.9807 | 0.9933 | +0.0126 |
| 5 | 0.9960 | 0.9985 | +0.0025 | 0.0068 | 0.0071 | +0.0003 | 0.9816 | 0.9938 | +0.0122 |
| 10 | 0.9968 | 0.9988 | +0.0020 | 0.0133 | 0.0141 | +0.0008 | 0.9817 | 0.9938 | +0.0121 |

*Changes are calculated as semantic minus lexical. Values are rounded to four decimal places.*

## 5. Interpretation

### 5.1 Hit Rate

Semantic retrieval achieved a Hit Rate@1 of 99.13%, compared with 97.13% for lexical retrieval. The improvement is 2.00 percentage points.

At k = 5, semantic retrieval achieved 99.85%, compared with 99.60% for lexical retrieval. The smaller difference reflects the strong performance of the lexical baseline at larger cutoffs.

### 5.2 Mean Reciprocal Rank

Semantic retrieval achieved an MRR@5 of 0.9938, compared with 0.9816 for lexical retrieval. This indicates that matching-intent examples tend to appear earlier in the semantic ranking under the chosen relevance definition.

### 5.3 Recall

Recall improved slightly for semantic retrieval at k = 3, 5, and 10. At k = 1, the rounded scores were tied.

Recall values are small because each query can have many relevant training examples, while the retriever returns at most ten results.

### 5.4 Current decision

Semantic retrieval is the preferred candidate based on the aggregate validation metrics.

This is a provisional decision, not proof that semantic retrieval is superior for every query or every downstream use case. Query-level wins and regressions, response quality, latency, memory usage, and operational requirements should also be considered.

## 6. Limitations

1. **Intent labels are a relevance proxy.** Similar intent does not guarantee that an example's response is suitable for the current query.
2. **Aggregate metrics hide individual failures.** A strong overall score can conceal difficult queries or confusion between related intents.
3. **The dataset contains templated instructions, placeholders, and spelling errors.** These characteristics may affect both retrieval methods.
4. **No human response-quality evaluation is included in these aggregate metrics.**
5. **The held-out test split remains unused.** It should be reserved for final evaluation after design choices are complete.
6. **Runtime costs differ.** Semantic retrieval requires loading a pretrained model and encoding text; lexical retrieval has a lighter runtime dependency footprint.

## 7. Reproducibility and Artifacts

Implementation:

- `src/supportops/retrieval.py`
- `src/supportops/semantic_retrieval.py`

Evaluation scripts:

- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_semantic_retrieval.py`
- `scripts/compare_retrieval_methods.py`

Reports:

- `reports/retrieval_metrics.csv`
- `reports/semantic_retrieval_metrics.csv`
- `reports/retrieval_method_comparison.csv`
- `reports/retrieval_query_results.csv`
- `reports/semantic_retrieval_query_results.csv`

The semantic model is downloaded separately and cached by the Sentence Transformers/Hugging Face tooling. A reproducible deployment should pin the model revision and dependency versions.

## 8. Next Steps

1. Compare individual query outcomes to identify semantic retrieval wins and regressions.
2. Inspect retrieved responses for factual suitability and relevance.
3. Continue with the SupportOps AI pipeline, retaining the ability to compare or switch retrieval methods.
4. Evaluate the final chosen approach on the held-out test split once model selection is complete.

## 9. Conclusion

Semantic retrieval outperformed the lexical TF-IDF baseline on Hit Rate and MRR at all evaluated cutoffs, with small improvements in Recall at k = 3, 5, and 10. It is the preferred candidate for the next development stage, subject to further query-level and response-quality evaluation.
