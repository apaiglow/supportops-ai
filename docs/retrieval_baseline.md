# SupportOps AI — Retrieval Baseline

## 1. Objective

Build a lexical retrieval baseline that searches historical customer support examples and returns the most similar training instructions, along with their intent labels and associated responses.

This component retrieves candidate examples. It does not independently determine whether a response is correct or appropriate for a customer's specific situation.

## 2. Dataset and data separation

- Dataset: Bitext Customer Support Training Dataset.
- Indexed training examples: 18,800.
- Validation queries evaluated: 4,010.
- Retrieval corpus: training split only.
- Evaluation queries: validation split.
- Test split: not used for retrieval evaluation.

The TF-IDF vectorizer is fitted on training instructions. Validation instructions are transformed using the fitted vectorizer; they are not used to fit the retrieval index.

## 3. Retrieval implementation

The retriever is implemented in `src/supportops/retrieval.py`.

- Representation: word-level TF-IDF.
- N-grams: unigrams and bigrams (`ngram_range=(1, 2)`).
- Minimum document frequency: 1.
- Term-frequency scaling: sublinear TF.
- Similarity: cosine similarity.
- Default retrieval depth: 5 examples.

The search function transforms a query, computes its cosine similarity against indexed training instructions, ranks candidates by similarity, and returns the highest-scoring results with their associated intents and responses.

Queries with no vocabulary overlap can return an empty result set.

## 4. Evaluation methodology

The evaluation script is `scripts/evaluate_retrieval.py`.

For this baseline, a retrieved example is considered relevant when its intent label matches the validation query's known intent.

The reported metrics are averaged over validation queries:

- **Hit Rate@k:** proportion of queries with at least one same-intent example in the top k.
- **Recall@k:** for each query, the number of same-intent training examples retrieved in the top k divided by the total number of same-intent training examples in the indexed corpus; these per-query fractions are averaged.
- **MRR@k:** mean reciprocal rank of the first same-intent result, with zero assigned when no relevant result appears within k.

This relevance definition is an evaluation proxy. It does not establish semantic equivalence or response correctness.

## 5. Results

| k  |Hit Rate| Recall |   MRR  |
| 1  | 0.9713 | 0.0014 | 0.9713 |
| 3  | 0.9920 | 0.0041 | 0.9807 |
| 5  | 0.9960 | 0.0068 | 0.9816 |
| 10 | 0.9968 | 0.0133 | 0.9817 |

At k = 5, the retriever finds at least one same-intent example for 99.60% of validation queries.

Sixteen of the 4,010 validation queries have no same-intent result in the top five, giving a failure rate of approximately 0.3990%.

Of those 16 failures, three have their first same-intent result at rank 6. The remaining 13 have no same-intent result within the retrieved top 10.

## 6. Failure analysis

Failure cases are saved in `reports/retrieval_errors_top5.csv`.

Observed failure patterns include:

- Misspellings, such as `purchse` and `cancell`.
- Merged words, such as `contactsomebody` and `thepayments`.
- Vocabulary differences between messages expressing similar meanings.
- Relevant examples ranked just outside the top five.

The current word-level TF-IDF representation is sensitive to token overlap. Character n-grams and semantic embeddings are potential improvements, but they should be evaluated against this documented baseline rather than adopted without comparison.

## 7. Generated reports

- `reports/retrieval_metrics.csv`: aggregate retrieval metrics.
- `reports/retrieval_query_results.csv`: per-query evaluation results.
- `reports/retrieval_errors_top5.csv`: validation queries without a same-intent hit in the top five.

Scripts:

- `scripts/evaluate_retrieval.py`
- `scripts/analyze_retrieval_errors.py`
- `scripts/search_examples.py`

## 8. Limitations and next steps

1. Matching intent labels do not guarantee that retrieved responses are correct or suitable for the current customer.
2. The dataset contains misspellings, merged words, placeholders, and templated language.
3. High hit rate does not mean every retrieved candidate is useful.
4. Recall is low under the current definition because every training example sharing an intent counts as relevant, while only a small number can be returned.
5. The evaluation does not measure generation quality, factual grounding, or customer satisfaction.

**Next step:** Compare lexical TF-IDF retrieval with semantic embedding-based retrieval on the same train/validation split. Measure the same metrics and inspect failure cases before selecting the retrieval approach for downstream response generation.
