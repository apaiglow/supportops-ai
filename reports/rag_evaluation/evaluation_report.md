# SupportOps AI: Baseline vs. RAG Evaluation

## Purpose

Compare direct LLM generation with generation augmented by semantic retrieval of historical support examples.

## Experiment setup

- Model and temperature are recorded per generation in the results CSV.
- Both modes use the same customer message and verified facts for each scenario.
- Baseline uses no retrieved examples.
- RAG uses the configured top-k semantic retrieval results.
- The retriever is built from `data/processed/train.csv` only.
- This is a small, exploratory evaluation, not a statistically conclusive benchmark.

## Run status

- Generation records: 6
- Baseline: 3/3 generation calls succeeded
- Rag: 3/3 generation calls succeeded

## Latency

| Mode | Successful calls | Mean seconds | Median seconds |
|---|---:|---:|---:|
| baseline | 3 | 11.759 | 7.569 |
| rag | 3 | 8.866 | 8.819 |

Latency includes per-request work performed inside the evaluation function. The one-time retriever build is excluded. Results from a few CPU-based calls should be treated as descriptive observations.

## Manual response-quality review

Manual scoring has not been completed. Fill in `manual_scores.csv`, then rerun this script.

### Unsupported claims

No valid Yes/No assessments recorded yet.

## Retrieval audit

- `cancellation_policy_unknown`: 3 retrieved examples recorded
- `order_verified_facts`: 3 retrieved examples recorded
- `refund_missing_info`: 3 retrieved examples recorded

## Limitations

- Only three synthetic customer scenarios were evaluated.
- Manual quality ratings are subjective and require consistent criteria.
- Similarity scores do not establish factual correctness.
- Historical examples can be irrelevant or contain claims that are not verified for the current customer.
- A successful model call does not guarantee a safe or correct draft.
- Results do not establish production readiness.

## Conclusion

Complete this section after reviewing the recorded outputs and manual scores. State whether RAG helped, hurt, or made little difference on these scenarios, citing specific examples. Do not generalize beyond this small evaluation.
