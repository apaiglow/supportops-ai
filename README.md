# SupportOps AI

An intelligent customer-support platform combining NLP classification, semantic retrieval, local LLM response drafting, and production-style APIs.

## Technology Direction

- Python
- pandas and scikit-learn
- Text embeddings and semantic retrieval, if justified
- A small local instruction LLM, subject to feasibility testing
- FastAPI
- A lightweight web interface
- Automated testing and Docker, where feasible

## Repository Structure

```text
supportops-ai/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── scripts/
├── README.md
└── .gitignore
```

The structure will evolve as implementation requirements become clear.

## Baseline vs. RAG Evaluation

SupportOps AI compares direct LLM generation with retrieval-augmented generation (RAG) using semantic search over historical customer-support examples.

### Run the experiment

From the repository root, with the project virtual environment activated or available:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_baseline_vs_rag.py
```

The default local model is `qwen2.5:1.5b`. Ensure Ollama is running and the model is available locally.

### Summarize the results

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_rag_evaluation.py
```

### Outputs

- `reports/rag_evaluation/baseline_vs_rag.csv` — generated drafts, configuration, latency, and errors.
- `reports/rag_evaluation/retrieved_examples.csv` — historical examples returned for RAG requests.
- `reports/rag_evaluation/manual_scores.csv` — human response-quality assessments.
- `reports/rag_evaluation/evaluation_report.md` — generated experiment summary.

### Evaluation caveats

The experiment uses a small set of synthetic scenarios and is exploratory. Retrieval relevance does not guarantee factual correctness, and generated responses should be reviewed before being used with real customers. Results must be interpreted from the recorded outputs and completed manual assessments.
