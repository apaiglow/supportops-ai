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

## Running the API

SupportOps AI exposes a FastAPI endpoint for generating customer-support response drafts using semantic retrieval and a local language model.

### Prerequisites

- Python 3.12 and the project's `.venv` environment
- The processed training dataset at `data/processed/train.csv`
- Ollama installed and running, with the configured model available locally

### Start the API

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn supportops.api:app --host 127.0.0.1 --port 8000
```

The application loads the training data and initializes the semantic retriever during startup.

### Available endpoints

- `GET /health` — basic health check.
- `POST /generate` — generates a support-response draft using retrieved examples and supplied facts.
- `GET /docs` — interactive API documentation.

### Generate a draft

Send a request to `POST /generate`:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "customer_message": "Where is my order?",
    "top_k": 3
  }'
```

The response includes the generated draft, retrieved examples, and request latency.

The `customer_message` field is required. `available_facts` can provide verified information for the model to use, and `top_k` controls the number of examples retrieved (1–5).

### Error handling

The API uses HTTP status codes to distinguish invalid requests, unavailable services, generation timeouts, and pipeline errors. See `/docs` for the request and response schemas.

### RAG Evaluation

SupportOps AI includes a repeatable smoke-evaluation script for inspecting generated support drafts.

Run the evaluation from the project root:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_rag.py
```

The script loads the processed training data, initializes the semantic retriever, runs the configured local language model on the fixed cases in `data/evaluation/support_cases.json`, and writes the results to `reports/rag_evaluation.csv`.

The report records generated drafts, retrieved examples, latency, and fields for manual quality review.

The evaluation set is small and is intended for qualitative smoke testing, not as a statistically representative benchmark. Retrieval metrics alone do not establish that generated responses are factually correct.
