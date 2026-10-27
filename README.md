# SupportOps AI

SupportOps AI is an AI-assisted customer-support drafting system. It combines
text preprocessing, intent classification, semantic retrieval over historical
support examples, and local LLM generation behind a FastAPI service.

The project focuses on measurable model evaluation, retrieval quality,
API reliability, containerized deployment, and documenting operational
limitations.

> **Status:** Portfolio/learning project, not a production-ready customer
> support agent. Generated drafts require human review.

## Architecture

1. **Data preparation:** Validate and preprocess the support dataset and
   create train/validation splits.
2. **Intent classification:** Train and compare text classifiers, including
   Logistic Regression and LinearSVC.
3. **Retrieval:** Compare lexical and embedding-based semantic retrieval over
   historical support examples.
4. **Response drafting:** Retrieve relevant examples and combine them with
   caller-provided facts in a prompt for a locally served LLM.
5. **API:** Expose health and generation endpoints through FastAPI, with
   request timing and error logging.
6. **Deployment:** Package the API in Docker and use Docker Compose to connect
   it to Ollama running on the host.

## Technology

- Python 3.12
- pandas and scikit-learn
- Text embeddings and semantic retrieval
- FastAPI and Pydantic
- Ollama with `qwen2.5:1.5b` as the configured default model
- pytest and Docker Compose

## Repository map

- `src/supportops/` — preprocessing, classification, retrieval, LLM
  integration, RAG pipeline, and API.
- `scripts/` — data preparation, model evaluation, retrieval evaluation,
  and test scripts.
- `data/` — dataset and evaluation inputs.
- `reports/` — classification, retrieval, RAG, and observability results.
- `docs/` — project documentation.

## Evaluation results

The following results are recorded from the project's validation and
retrieval experiments:

| Experiment | Result |
|---|---|
| Intent classification | LinearSVC validation accuracy and macro-F1 were approximately 0.993 |
| Semantic retrieval | Top-5 hit rate: 0.9985; MRR: 0.9938 |
| Lexical retrieval | Top-5 hit rate: 0.9960; MRR: 0.9816 |
| Automated tests | 23 passed in the final recorded test run |

These metrics measure different tasks. Classification scores do not measure
response quality, and retrieval hit rate does not prove that a generated
answer is correct or grounded.

### Baseline vs. RAG

The exploratory evaluation recorded three successful baseline generations
and three successful RAG generations:

| Mode | Mean latency | Median latency |
|---|---:|---:|
| Baseline | 11.759 s | 7.569 s |
| RAG | 8.866 s | 8.819 s |

These are observations from only three synthetic scenarios per mode, not a
statistically conclusive benchmark. Manual response-quality scoring remains
incomplete, so these results do not establish that RAG improves answer quality.

## Run locally

### Prerequisites

- Python 3.12 and a project virtual environment
- The project's dependencies installed from `requirements.txt`
- The processed training data at `data/processed/train.csv`
- Ollama running locally with the configured model available

### Start the API

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn supportops.api:app \
  --host 127.0.0.1 --port 8000
```

On startup, the API loads the training data and builds the semantic retriever.
The first startup can take time while the embedding model is initialized.

### Endpoints

- `GET /health` — reports API and retriever readiness.
- `POST /generate` — retrieves examples and generates a support-response draft.
- `GET /docs` — interactive API documentation.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "customer_message": "Where is my order?",
    "available_facts": "No verified order information is available.",
    "top_k": 3
  }'
```

`top_k` accepts values from 1 to 5. Supply verified facts when available;
the service cannot infer private account or order records that were not supplied.

### Run tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q scripts
```

### Run evaluation scripts

Baseline vs. RAG experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_baseline_vs_rag.py
PYTHONPATH=src .venv/bin/python scripts/summarize_rag_evaluation.py
```

RAG smoke evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_rag.py
```

Results are written under `reports/`. The evaluation inputs are small and
exploratory; inspect generated outputs and complete the manual scoring sheet
before making claims about response quality.

## Docker deployment

Prerequisites:

- Docker Desktop with Docker Compose
- Ollama running on the host
- The configured model available to that Ollama instance
- Enough memory for the embedding model and LLM inference

Build and start the API:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

The API is exposed at `http://localhost:8000`; documentation is at
`http://localhost:8000/docs`.

The Compose configuration uses `host.docker.internal:11434` to reach Ollama
on the host and configures `qwen2.5:1.5b` as the model. Stop the service with:

```bash
docker compose down
```

## Observability and known limitations

- The API measures request duration and the RAG pipeline records retrieval,
  generation, and total pipeline timings when returned by the pipeline.
- The health endpoint is not a generation-latency benchmark. A healthy API
  does not guarantee that Ollama is reachable or generation will succeed.
- A recorded successful API generation took approximately 55.9 seconds of
  API-reported latency (about 57.9 seconds total curl time) on the tested
  CPU-based setup.
- A generated draft was observed to contain unsupported template placeholders
  and an incomplete final sentence.
- The successful-generation timing log has not yet been verified after the
  application logger configuration fix.
- Manual response-quality scoring for the baseline-vs-RAG experiment is
  incomplete.
- Similarity scores and relevant retrieved examples do not guarantee factual
  correctness. Generated responses must be reviewed by a human before use.

## Project scope

This project demonstrates an end-to-end ML/NLP workflow: dataset preparation,
classifier comparison, retrieval evaluation, RAG integration, API development,
testing, observability, and Docker packaging. It is a portfolio project and
has not been validated for autonomous or production customer support.
