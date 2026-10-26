
from time import perf_counter
from typing import Any
from supportops.llm import generate_response
from supportops.prompts import build_support_prompt
from supportops.semantic_retrieval import search_semantic_examples

DEFAULT_FACTS = "No account or order records are available."

def generate_support_draft(
    customer_message: str,
    retriever: dict,
    available_facts: str = DEFAULT_FACTS,
    top_k: int = 3,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 60,
) -> dict[str, Any]:
    total_started_at = perf_counter()
    if not isinstance(customer_message, str) or not customer_message.strip():
        raise ValueError("Customer message must be a non-empty string.")
    if not isinstance(available_facts, str) or not available_facts.strip():
        raise ValueError("Available facts must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    retrieval_started_at = perf_counter()
    results = search_semantic_examples(
        retriever=retriever,
        query=customer_message,
        top_k=top_k,
    )
    retrieval_seconds = perf_counter() - retrieval_started_at

    retrieved_examples = results.to_dict(orient="records")

    prompt = build_support_prompt(
        customer_message=customer_message,
        available_facts=available_facts,
        retrieved_examples=retrieved_examples,
    )

    generation_started_at = perf_counter()
    draft = generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    generation_seconds = perf_counter() - generation_started_at

    total_pipeline_seconds = perf_counter() - total_started_at

    return {
        "draft": draft,
        "retrieved_examples": retrieved_examples,
        "timings": {
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "total_pipeline_seconds": round(total_pipeline_seconds, 4),
        },
    }
