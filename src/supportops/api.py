from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from supportops.rag import DEFAULT_FACTS, generate_support_draft
from supportops.semantic_retrieval import build_semantic_retriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the semantic retriever once when the API starts."""
    logger.info("Loading SupportOps AI training data.")

    if not TRAINING_DATA_PATH.is_file():
        raise RuntimeError(
            f"Training data not found at {TRAINING_DATA_PATH}"
        )

    training_data = pd.read_csv(TRAINING_DATA_PATH)

    logger.info(
        "Building semantic retriever from %s training rows.",
        len(training_data),
    )

    app.state.retriever = build_semantic_retriever(training_data)

    logger.info("Semantic retriever is ready.")

    try:
        yield
    finally:
        app.state.retriever = None
        logger.info("SupportOps AI API is shutting down.")


app = FastAPI(
    title="SupportOps AI",
    description="AI-assisted customer support response drafting API.",
    version="0.1.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    customer_message: str = Field(
        ...,
        min_length=1,
        description="The customer's message to respond to.",
    )
    available_facts: str | None = Field(
        default=None,
        description="Facts supplied by the caller for this draft.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of similar examples to retrieve.",
    )

    @field_validator("customer_message")
    @classmethod
    def validate_customer_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Customer message cannot be blank.")
        return value.strip()

    @field_validator("available_facts")
    @classmethod
    def validate_available_facts(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Available facts cannot be blank.")
        return value.strip() if value is not None else None


class RetrievedExample(BaseModel):
    rank: int
    similarity: float
    instruction_clean: str
    intent: str
    response: str


class GenerateResponse(BaseModel):
    draft: str
    retrieved_examples: list[RetrievedExample]
    latency_seconds: float


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the API is ready to handle generation requests."""
    retriever = getattr(app.state, "retriever", None)

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="The retrieval system is not ready.",
        )

    return {
        "status": "ok",
        "retriever": "ready",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> dict[str, Any]:
    """Generate a support draft using retrieved examples and the local LLM."""
    retriever = getattr(app.state, "retriever", None)

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="The retrieval system is not ready.",
        )

    started_at = perf_counter()

    try:
        result = generate_support_draft(
            customer_message=request.customer_message,
            retriever=retriever,
            available_facts=request.available_facts or DEFAULT_FACTS,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        elapsed = round(perf_counter() - started_at, 3)
        logger.warning(
            "Support draft generation failed: "
            "failure_type=RuntimeError request_seconds=%.3f error=%s",
            elapsed,
            exc,
        )
        message = str(exc).lower()

        if "timeout" in message or "did not respond within" in message:
            raise HTTPException(
                status_code=504,
                detail="The response-generation service timed out.",
            ) from exc

        logger.warning("Response generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The response-generation service is unavailable.",
        ) from exc
    except Exception:
        elapsed = round(perf_counter() - started_at, 3)
        logger.exception(
            "Unexpected error during response generation "
            "(request_seconds=%.3f).",
            elapsed,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the response.",
        ) from None

    latency_seconds = round(perf_counter() - started_at, 3)
    timings = result.get("timings")

    if timings is not None:
        logger.info(
            "Support draft generated successfully: "
            "retrieval_seconds=%.4f generation_seconds=%.4f "
            "pipeline_seconds=%.4f request_seconds=%.3f",
            timings["retrieval_seconds"],
            timings["generation_seconds"],
            timings["total_pipeline_seconds"],
            latency_seconds,
        )
    else:
        logger.info(
            "Support draft generated successfully: request_seconds=%.3f",
            latency_seconds,
        )

    return {
        "draft": result["draft"],
        "retrieved_examples": result["retrieved_examples"],
        "latency_seconds": latency_seconds,
    }
