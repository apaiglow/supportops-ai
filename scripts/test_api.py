from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from supportops.api import app
from pathlib import Path
import pandas as pd

@pytest.fixture
def client():
    """Create a test client without loading the real model or dataset."""

    with (
        patch("supportops.api.pd.read_csv", return_value=pd.DataFrame()),
        patch(
            "supportops.api.build_semantic_retriever",
            return_value={"mock": True},
        ),
    ):
        with TestClient(app) as test_client:
            yield test_client

    app.state.retriever = None

def fake_generate_support_draft(
    customer_message,
    retriever,
    available_facts,
    top_k,
):
    """Return a predictable response without calling the LLM."""
    return {
        "draft": "Thank you for contacting support. We can help with your request.",
        "retrieved_examples": [
            {
                "rank": 1,
                "similarity": 0.91,
                "instruction_clean": "cancel my order",
                "intent": "cancel_order",
                "response": "We can help you with order cancellation.",
            }
        ][:top_k],
    }

def test_health_endpoint_reports_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "retriever": "ready",
    }


def test_health_endpoint_reports_not_ready(client):
    app.state.retriever = None

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The retrieval system is not ready."
    )

@patch("supportops.api.generate_support_draft", side_effect=fake_generate_support_draft)
def test_generate_success(mock_generate, client):
    response = client.post(
        "/generate",
        json={
            "customer_message": "I want to cancel my order.",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["draft"].startswith("Thank you")
    assert len(body["retrieved_examples"]) == 1
    assert body["retrieved_examples"][0]["intent"] == "cancel_order"
    assert body["latency_seconds"] >= 0
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["customer_message"] == "I want to cancel my order."
    assert call_kwargs["available_facts"] == (
        "No account or order records are available."
    )
    assert call_kwargs["top_k"] == 3

def test_generate_rejects_blank_message(client):
    response = client.post(
        "/generate",
        json={"customer_message": "    "},
    )
    assert response.status_code == 422


def test_generate_rejects_top_k_out_of_range(client):
    for top_k in (0, 6):
        response = client.post(
            "/generate",
            json={
                "customer_message": "Where is my order?",
                "top_k": top_k,
            },
        )
        assert response.status_code == 422


def test_generate_rejects_blank_available_facts(client):
    response = client.post(
        "/generate",
        json={
            "customer_message": "Where is my order?",
            "available_facts": "   ",
        },
    )

    assert response.status_code == 422


def test_generate_returns_503_when_retriever_is_missing(client):
    app.state.retriever = None

    response = client.post(
        "/generate",
        json={"customer_message": "Where is my order?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "The retrieval system is not ready."


@patch(
    "supportops.api.generate_support_draft",
    side_effect=RuntimeError("Ollama connection failed"),
)
def test_generate_returns_503_when_llm_is_unavailable(mock_generate, client):
    response = client.post(
        "/generate",
        json={"customer_message": "Where is my order?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The response-generation service is unavailable."
    )


@patch(
    "supportops.api.generate_support_draft",
    side_effect=RuntimeError("The service did not respond within 60 seconds"),
)
def test_generate_returns_504_on_timeout(mock_generate, client):
    response = client.post(
        "/generate",
        json={"customer_message": "Where is my order?"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "The response-generation service timed out."
    )


@patch(
    "supportops.api.generate_support_draft",
    side_effect=ValueError("Invalid customer message"),
)
def test_generate_returns_422_for_pipeline_value_error(mock_generate, client):
    response = client.post(
        "/generate",
        json={"customer_message": "Where is my order?"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid customer message"

@patch(
    "supportops.api.generate_support_draft",
    side_effect=TypeError("Internal implementation detail"),
)
def test_generate_returns_500_for_unexpected_error(mock_generate, client):
    response = client.post(
        "/generate",
        json={"customer_message": "Where is my order?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An unexpected error occurred while generating the response."
    )
    assert "Internal implementation detail" not in response.text
    mock_generate.assert_called_once()

def test_generate_rejects_missing_customer_message(client):
    response = client.post("/generate", json={})

    assert response.status_code == 422

def test_generate_rejects_non_integer_top_k(client):
    response = client.post(
        "/generate",
        json={
            "customer_message": "Where is my order?",
            "top_k": "many"
        },
    )

    assert response.status_code == 422


def test_generate_rejects_malformed_json(client):
    response = client.post(
        "/generate",
        content='{"customer_message": ',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


@patch(
    "supportops.api.generate_support_draft",
    side_effect=fake_generate_support_draft
)
def test_generate_strips_customer_message_whitespace(mock_generate, client):
    response = client.post(
        "/generate",
        json={"customer_message": "  Where is my order?  "},
    )

    assert response.status_code == 200
    assert (
        mock_generate.call_args.kwargs["customer_message"]
        == "Where is my order?"
    )


def test_generate_rejects_non_string_customer_message(client):
    response = client.post(
        "/generate",
        json={"customer_message": 123},
    )

    assert response.status_code == 422

def test_startup_fails_when_training_data_is_missing():
    missing_path = Path("/tmp/supportops-missing-training-data.csv")

    with (
        patch("supportops.api.TRAINING_DATA_PATH", missing_path),
        pytest.raises(RuntimeError, match="Training data not found"),
    ):
        with TestClient(app):
            pass

def test_retriever_is_cleared_after_shutdown():
    with (
        patch("supportops.api.pd.read_csv", return_value=pd.DataFrame()),
        patch(
            "supportops.api.build_semantic_retriever",
            return_value={"mock": True},
        ),
    ):
        with TestClient(app):
            assert app.state.retriever == {"mock": True}

        assert app.state.retriever is None