
import unittest
from unittest.mock import patch
import pandas as pd
from supportops.rag import generate_support_draft

def make_examples():
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "similarity": 0.87,
                "instruction_clean": "where is my refund",
                "intent": "track_refund",
                "response": "Check the appropriate support process for your refund status.",
            },
            {
                "rank": 2,
                "similarity": 0.81,
                "instruction_clean": "when will my refund arrive",
                "intent": "track_refund",
                "response": "The refund timeline needs to be verified.",
            },
        ], columns=["rank", "similarity", "instruction_clean", "intent", "response"]
    )


class TestRAGPipeline(unittest.TestCase):

    @patch("supportops.rag.generate_response")
    @patch("supportops.rag.search_semantic_examples")
    def test_retrieval_prompt_and_generation_are_connected(self, mock_search, mock_generate):
        mock_search.return_value = make_examples()
        mock_generate.return_value = (
            "I can help you check your refund status. "
            "Please use the appropriate support process to verify it."
        )

        result = generate_support_draft(
            customer_message="Where is my refund?",
            retriever={"test": "retriever"},
            available_facts="No account or order records are available.",
            top_k=3,
        )

        mock_search.assert_called_once_with(
            retriever={"test": "retriever"},
            query="Where is my refund?",
            top_k=3,
        )

        prompt_sent_to_llm = mock_generate.call_args.kwargs["prompt"]
        self.assertIn("where is my refund", prompt_sent_to_llm)
        self.assertIn("HISTORICAL EXAMPLES", prompt_sent_to_llm)
        self.assertIn("VERIFIED FACTS", prompt_sent_to_llm)
        self.assertEqual(result["draft"], mock_generate.return_value)
        self.assertEqual(len(result["retrieved_examples"]), 2)

    @patch("supportops.rag.generate_response")
    @patch("supportops.rag.search_semantic_examples")
    def test_empty_retrieval_still_generates_a_draft(
        self, mock_search, mock_generate
    ):
        mock_search.return_value = pd.DataFrame(
            columns=[
                "rank",
                "similarity",
                "instruction_clean",
                "intent",
                "response",
            ]
        )
        mock_generate.return_value = (
            "I can help you check this. "
            "Please contact support to verify the relevant details."
        )

        result = generate_support_draft(
            customer_message="Where is my refund?",
            retriever={"test": "retriever"},
        )

        prompt_sent_to_llm = mock_generate.call_args.kwargs["prompt"]

        self.assertIn("No retrieved examples are available.", prompt_sent_to_llm)
        self.assertEqual(result["retrieved_examples"], [])
        mock_generate.assert_called_once()

    @patch("supportops.rag.generate_response")
    @patch("supportops.rag.search_semantic_examples")
    def test_verified_facts_are_passed_to_prompt(
        self, mock_search, mock_generate
    ):
        mock_search.return_value = make_examples()
        mock_generate.return_value = (
            "Your order is marked as shipped. "
            "Your tracking number is TRK12345."
        )

        facts = "Order status: shipped. Tracking number: TRK12345."

        generate_support_draft(
            customer_message="Where is my order?",
            retriever={"test": "retriever"},
            available_facts=facts,
        )

        prompt_sent_to_llm = mock_generate.call_args.kwargs["prompt"]
        self.assertIn(facts, prompt_sent_to_llm)

    @patch("supportops.rag.generate_response")
    @patch("supportops.rag.search_semantic_examples")
    def test_empty_customer_message_is_rejected(
        self, mock_search, mock_generate
    ):
        with self.assertRaises(ValueError):
            generate_support_draft(
                customer_message="   ",
                retriever={"test": "retriever"},
            )

        mock_search.assert_not_called()
        mock_generate.assert_not_called()

    @patch("supportops.rag.generate_response")
    @patch("supportops.rag.search_semantic_examples")
    def test_generation_error_is_not_silently_hidden(
        self, mock_search, mock_generate
    ):
        mock_search.return_value = make_examples()
        mock_generate.side_effect = RuntimeError("Ollama is unavailable")

        with self.assertRaisesRegex(RuntimeError, "Ollama is unavailable"):
            generate_support_draft(
                customer_message="Where is my refund?",
                retriever={"test": "retriever"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
