
from supportops.prompts import build_support_prompt

examples = [
    {
        "rank": 1,
        "similarity": 0.8731,
        "instruction_clean": "where is my refund",
        "intent": "track_refund",
        "response": "You can check your refund status through the appropriate support process.",
    },
    {
        "rank": 2,
        "similarity": 0.8214,
        "instruction_clean": "when will my refund arrive",
        "intent": "track_refund",
        "response": "The refund timeline needs to be verified for your order.",
    },
]

prompt = build_support_prompt(
    customer_message="Where is my refund?",
    available_facts="No account or order records are available.",
    retrieved_examples=examples,
)

assert "HISTORICAL EXAMPLES" in prompt
assert "VERIFIED FACTS" in prompt
assert "where is my refund" in prompt
assert "Similarity score: 0.873" in prompt
assert "never treat historical examples" in prompt.lower()

empty_prompt = build_support_prompt(
    customer_message="Where is my refund?",
    retrieved_examples=[],
)

assert "No retrieved examples are available." in empty_prompt
assert "No account or order records are available." in empty_prompt

basic_prompt = build_support_prompt("I need help with my order.")

assert "I need help with my order." in basic_prompt

try:
    build_support_prompt("   ")
except ValueError:
    pass
else:
    raise AssertionError("Expected empty customer message to be rejected.")

print("All RAG prompt construction tests passed.")
