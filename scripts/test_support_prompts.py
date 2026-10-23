from supportops.llm import generate_response
from supportops.prompts import build_support_prompt

TEST_CASES = [
    {
        "name": "Refund status",
        "message": "Has my refund been processed?",
        "facts": "No account or order records are available."
    },
    {
        "name": "Order tracking",
        "message": "Where is my order?",
        "facts": "No tracking information or order status is available."
    },
    {
        "name": "Order cancellation",
        "message": "I want to cancel my order.",
        "facts": "The order number and shipment status are unknown."
    },
    {
        "name": "Known fact",
        "message": "What is the status of my order?",
        "facts": "Order status: shipped. Tracking number: TRK12345."
    },
    {
        "name": "Unsupported refund claim",
        "message": "Please confirm that my refund is approved.",
        "facts": "No refund decision or account records are available."
    },
]

def main() -> None:
    for index, case in enumerate(TEST_CASES, start=1):
        prompt = build_support_prompt(
            customer_message=case["message"],
            available_facts=case["facts"]
        )

        response = generate_response(prompt)

        print("\n" + "=" * 70)
        print(f"Test {index}: {case['name']}")
        print("-" * 70)
        print(f"Customer: {case['message']}")
        print(f"Available facts: {case['facts']}")
        print(f"Draft: {response}")

if __name__ == "__main__":
    main()