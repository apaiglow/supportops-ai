def build_support_prompt(
    customer_message: str,
    available_facts: str = "No account or order records are available.",
) -> str:

    if not isinstance(customer_message, str) or not customer_message.strip():
        raise ValueError("Customer message must be non-empty.")

    if not isinstance(available_facts, str) or not available_facts.strip():
        raise ValueError("Available facts must be a non-empty string.")

    return f"""
You are a professional customer support assistant.

Your task is to draft a concise, helpful response to the customer's message.

Rules:
1. Be polite, clear, and professional.
2. Use only the information in the available facts.
3. Never invent order details, account information, policies, or statuses.
4. Never claim that you checked a system or completed an action unless the
   available facts explicitly confirm it.
5. If information is missing, explain what needs to be verified and ask for
   the relevant details or direct the customer to the appropriate support
   process.
6. Do not refuse a request when you can provide a safe, useful next step.
7. Write 2–4 sentences. Do not include a placeholder name or signature.
8. Return only the customer-facing draft, without analysis or commentary.

Available facts:
{available_facts.strip()}

Customer message:
{customer_message.strip()}

Customer-facing draft:
""".strip()