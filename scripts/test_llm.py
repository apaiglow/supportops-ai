from supportops.llm import generate_response

def main() -> None:
    prompt = (
        'A customer asks whether their refund has been processed.'
        'You cannot access account records. D onot invent a refund '
        'status. Write a concise, professional response that explains '
        'the status needs to be verified using their order details.'
              )
    response = generate_response(prompt)
    print('=' * 60)
    print('SupportOps AI - Local LLM Smoke Test')
    print('=' * 60)
    print(response)
    print('=' * 60)

if __name__ == '__main__':
    main()