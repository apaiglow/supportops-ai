# Local LLM Integration — SupportOps AI

## 1. Objective

Integrate a small local instruction-tuned language model into SupportOps AI through a reusable Python interface. The objective is to test whether a local model can generate professional customer-support drafts while keeping inference costs low and retaining control over the application's behavior.

The model is experimental. Generated responses are drafts, not verified facts or completed customer actions.

## 2. Runtime and model selection

- **Runtime:** Ollama
- **Runtime version:** 0.40.8
- **Selected model:** Qwen 2.5 1.5B (`qwen2.5:1.5b`)
- **Alternative tested:** Qwen 2.5 0.5B (`qwen2.5:0.5b`)
- **API endpoint:** `http://127.0.0.1:11434`
- **Execution environment:** Linux under WSL2, CPU inference
- **Model storage sizes reported by Ollama:** 986 MB for 1.5B and 397 MB for 0.5B

### Installation findings

The initial Ollama installation attempts failed during large archive downloads. The executable was present, but the required `llama-server` runtime binary was missing. Rerunning the installer completed the installation, after which the server and model inference worked.

The Ollama server must be running before Python can call its API. A port-binding error when starting `ollama serve` indicated that a server was already listening on the default port; it did not necessarily indicate a failure.

## 3. Python integration

### `src/supportops/llm.py`

Provides `generate_response()`, a reusable interface to Ollama's `/api/generate` endpoint.

Responsibilities include:

- Selecting a model through a function argument or `SUPPORTOPS_LLM_MODEL`.
- Selecting the API endpoint through `OLLAMA_BASE_URL`.
- Sending a non-streaming JSON request.
- Configuring temperature, output-token limit, and timeout.
- Validating input parameters.
- Converting connection, HTTP, timeout, and response errors into actionable exceptions.
- Returning non-empty generated text.

Default settings are Qwen 2.5 1.5B, temperature 0.2, a 60-second timeout, and a generation limit of 180 tokens.

The interface does not establish whether the generated answer is true. Factual validation remains a separate responsibility.

### `src/supportops/prompts.py`

Provides `build_support_prompt()` to create a consistent customer-support prompt.

The prompt instructs the model to:

- Use only the supplied facts.
- Avoid inventing statuses, policies, or completed actions.
- Acknowledge missing information.
- Offer a useful next step instead of refusing unnecessarily.
- Produce concise, professional customer-facing text.

The prompt is guidance, not a guarantee of factual correctness.

### Test scripts

- `scripts/test_llm.py`: smoke-tests the Python-to-Ollama generation path.
- `scripts/test_support_prompts.py`: tests five support scenarios.
- `scripts/evaluate_llm.py`: records response latency and saves a CSV for manual quality evaluation.

## 4. Initial model comparison

The 0.5B model sometimes refused simple support requests and, in one test, invented a non-refundable order status that had not been supplied.

The 1.5B model produced a more appropriate response to the same constrained refund prompt. However, this comparison involved only a small number of examples and does not establish that the 1.5B model is consistently more reliable.

**Decision:** Use Qwen 2.5 1.5B as the provisional experimental generator. Do not treat its output as production-safe without further evaluation and safeguards.

## 5. Latency measurements

Five sequential generation requests produced the following results:

| Test case | Latency (seconds) |
|---|---:|
| Refund status | 22.47 |
| Order tracking | 5.34 |
| Order cancellation | 7.61 |
| Known order facts | 3.60 |
| Unsupported refund claim | 8.85 |
| **Total** | **47.87** |
| **Mean** | **9.57** |
| **Median** | **7.61** |

The first request was substantially slower than the remaining requests. Model loading or initial runtime overhead is a possible explanation, but this was not independently verified. Prompt length and generated-token count also differ between cases.

These results are preliminary measurements from five requests, not a reliable estimate of long-run latency or tail latency.

## 6. Quality evaluation

Five scenarios were designed:

1. Asking about refund status without account records.
2. Asking about order tracking without tracking information.
3. Requesting order cancellation with unknown order and shipment details.
4. Asking about an order when a shipped status and tracking number are supplied as test facts.
5. Asking the model to confirm a refund without evidence of a refund decision.

The model responses appeared well-formatted and followed the base prompt template in the developer's review. However, the completed per-case quality scores have not been recorded in this document.

Formatting and instruction following alone do not establish factual reliability. A response can sound professional and still introduce unsupported claims.

## 7. Known limitations and risks

- A small language model can ignore instructions or fabricate information.
- Prompt instructions alone are not a safety mechanism.
- The five latency measurements are too few to establish stable performance.
- Peak memory use has not been isolated and measured.
- Quality scores and detailed failure analysis remain incomplete.
- Historical retrieval examples must not be treated as evidence of a current customer's account status.
- The generator must not claim that an account was checked, a refund was approved, or an order was cancelled unless a trusted system explicitly confirms that action.

## 8. Next steps

1. Complete manual scoring of the generated responses.
2. Measure latency again with the model already loaded, and record memory observations.
3. Connect generation to the retrieval component using explicit evidence boundaries.
4. Add deterministic validation and human-review routing for uncertain or unsupported answers.
5. Evaluate end-to-end quality rather than relying only on fluency or formatting.

## 9. Conclusion

A local LLM generation path now works through a reusable Python interface. Qwen 2.5 1.5B is the provisional model for further experiments because its initial response was more useful than the 0.5B model's response in a constrained test.

The integration demonstrates technical feasibility, not production readiness. The next phase should focus on grounding, evaluation, and safe failure handling.