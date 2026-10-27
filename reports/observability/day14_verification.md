# Day 14 — Deployment and Operational Verification

## Objective
Verify the Docker deployment, API availability, application logging configuration,
and end-to-end generation behavior without adding new features.

## Deployment verification
- Docker Compose configuration and image rebuild succeeded.
- Container: `supportops-ai-api-1`.
- Final observed container health: `healthy`.
- Application startup completed successfully.
- Semantic retrieval index loaded successfully; embedding initialization took
  approximately 55 seconds in this startup.
- `GET /health`: HTTP 200, response `{"status":"ok","retriever":"ready"}`.
- `GET /docs`: HTTP 200.
- The Hugging Face unauthenticated-request warning was non-fatal.

## Automated tests
- Full test suite: 23 passed.
- One Starlette deprecation warning from the installed TestClient/httpx
  integration; no test failures.
- `git diff --check` passed before the Docker rebuild.

## Generation smoke test
A successful generation request was completed before the logger configuration
change and subsequent image rebuild.

- Input: customer asks how to cancel an order.
- Available facts explicitly stated that account and order records were unavailable.
- `top_k`: 3.
- HTTP status: 200.
- Reported API latency: approximately 55.941 seconds.
- Total curl time: approximately 57.946 seconds.
- Retrieved examples were relevant cancellation examples, with similarity scores
  around 0.86.

### Observed response-quality limitation
The generated draft included unsupported placeholders such as
`{{Online Company Portal Info}}` and `{{Online Order Interaction}}`, and its
final sentence was incomplete. This demonstrates that relevant retrieval and
successful HTTP status do not guarantee a complete, grounded response.
The model output needs further quality and grounding evaluation.

## Logging verification and limitation
- The application logger is explicitly configured at `INFO` level.
- The rebuilt container reports an effective logger level of `INFO`.
- Container logs show startup and HTTP access logs.
- A generation request has not been repeated after the logger change, so the
  successful-generation timing log has not yet been verified after the fix.
- Do not treat the existence of timing instrumentation or the logger level alone
  as proof that a timing entry was emitted for a successful generation.

## Operational conclusion
The rebuilt API is healthy, the health and documentation endpoints respond
successfully, and all 23 automated tests pass. The previous end-to-end request
also succeeded, but was slow and exposed response-quality issues. Further work
should focus on grounding/completeness evaluation and reliable timing-log
verification rather than adding features prematurely.
