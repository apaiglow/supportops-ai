# Day 13 — Observability and Latency Findings

## Objectives
Add request-level logging and measure retrieval, generation, and total
pipeline duration for SupportOps AI.

## Health Endpoint Baseline
Five requests to GET /health returned HTTP 200.

| Metric | Result |
|---|---:|
| Successful checks | 5/5 |
| Minimum latency | 3.2 ms |
| Median latency | 5.2 ms |
| Maximum latency | 41.9 ms |
| Mean latency | 12.4 ms |

These measurements cover the health endpoint only, not the RAG pipeline.

## Instrumentation
- RAG retrieval duration is measured separately.
- LLM generation duration is measured separately.
- Total RAG pipeline duration is measured.
- API request duration and generation failures are logged.
- Customer message content is not intentionally included in the new timing logs.

## Verification
- Targeted API and RAG tests: 23 passed.
- Full test-suite result: recorded in the Day 13 terminal verification.
- A Starlette/httpx deprecation warning was observed during targeted tests.
  It did not cause a test failure.

## Limitations
The health endpoint is not representative of model-generation latency.
A previous model request was slow during model loading and another API
request timed out. These observations indicate that cold starts and
CPU-only inference can dominate response time.

Successful retrieval/generation stage timings have not yet been recorded
from a confirmed run using the updated application image. The running
Docker image may predate the latest source changes; rebuild it before
using container logs to validate the new instrumentation.

## Conclusion
Timing instrumentation and failure logging are implemented in source.
Health-check responsiveness is good, but end-to-end generation performance
still needs a successful, instrumented run before it can be quantified.
