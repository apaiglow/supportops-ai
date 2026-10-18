
# SupportOps AI — Project Specification

## 1. Problem Statement

Support teams receive customer messages that must be categorized, routed, investigated, and answered. SupportOps AI aims to assist support agents by combining text classification, knowledge retrieval, and evidence-grounded response drafting.

The system provides recommendations and drafts. It does not autonomously execute account changes, refunds, or other consequential customer actions.

## 2. Primary Objectives

- Classify incoming support messages by category and intent.
- Retrieve relevant historical examples or knowledge.
- Draft responses grounded in retrieved evidence.
- Detect uncertainty and flag cases for human review.
- Expose the pipeline through a documented API and usable interface.
- Evaluate individual components and end-to-end behavior.
- Build a reproducible, testable, modular software system.

## 3. Proposed Components

1. Input validation and deterministic preprocessing.
2. Text classification using a measurable baseline.
3. Knowledge retrieval using lexical and/or semantic search.
4. Local LLM response generation, subject to hardware feasibility.
5. Response validation and human-review flags.
6. FastAPI backend.
7. User interface.
8. Testing, logging, evaluation, and documentation.

Component choices remain provisional until implementation and evaluation provide sufficient evidence.

## 4. Engineering Principles

- Establish a baseline before introducing complexity.
- Prefer deterministic code when machine learning is unnecessary.
- Compare alternatives using consistent evaluation data.
- Prevent data leakage between training and evaluation.
- Investigate errors and implement justified improvements.
- Measure resource usage and latency on the target machine.
- Never claim that a generated response is correct merely because
  it is fluent or resembles a retrieved example.
- Document limitations and report actual measured results.

## 5. Constraints

- Development environment: VS Code and WSL.
- Local hardware: approximately 8 GB RAM, CPU-only.
- External API spending: avoid unless explicitly reconsidered.
- Raw datasets, secrets, and generated model artifacts must not be committed to the public repository.

## 6. Initial Success Criteria

- A reproducible classification training and evaluation pipeline.
- A retrieval system evaluated on a defined query set.
- A working response-drafting pipeline with evidence-aware safeguards.
- A FastAPI backend and usable interface.
- Automated tests for important components and failure cases.
- An end-to-end evaluation report with actual results.
- Clear setup instructions, architecture documentation, and documented limitations.

Exact metric targets will be established after the dataset, baselines, and evaluation requirements are understood.

## 7. Known Risks

- Synthetic training examples may not represent real customer traffic.
- Repeated or templated instructions may cause data leakage.
- Similarity scores do not guarantee relevant retrieval.
- Local LLM latency and memory requirements are uncertain.
- Generated responses may contain unsupported claims.
- Classification confidence may be poorly calibrated.

## 8. Current Status

Day 1: Initial requirements defined and candidate dataset audited.

Next: Investigate duplicate instructions and label distributions,
then prepare a reproducible data-splitting strategy.
