# EvidenceRAG Implementation Plan

EvidenceRAG is a local-first, domain-neutral RAG system for PDFs with page-level citations.
The goal is to demonstrate production engineering judgment while staying practical on a
Windows laptop with 8 GB RAM and a GTX 1650.

## Current Phase: Reranking And Safety Hardening

- Use Python 3.12 in a local virtual environment.
- Keep the pipeline transparent before adding orchestration frameworks.
- Validate formatting, linting, tests, app startup, ingestion, retrieval, and generation locally.
- Avoid committing virtual environments, databases, uploaded PDFs, model weights, caches, or secrets.
- Use evaluation metrics before accepting reranking, query rewriting, or chunking changes.

## Review Findings

- Packaging needed explicit package discovery so `data/` is not treated as a Python package.
- Uploads need stronger production guards: size limits, duplicate handling, stable document IDs,
  and lifecycle operations.
- The SQLite FTS index needs delete/update maintenance before document replacement or deletion.
- Brute-force vector search is acceptable for Phase 1 but should be swapped for Qdrant only after
  the local version is correct and evaluated.
- Generation should stay grounded in retrieved evidence and refuse unsupported answers.
- Prompt-injection defenses should be added before ingesting untrusted documents in demos.

## Phase 1: Reliable Local RAG

- Complete: document IDs, metadata, file hashes, and duplicate detection.
- Complete: document listing, detail, deletion, re-indexing, and chunk inspection endpoints.
- Complete: FTS maintenance for insert, update, and delete paths.
- Complete: configurable chunk size and overlap.
- Complete: health checks for SQLite and Ollama.
- Complete: expanded unit and API tests.

## Phase 2: Retrieval Quality

- Complete: small JSONL evaluation dataset with questions and relevant evidence labels.
- Complete: Recall@K, Precision@K, MRR, and nDCG metric implementation.
- Complete: command-line evaluation report for local retrieval experiments.
- Next: add cross-encoder reranking with a small model suitable for CPU or 4 GB VRAM.
- Next: add query rewriting only when evaluation shows it helps.

## Phase 3: Grounding, Observability, And UX

- Add answer faithfulness checks and citation correctness checks.
- Add structured logs, request IDs, latency measurements, and retry policies.
- Add prompt-injection filtering and evidence-only answer constraints.
- Build a Streamlit UI for upload, search, answer display, and citation inspection.
- Add GitHub Actions CI.

## Phase 4: Later Deployment Track

- Add Docker Compose only after local validation.
- Move vector storage to Qdrant when the SQLite baseline is measured.
- Design and implement Azure deployment once local quality and resource needs are known.
