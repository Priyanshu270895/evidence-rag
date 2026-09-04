# Azure Deployment Track

This is the Phase 4 deployment design for EvidenceRAG. It is implementation-ready, but actual
deployment requires an Azure subscription, Azure CLI login, and cost controls.

## Target Architecture

```text
GitHub Actions
  -> build and test
  -> build container image
  -> push image to Azure Container Registry
  -> deploy API container to Azure Container Apps
  -> connect API to persistent storage and vector storage
```

Recommended first Azure version:

- Azure Container Registry stores the EvidenceRAG API image.
- Azure Container Apps runs the FastAPI API container with HTTPS ingress.
- Azure Files stores uploaded PDFs and the SQLite fallback database if SQLite is used.
- Qdrant should run as a managed service or a separate container app only after local Docker
  validation.
- Ollama should be treated carefully in Azure. CPU-only hosted generation can be slow, and GPU
  hosting should be planned separately.

## Why Azure Container Apps

Azure Container Apps is a good fit for the API because it runs containerized applications without
managing servers and supports HTTP ingress and scaling. Microsoft documentation describes it as a
serverless platform for containerized applications.

Reference: https://learn.microsoft.com/en-ie/azure/container-apps/overview

## Why Azure Container Registry

Azure Container Registry stores private container images and supports normal Docker push/pull
workflows. It gives the deployment track a standard place to publish the API image before deploying
to Container Apps.

Reference: https://learn.microsoft.com/en-us/azure/container-registry/

## Persistent Storage

Container filesystems are not durable enough for uploaded PDFs or SQLite data. Azure Container Apps
supports Azure Files storage mounts for persisted files, so the API should mount a file share at the
same path used by `UPLOAD_DIR` and, if needed, `DATABASE_PATH`.

Reference: https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts

## Environment Variables

```text
DATABASE_PATH=/app/data/evidence_rag.db
UPLOAD_DIR=/app/data/uploads
OLLAMA_BASE_URL=<ollama endpoint>
OLLAMA_MODEL=qwen2.5:1.5b
VECTOR_BACKEND=sqlite or qdrant
QDRANT_URL=<qdrant endpoint>
QDRANT_COLLECTION=evidence_rag_chunks
LOG_LEVEL=INFO
```

## Deployment Steps

1. Run local validation with Python first.
2. Run Docker Compose locally after Docker Desktop is installed.
3. Build the API image.
4. Push the API image to Azure Container Registry.
5. Create an Azure Container Apps environment.
6. Mount Azure Files for persistent uploads and SQLite fallback storage.
7. Deploy the API container with environment variables.
8. Add health-check monitoring against `/health`.
9. Move vector search to Qdrant only after the SQLite baseline metrics are recorded.

## Production Checks Before Public Demo

- Add authentication or keep the app private.
- Pin container image versions or digests.
- Add resource limits and autoscaling rules.
- Store secrets in platform-managed secrets, not `.env`.
- Expand the evaluation dataset beyond the smoke PDF.
- Re-run retrieval metrics after every retrieval or chunking change.
- Validate prompt-injection behavior with adversarial PDFs.
