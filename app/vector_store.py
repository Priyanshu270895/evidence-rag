from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from app.config import settings
from app.store import ChunkStore


class VectorStoreUnavailableError(RuntimeError):
    pass


class VectorIndex(Protocol):
    def upsert(self, rows: list[dict]) -> None:
        pass

    def delete_document(self, document_id: str) -> None:
        pass

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[str]:
        pass

    def health_check(self) -> bool:
        pass


@dataclass
class SQLiteVectorIndex:
    store: ChunkStore

    def upsert(self, rows: list[dict]) -> None:
        return None

    def delete_document(self, document_id: str) -> None:
        return None

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[str]:
        return self.store.vector_search(query_embedding, limit, document_ids=document_ids)

    def health_check(self) -> bool:
        return True


class QdrantVectorIndex:
    def __init__(self, url: str, collection_name: str):
        try:
            from qdrant_client import QdrantClient, models
            from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
        except ImportError as exc:
            raise VectorStoreUnavailableError(
                "Qdrant vector backend requires `pip install -e .[qdrant]`."
            ) from exc

        self.client = QdrantClient(**qdrant_client_kwargs(url))
        self.models = models
        self.collection_name = collection_name
        self.qdrant_errors = (ResponseHandlingException, UnexpectedResponse)

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            self._ensure_collection(len(rows[0]["embedding"]))
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    self.models.PointStruct(
                        id=_qdrant_point_id(row["id"]),
                        vector=row["embedding"],
                        payload={
                            "chunk_id": row["id"],
                            "document_id": row["document_id"],
                            "document": row["document"],
                            "page": row["page"],
                            "chunk_index": row.get("chunk_index", 0),
                        },
                    )
                    for row in rows
                ],
                wait=True,
            )
        except self.qdrant_errors as exc:
            raise VectorStoreUnavailableError("Qdrant vector store is unavailable.") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            if not self.client.collection_exists(self.collection_name):
                return
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.FilterSelector(
                    filter=self.models.Filter(
                        must=[
                            self.models.FieldCondition(
                                key="document_id",
                                match=self.models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except self.qdrant_errors as exc:
            raise VectorStoreUnavailableError("Qdrant vector store is unavailable.") from exc

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[str]:
        try:
            self._ensure_collection(len(query_embedding))
            query_filter = None
            if document_ids:
                query_filter = self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key="document_id",
                            match=self.models.MatchAny(any=document_ids),
                        )
                    ]
                )
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
            )
            return [
                str(point.payload["chunk_id"])
                for point in result.points
                if isinstance(point.payload, dict) and point.payload.get("chunk_id")
            ]
        except self.qdrant_errors as exc:
            raise VectorStoreUnavailableError("Qdrant vector store is unavailable.") from exc

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except self.qdrant_errors:
            return False

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            if self.client.collection_exists(self.collection_name):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=vector_size,
                    distance=self.models.Distance.COSINE,
                ),
            )
        except self.qdrant_errors:
            raise VectorStoreUnavailableError("Qdrant vector store is unavailable.") from None


def build_vector_index(store: ChunkStore) -> VectorIndex:
    if settings.vector_backend == "sqlite":
        return SQLiteVectorIndex(store)
    if settings.vector_backend == "qdrant":
        return QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection)
    raise VectorStoreUnavailableError(f"Unsupported vector backend: {settings.vector_backend}")


def qdrant_client_kwargs(url: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"url": url, "timeout": settings.qdrant_timeout_seconds}
    if settings.qdrant_api_key is not None and settings.qdrant_api_key.get_secret_value():
        kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()
    return kwargs


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"evidence-rag:{chunk_id}"))
