"""Qdrant collection management and hybrid retrieval primitives."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from src.config import Settings
from src.docs.sections import DocChunk
from src.parser.base import CodeChunk
from src.vector.embedder import SparseVec

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Fixed namespace so point ids are reproducible across processes and restarts.
POINT_NAMESPACE = uuid.UUID("8f1b6f52-0c2e-5a4f-9b7a-2f0d9c3b41ae")

# Without these, every filtered search degrades to a full scan. The spec omitted
# them entirely.
CODEBASE_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "repo_name": models.PayloadSchemaType.KEYWORD,
    "file_path": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "node_type": models.PayloadSchemaType.KEYWORD,
    "identifier": models.PayloadSchemaType.KEYWORD,
}

DOCS_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "framework": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.KEYWORD,
    # Re-ingesting a source deletes its points by this, so it must not scan.
    "source_url": models.PayloadSchemaType.KEYWORD,
    "doc_title": models.PayloadSchemaType.KEYWORD,
}


def point_id(repo_name: str, file_path: str, node_path: str, chunk_index: int) -> str:
    """Deterministic id, so re-syncing a file upserts instead of duplicating it.

    The spec left ids unspecified, which would have meant a growing pile of stale
    duplicates on every re-index.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, f"{repo_name}:{file_path}:{node_path}:{chunk_index}"))


async def with_retry(operation, what: str, attempts: int = 3):
    """Retry a Qdrant write through a transient failure.

    A long ingest issues thousands of writes; one timed-out upsert should cost a
    few seconds, not the whole job. Timeouts and connection errors are retried,
    and nothing else — a malformed request would only fail again.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except (httpx.TimeoutException, httpx.NetworkError, ResponseHandlingException) as exc:
            if attempt == attempts:
                raise
            delay = 2.0 * attempt
            logger.warning(
                "%s failed (%s: %s); retrying in %.0fs (attempt %d/%d)",
                what,
                type(exc).__name__,
                exc or "no detail",
                delay,
                attempt,
                attempts,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def doc_point_id(source_url: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"doc:{source_url}:{chunk_index}"))


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout,
        )

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._settings.codebase_collection

    # --- lifecycle -----------------------------------------------------------

    async def ensure_collections(self) -> None:
        await self._ensure(self._settings.codebase_collection, CODEBASE_PAYLOAD_INDEXES)
        await self._ensure(self._settings.docs_collection, DOCS_PAYLOAD_INDEXES)

    async def _ensure(self, name: str, indexes: dict[str, models.PayloadSchemaType]) -> None:
        if not await self._client.collection_exists(name):
            logger.info("creating collection %s", name)
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=self._settings.dense_dim, distance=models.Distance.COSINE
                    )
                },
                # IDF is required for BM25 scoring to behave correctly server-side.
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                on_disk_payload=True,
            )
        for field_name, schema in indexes.items():
            try:
                await self._client.create_payload_index(
                    collection_name=name, field_name=field_name, field_schema=schema
                )
            except Exception as exc:  # pragma: no cover - local mode has no indexes
                logger.debug("payload index %s on %s skipped: %s", field_name, name, exc)

    async def healthy(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.close()

    # --- writes --------------------------------------------------------------

    async def upsert_chunks(
        self,
        *,
        repo_name: str,
        file_path: str,
        commit_sha: str | None,
        dirty: bool,
        chunks: list[CodeChunk],
        dense: list[list[float]],
        sparse: list[SparseVec],
    ) -> int:
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=point_id(repo_name, file_path, chunk.node_path, chunk.chunk_index),
                vector={
                    DENSE_VECTOR: dense[i],
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse[i].indices, values=sparse[i].values
                    ),
                },
                payload={
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "commit_sha": commit_sha,
                    "dirty": dirty,
                    "language": chunk.language,
                    "node_type": chunk.node_type,
                    "identifier": chunk.identifier,
                    "parent_identifier": chunk.parent_identifier,
                    "node_path": chunk.node_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_index": chunk.chunk_index,
                    "chunk_total": chunk.chunk_total,
                    "content_hash": chunk.content_hash,
                    "is_truncated": chunk.is_truncated,
                    "content": chunk.content,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        await with_retry(
            lambda: self._client.upsert(
                collection_name=self._settings.codebase_collection, points=points, wait=True
            ),
            f"upsert {len(points)} code chunks for {file_path}",
        )
        return len(points)

    async def delete_file(self, repo_name: str, file_path: str) -> None:
        """Drop every point for one file.

        Called before re-upserting a changed file so that a file which now yields
        fewer chunks does not leave orphans behind, and when a file disappears
        from the working tree. The spec had no deletion path at all.
        """
        await self._client.delete(
            collection_name=self._settings.codebase_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_name", match=models.MatchValue(value=repo_name)
                        ),
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=file_path)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    async def upsert_doc_chunks(
        self,
        *,
        source_url: str,
        source_type: str,
        doc_title: str,
        framework: str | None,
        version: str | None,
        link_base: str | None,
        chunks: list[DocChunk],
        dense: list[list[float]],
        sparse: list[SparseVec],
    ) -> int:
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=doc_point_id(source_url, chunk.chunk_index),
                vector={
                    DENSE_VECTOR: dense[i],
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse[i].indices, values=sparse[i].values
                    ),
                },
                payload={
                    "doc_title": doc_title,
                    "source_type": source_type,
                    "source_url": source_url,
                    "framework": framework,
                    "version": version,
                    "heading_hierarchy": chunk.heading_path,
                    "location": chunk.location,
                    "url": (link_base + chunk.location) if link_base else None,
                    "chunk_index": chunk.chunk_index,
                    "part": chunk.part,
                    "parts": chunk.parts,
                    "content": chunk.content,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        await with_retry(
            lambda: self._client.upsert(
                collection_name=self._settings.docs_collection, points=points, wait=True
            ),
            f"upsert {len(points)} doc chunks for {doc_title}",
        )
        return len(points)

    async def delete_doc_source(self, source_url: str) -> None:
        await self._client.delete(
            collection_name=self._settings.docs_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url", match=models.MatchValue(value=source_url)
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def count_docs(self, source_url: str | None = None) -> int:
        flt = None
        if source_url:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_url", match=models.MatchValue(value=source_url)
                    )
                ]
            )
        result = await self._client.count(
            collection_name=self._settings.docs_collection, count_filter=flt, exact=True
        )
        return result.count

    async def delete_repo(self, repo_name: str) -> None:
        await self._client.delete(
            collection_name=self._settings.codebase_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_name", match=models.MatchValue(value=repo_name)
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def count(self, repo_name: str | None = None) -> int:
        flt = None
        if repo_name:
            flt = models.Filter(
                must=[
                    models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))
                ]
            )
        result = await self._client.count(
            collection_name=self._settings.codebase_collection, count_filter=flt, exact=True
        )
        return result.count

    # --- reads ---------------------------------------------------------------

    async def hybrid_search(
        self,
        *,
        dense_query: list[float],
        sparse_query: SparseVec,
        limit: int,
        query_filter: models.Filter | None = None,
        collection: str | None = None,
    ) -> list[Hit]:
        """Dense kNN and BM25 sparse, fused with Reciprocal Rank Fusion in Qdrant.

        Fusion happens server-side in a single request, so hybrid costs one round
        trip rather than two plus a client-side merge.
        """
        prefetch_limit = max(limit * self._settings.prefetch_multiplier, limit)
        response = await self._client.query_points(
            collection_name=collection or self._settings.codebase_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_query, using=DENSE_VECTOR, limit=prefetch_limit, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices, values=sparse_query.values
                    ),
                    using=SPARSE_VECTOR,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            Hit(id=str(p.id), score=float(p.score), payload=p.payload or {})
            for p in response.points
        ]
