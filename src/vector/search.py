"""Retrieval orchestration: hybrid fusion, duplicate collapsing, rerank, budget."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from qdrant_client import models

from src.config import Settings
from src.vector.embedder import Embedder
from src.vector.qdrant import Hit, QdrantStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    repo_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    node_type: str
    identifier: str
    node_path: str
    score: float
    content: str
    content_truncated: bool
    part: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DocResult:
    doc_title: str
    section: str
    location: str
    url: str | None
    framework: str | None
    version: str | None
    score: float
    content: str
    content_truncated: bool
    part: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _is_ancestor(outer: str, inner: str) -> bool:
    """True if `outer` is an enclosing scope of `inner` (AuthService vs AuthService.login)."""
    return inner.startswith(outer + ".")


def collapse_overlaps(hits: list[Hit]) -> list[Hit]:
    """Drop a hit that is wholly contained by a better-scoring hit from the same file.

    Nested chunking indexes a class and each of its methods, so a single query can
    match both. Returning both wastes the caller's context on duplicated code, so
    the stronger match wins and the weaker relative is dropped.
    """
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    kept: list[Hit] = []
    for hit in ordered:
        path = hit.payload.get("node_path", "")
        file_path = hit.payload.get("file_path", "")
        redundant = False
        for winner in kept:
            if winner.payload.get("file_path") != file_path:
                continue
            wpath = winner.payload.get("node_path", "")
            if wpath == path or _is_ancestor(wpath, path) or _is_ancestor(path, wpath):
                redundant = True
                break
        if not redundant:
            kept.append(hit)
    return kept


class SearchService:
    def __init__(self, settings: Settings, store: QdrantStore, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder

    async def search_codebase(
        self,
        query: str,
        *,
        language: str | None = None,
        repo_name: str | None = None,
        node_type: str | None = None,
        rerank: bool = False,
        limit: int | None = None,
    ) -> list[SearchResult]:
        limit = limit or self._settings.default_limit
        query_filter = self._build_filter(
            language=language, repo_name=repo_name, node_type=node_type
        )

        # Pull a wider candidate set when reranking, since the cross-encoder is
        # what decides the final order.
        fetch = self._settings.rerank_candidates if rerank else limit * 2
        dense_query, sparse_query = await self._embedder.embed_query(query)
        hits = await self._store.hybrid_search(
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=fetch,
            query_filter=query_filter,
        )

        hits = collapse_overlaps(hits)

        if rerank and hits:
            # The reranker window (512 tokens) is far smaller than the embedding
            # window, so it scores the same capped text the caller will receive.
            texts = [self._capped(h.payload.get("content", ""))[0] for h in hits]
            scores = await self._embedder.rerank(query, texts)
            ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            hits = [hit for hit, _ in ranked]
            rescored = [score for _, score in ranked]
        else:
            rescored = [h.score for h in hits]

        results: list[SearchResult] = []
        for hit, score in list(zip(hits, rescored, strict=True))[:limit]:
            payload = hit.payload
            content, truncated = self._capped(payload.get("content", ""))
            total = payload.get("chunk_total", 1) or 1
            results.append(
                SearchResult(
                    repo_name=payload.get("repo_name", ""),
                    file_path=payload.get("file_path", ""),
                    start_line=int(payload.get("start_line", 0)),
                    end_line=int(payload.get("end_line", 0)),
                    language=payload.get("language", ""),
                    node_type=payload.get("node_type", ""),
                    identifier=payload.get("identifier", ""),
                    node_path=payload.get("node_path", ""),
                    score=round(float(score), 6),
                    content=content,
                    content_truncated=truncated,
                    part=(
                        f"{int(payload.get('chunk_index', 0)) + 1}/{total}" if total > 1 else None
                    ),
                )
            )
        return results

    async def search_docs(
        self,
        topic: str,
        *,
        framework: str | None = None,
        version: str | None = None,
        source_type: str | None = None,
        rerank: bool = False,
        limit: int | None = None,
    ) -> list[DocResult]:
        limit = limit or self._settings.docs_default_limit
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (
                ("framework", framework and framework.lower()),
                ("version", version),
                ("source_type", source_type),
            )
            if value
        ]
        query_filter = models.Filter(must=conditions) if conditions else None
        fetch = self._settings.rerank_candidates if rerank else limit
        dense_query, sparse_query = await self._embedder.embed_query(topic)
        hits = await self._store.hybrid_search(
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=fetch,
            query_filter=query_filter,
            collection=self._settings.docs_collection,
        )
        scores = [h.score for h in hits]
        if rerank and hits:
            texts = [self._capped(h.payload.get("content", ""))[0] for h in hits]
            reranked = await self._embedder.rerank(topic, texts)
            ranked = sorted(zip(hits, reranked, strict=True), key=lambda p: p[1], reverse=True)
            hits = [h for h, _ in ranked]
            scores = [sc for _, sc in ranked]

        results: list[DocResult] = []
        for hit, score in list(zip(hits, scores, strict=True))[:limit]:
            payload = hit.payload
            content, truncated = self._capped(payload.get("content", ""), noun="section")
            parts = int(payload.get("parts", 1) or 1)
            results.append(
                DocResult(
                    doc_title=payload.get("doc_title", ""),
                    section=" > ".join(payload.get("heading_hierarchy") or []),
                    location=payload.get("location", ""),
                    url=payload.get("url"),
                    framework=payload.get("framework"),
                    version=payload.get("version"),
                    score=round(float(score), 6),
                    content=content,
                    content_truncated=truncated,
                    part=f"{int(payload.get('part', 0)) + 1}/{parts}" if parts > 1 else None,
                )
            )
        return results

    def _capped(self, content: str, noun: str = "file") -> tuple[str, bool]:
        """Bound what one hit can cost the caller's context window.

        Exact line references accompany every result, so the agent can open the
        file when it needs the remainder.
        """
        cap = self._settings.max_result_chars
        if len(content) <= cap:
            return content, False
        return content[:cap].rstrip() + f"\n... [truncated, open the {noun} for the rest]", True

    @staticmethod
    def _build_filter(
        *, language: str | None, repo_name: str | None, node_type: str | None
    ) -> models.Filter | None:
        conditions = []
        for key, value in (
            ("language", language),
            ("repo_name", repo_name),
            ("node_type", node_type),
        ):
            if value:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions) if conditions else None
