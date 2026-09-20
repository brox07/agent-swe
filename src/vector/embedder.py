"""FastEmbed/ONNX wrapper: dense code embeddings, BM25 sparse, cross-encoder rerank.

ONNX inference is synchronous and CPU-bound. Every call here is dispatched to a
thread pool, because running it on the event loop would stall all concurrent
requests including the MCP session keepalive.

The dense model is a code-trained model with an 8k context window rather than the
spec's ``bge-small-en-v1.5``, whose 512-token window silently truncated any class
larger than roughly 130 lines.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Protocol, runtime_checkable

from src.config import Settings

logger = logging.getLogger(__name__)

# Module scope, so it bounds every job sharing this process.
_bulk_embed = asyncio.Semaphore(1)


def plan_batches(texts: list[str], max_count: int, max_chars: int) -> list[list[str]]:
    """Group texts so no batch exceeds a count or a total size.

    The runtime pads a batch to its longest member and holds a workspace sized
    for it, so batching purely by count lets one long chunk multiply the cost of
    everything beside it. A text over the budget embeds on its own.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and (len(current) >= max_count or size + len(text) > max_chars):
            groups.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        groups.append(current)
    return groups


class SparseVec(NamedTuple):
    indices: list[int]
    values: list[float]


@runtime_checkable
class Embedder(Protocol):
    """Interface the retrieval layer depends on, so tests can substitute a stub."""

    dense_dim: int

    async def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVec]]: ...

    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]: ...

    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...

    async def aclose(self) -> None: ...


class FastEmbedder:
    """Concrete embedder backed by fastembed's ONNX runtime models."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.dense_dim = settings.dense_dim
        self._cache_dir = str(settings.fastcache_dir)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.inference_workers, thread_name_prefix="onnx"
        )
        self._dense = None
        self._sparse = None
        self._reranker = None
        self._dense_lock = asyncio.Lock()
        self._sparse_lock = asyncio.Lock()
        self._reranker_lock = asyncio.Lock()

    # --- model access --------------------------------------------------------

    async def _get_dense(self):
        if self._dense is None:
            async with self._dense_lock:
                if self._dense is None:
                    from fastembed import TextEmbedding

                    logger.info("loading dense model %s", self._settings.dense_model)
                    self._dense = await self._run(
                        TextEmbedding,
                        model_name=self._settings.dense_model,
                        cache_dir=self._cache_dir,
                    )
        return self._dense

    async def _get_sparse(self):
        if self._sparse is None:
            async with self._sparse_lock:
                if self._sparse is None:
                    from fastembed import SparseTextEmbedding

                    logger.info("loading sparse model %s", self._settings.sparse_model)
                    self._sparse = await self._run(
                        SparseTextEmbedding,
                        model_name=self._settings.sparse_model,
                        cache_dir=self._cache_dir,
                    )
        return self._sparse

    async def _get_reranker(self):
        if self._reranker is None:
            async with self._reranker_lock:
                if self._reranker is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    logger.info("loading reranker %s", self._settings.reranker_model)
                    self._reranker = await self._run(
                        TextCrossEncoder,
                        model_name=self._settings.reranker_model,
                        cache_dir=self._cache_dir,
                    )
        return self._reranker

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    # --- public API ----------------------------------------------------------

    async def warmup(self) -> None:
        """Load models and run one real inference before /health reports ready.

        First boot downloads roughly 1GB into the model cache volume, so a cold
        start is slow exactly once.
        """
        await self.embed_documents(["def warmup():\n    return True\n"])
        await self.embed_query("warmup")
        if self._settings.eager_model_load:
            await self.rerank("warmup", ["def warmup(): ..."])

    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        if not texts:
            return [], []
        # One bulk embed at a time across every job. Parallel jobs cannot go
        # faster — they share one executor — but each holds its own batch and
        # runtime arenas, and that is what grew the engine to 21GB resident and
        # had the host kernel kill it mid-ingest. Queries use embed_query and
        # are never held behind this.
        async with _bulk_embed:
            return await self._embed_documents(texts)

    async def _embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVec]]:
        dense_model = await self._get_dense()
        sparse_model = await self._get_sparse()
        batch = self._settings.embed_batch_size
        budget = self._settings.embed_batch_chars

        # A batch is padded to its longest text, so one long listing among short
        # chunks makes the whole batch pay for its length. Embedding in length
        # order keeps batches uniform: 1.5x faster measured on a real book.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = [texts[i] for i in order]

        def _work() -> tuple[list[list[float]], list[SparseVec]]:
            dense_sorted: list[list[float]] = []
            sparse_sorted: list[SparseVec] = []
            for group in plan_batches(ordered, batch, budget):
                dense_sorted.extend(
                    v.tolist() for v in dense_model.embed(group, batch_size=len(group))
                )
                sparse_sorted.extend(
                    SparseVec(indices=s.indices.tolist(), values=s.values.tolist())
                    for s in sparse_model.embed(group, batch_size=len(group))
                )
            dense: list[list[float]] = [[] for _ in texts]
            sparse: list[SparseVec] = [SparseVec([], []) for _ in texts]
            for position, original in enumerate(order):
                dense[original] = dense_sorted[position]
                sparse[original] = sparse_sorted[position]
            return dense, sparse

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _work)

    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        """Embed a query.

        ``query_embed`` matters for BM25: a query is a bag of terms with no term
        frequency weighting, which is not what ``embed`` produces.
        """
        dense_model = await self._get_dense()
        sparse_model = await self._get_sparse()

        def _work() -> tuple[list[float], SparseVec]:
            dense = next(iter(dense_model.query_embed(text))).tolist()
            s = next(iter(sparse_model.query_embed(text)))
            return dense, SparseVec(indices=s.indices.tolist(), values=s.values.tolist())

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _work)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Cross-encoder scores, higher is better.

        The reranker's own window is 512 tokens, far smaller than the dense
        model's, so callers pass the display-capped text rather than full chunks.
        """
        if not documents:
            return []
        reranker = await self._get_reranker()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: list(reranker.rerank(query, documents))
        )

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False)
