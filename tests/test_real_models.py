"""Opt-in checks against the real ONNX models.

Skipped unless RUN_MODEL_TESTS=1, because the first run downloads roughly 1GB.
These exist to verify the claims the retrieval design rests on: that the pinned
fastembed actually serves the chosen models, that the code embedder handles input
far longer than the 512-token window the spec's model had, and that the hybrid
sparse branch finds an exact identifier that dense similarity alone ranks poorly.

Run with:  RUN_MODEL_TESTS=1 uv run pytest tests/test_real_models.py -v
"""

from __future__ import annotations

import os

import pytest
from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.vector.embedder import FastEmbedder
from src.vector.qdrant import QdrantStore
from src.vector.search import SearchService

from .test_retrieval import chunk, index

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to exercise the real ONNX models (~1GB download)",
)


@pytest.fixture
def real_settings(tmp_path) -> Settings:
    return Settings(
        postgres_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        fastcache_dir=tmp_path.parent / "model-cache",
        repos_root=tmp_path / "repos",
        codebase_collection="real_codebase",
        docs_collection="real_docs",
        eager_model_load=True,
    )


async def test_dense_model_loads_with_the_expected_dimensions(real_settings: Settings):
    embedder = FastEmbedder(real_settings)
    dense, sparse = await embedder.embed_documents(["def validate_token(token): return True"])
    assert len(dense[0]) == real_settings.dense_dim == 768
    assert sparse[0].indices
    await embedder.aclose()


async def test_code_model_handles_input_far_beyond_512_tokens(real_settings: Settings):
    """The spec's bge-small would have silently truncated this at 512 tokens."""
    embedder = FastEmbedder(real_settings)
    big = "class Big:\n" + "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(300))
    assert len(big) > 512 * 4
    head, _ = await embedder.embed_documents([big])
    # The tail must influence the vector; if it were truncated, appending a
    # distinctive tail would leave the embedding unchanged.
    tailed, _ = await embedder.embed_documents([big + "\n    def sentinel_marker(self): pass\n"])
    assert head[0] != tailed[0]
    await embedder.aclose()


async def test_reranker_scores_the_relevant_document_higher(real_settings: Settings):
    embedder = FastEmbedder(real_settings)
    scores = await embedder.rerank(
        "how do I validate a bearer token",
        [
            "def resize_image(path, width):\n    return None",
            "def validate_token(token):\n    '''Check a bearer token.'''\n    return True",
        ],
    )
    assert scores[1] > scores[0]
    await embedder.aclose()


async def test_hybrid_finds_an_exact_identifier(real_settings: Settings):
    """The motivating case for adding a sparse branch to the spec's design."""
    embedder = FastEmbedder(real_settings)
    store = QdrantStore(real_settings, client=AsyncQdrantClient(location=":memory:"))
    await store.ensure_collections()
    search = SearchService(real_settings, store, embedder)

    await index(
        store,
        embedder,
        "repo",
        "a.py",
        [
            chunk("resize_image", "def resize_image(path, width):\n    return None"),
            chunk("send_email", "def send_email(to, body):\n    return None"),
            chunk("xyzzy_token_check", "def xyzzy_token_check(t):\n    return len(t) > 3"),
            chunk("parse_config", "def parse_config(path):\n    return {}"),
        ],
    )
    results = await search.search_codebase("xyzzy_token_check", limit=3)
    assert results
    assert results[0].identifier == "xyzzy_token_check"

    await store.aclose()
    await embedder.aclose()


async def test_length_sorted_batching_returns_vectors_in_input_order(real_settings: Settings):
    """Texts are embedded shortest-first to keep batches uniform; each vector must
    still come back aligned with its own text."""
    embedder = FastEmbedder(real_settings)
    texts = ["x = 1\n" * 200, "def short(): pass", "class Mid:\n    value = 2\n" * 10]
    together, together_sparse = await embedder.embed_documents(texts)
    for i, text in enumerate(texts):
        (alone,), (alone_sparse,) = await embedder.embed_documents([text])
        assert together[i] == pytest.approx(alone, abs=1e-4)
        assert together_sparse[i].indices == alone_sparse.indices
    await embedder.aclose()
