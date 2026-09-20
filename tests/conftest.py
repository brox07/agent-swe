"""Shared fixtures.

Retrieval is exercised against a real in-memory Qdrant, with a deterministic stub
embedder so assertions about ordering are exact rather than dependent on a
downloaded model. A separate opt-in test covers the real models.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.db.models import Base
from src.db.postgres import dispose_engine, init_engine
from src.ingest.git_sync import SyncService
from src.vector.embedder import SparseVec
from src.vector.qdrant import QdrantStore
from src.vector.search import SearchService

STUB_DIM = 64
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> list[str]:
    """Split identifiers the way a code searcher would: snake_case and camelCase."""
    out: list[str] = []
    for word in _TOKEN_RE.findall(text.lower()):
        out.append(word)
        out.extend(part for part in word.split("_") if part and part != word)
    return out


class StubEmbedder:
    """Hashing-trick embedder: deterministic, offline, and lexically meaningful.

    Dense vectors are normalised bag-of-token vectors, so cosine similarity
    behaves enough like the real thing for ordering assertions to mean something.
    """

    def __init__(self) -> None:
        self.dense_dim = STUB_DIM
        self.embed_calls = 0
        self.embedded_texts: list[str] = []
        self.rerank_calls = 0

    @staticmethod
    def _dense(text: str) -> list[float]:
        vec = [0.0] * STUB_DIM
        for token in _tokens(text):
            idx = int(hashlib.sha1(token.encode()).hexdigest(), 16) % STUB_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _sparse(text: str, *, query: bool) -> SparseVec:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            idx = int(hashlib.sha1(token.encode()).hexdigest(), 16) % 100_000
            counts[idx] = 1.0 if query else counts.get(idx, 0.0) + 1.0
        items = sorted(counts.items())
        return SparseVec(indices=[i for i, _ in items], values=[v for _, v in items])

    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        return [self._dense(t) for t in texts], [self._sparse(t, query=False) for t in texts]

    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        return self._dense(text), self._sparse(text, query=True)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Count query-token occurrences, so reranked order is predictable."""
        self.rerank_calls += 1
        wanted = set(_tokens(query))
        return [float(sum(1 for t in _tokens(doc) if t in wanted)) for doc in documents]

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        # Never read the developer's .env: a token or a path set there would
        # otherwise decide what the tests exercise.
        _env_file=None,
        postgres_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        dense_dim=STUB_DIM,
        repos_root=tmp_path / "repos",
        fastcache_dir=tmp_path / "cache",
        eager_model_load=False,
        codebase_collection="test_codebase",
        docs_collection="test_docs",
        max_result_chars=400,
    )


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest_asyncio.fixture
async def store(settings: Settings) -> AsyncIterator[QdrantStore]:
    store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    await store.ensure_collections()
    yield store
    await store.aclose()


@pytest_asyncio.fixture
async def database(settings: Settings) -> AsyncIterator[None]:
    engine = init_engine(settings.postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await dispose_engine()


@pytest.fixture
def search(settings: Settings, store: QdrantStore, embedder: StubEmbedder) -> SearchService:
    return SearchService(settings, store, embedder)


@pytest.fixture
def sync(settings: Settings, store: QdrantStore, embedder: StubEmbedder) -> SyncService:
    return SyncService(settings, store, embedder)


@pytest.fixture
def git_repo(settings: Settings) -> Path:
    """A real git repository with a small Python/TypeScript tree."""
    root = settings.repos_root / "demo"
    (root / "src").mkdir(parents=True)

    (root / "src" / "auth.py").write_text(
        '''"""Authentication helpers."""


class AuthService:
    """Issues and validates credentials."""

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def validate_token(self, token: str) -> bool:
        """Check a bearer token against the configured secret."""
        return token.startswith(self.secret)

    def refresh_token(self, token: str) -> str:
        """Exchange an expiring token for a new one."""
        return token + "-refreshed"


def hash_password(password: str) -> str:
    """One-way hash for storage."""
    return password[::-1]
'''
    )
    (root / "src" / "api.ts").write_text(
        """export interface Session { userId: string; }

export function createSession(userId: string): Session {
  return { userId };
}
"""
    )
    (root / "README.md").write_text("# demo\nNot indexable.\n")
    (root / ".gitignore").write_text("ignored/\n")
    (root / "ignored").mkdir()
    (root / "ignored" / "secret.py").write_text("def hidden():\n    return 1\n")

    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(root),
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True, env=env)
    return root
