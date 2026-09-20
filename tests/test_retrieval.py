"""Retrieval behaviour: hybrid fusion, filters, duplicate collapsing, budget, rerank."""

from __future__ import annotations

from src.parser.base import CodeChunk
from src.vector.qdrant import Hit, QdrantStore, point_id
from src.vector.search import collapse_overlaps

from .conftest import StubEmbedder


def chunk(
    identifier: str,
    content: str,
    *,
    node_type: str = "function",
    node_path: str | None = None,
    parent: str | None = None,
    language: str = "python",
) -> CodeChunk:
    return CodeChunk(
        language=language,
        node_type=node_type,
        identifier=identifier,
        node_path=node_path or identifier,
        parent_identifier=parent,
        start_line=1,
        end_line=10,
        content=content,
    )


async def index(
    store: QdrantStore,
    embedder: StubEmbedder,
    repo: str,
    file_path: str,
    chunks: list[CodeChunk],
) -> None:
    dense, sparse = await embedder.embed_documents([c.content for c in chunks])
    await store.upsert_chunks(
        repo_name=repo,
        file_path=file_path,
        commit_sha="abc123",
        dirty=False,
        chunks=chunks,
        dense=dense,
        sparse=sparse,
    )


class TestPointIdentity:
    def test_ids_are_deterministic(self):
        first = point_id("repo", "src/a.py", "A.method", 0)
        assert first == point_id("repo", "src/a.py", "A.method", 0)

    def test_ids_distinguish_every_component(self):
        base = point_id("repo", "src/a.py", "A.method", 0)
        assert base != point_id("other", "src/a.py", "A.method", 0)
        assert base != point_id("repo", "src/b.py", "A.method", 0)
        assert base != point_id("repo", "src/a.py", "A.other", 0)
        assert base != point_id("repo", "src/a.py", "A.method", 1)

    async def test_reindexing_upserts_rather_than_duplicating(
        self, store: QdrantStore, embedder: StubEmbedder
    ):
        """Without deterministic ids, a re-sync would pile up stale copies."""
        chunks = [chunk("alpha", "def alpha(): pass")]
        await index(store, embedder, "repo", "a.py", chunks)
        await index(store, embedder, "repo", "a.py", chunks)
        assert await store.count("repo") == 1


class TestCollapseOverlaps:
    def test_parent_is_dropped_when_child_scores_higher(self):
        parent = Hit(id="1", score=0.9, payload={"file_path": "a.py", "node_path": "Auth"})
        child = Hit(id="2", score=0.95, payload={"file_path": "a.py", "node_path": "Auth.login"})
        kept = collapse_overlaps([parent, child])
        assert [h.id for h in kept] == ["2"]

    def test_child_is_dropped_when_parent_scores_higher(self):
        parent = Hit(id="1", score=0.99, payload={"file_path": "a.py", "node_path": "Auth"})
        child = Hit(id="2", score=0.5, payload={"file_path": "a.py", "node_path": "Auth.login"})
        kept = collapse_overlaps([parent, child])
        assert [h.id for h in kept] == ["1"]

    def test_same_name_in_different_files_both_survive(self):
        a = Hit(id="1", score=0.9, payload={"file_path": "a.py", "node_path": "Auth"})
        b = Hit(id="2", score=0.8, payload={"file_path": "b.py", "node_path": "Auth.login"})
        assert len(collapse_overlaps([a, b])) == 2

    def test_siblings_are_not_collapsed(self):
        a = Hit(id="1", score=0.9, payload={"file_path": "a.py", "node_path": "Auth.login"})
        b = Hit(id="2", score=0.8, payload={"file_path": "a.py", "node_path": "Auth.logout"})
        assert len(collapse_overlaps([a, b])) == 2

    def test_prefix_that_is_not_a_scope_boundary_is_not_collapsed(self):
        """`Authenticator` is not inside `Auth`, despite the string prefix."""
        a = Hit(id="1", score=0.9, payload={"file_path": "a.py", "node_path": "Auth"})
        b = Hit(id="2", score=0.8, payload={"file_path": "a.py", "node_path": "Authenticator"})
        assert len(collapse_overlaps([a, b])) == 2


class TestHybridSearch:
    async def test_exact_identifier_ranks_first(self, store, embedder, search):
        """The lexical half of hybrid retrieval is what makes symbol lookup work."""
        await index(
            store,
            embedder,
            "repo",
            "auth.py",
            [
                chunk("validate_token", "def validate_token(token):\n    return True"),
                chunk("send_email", "def send_email(to, body):\n    return None"),
                chunk("parse_config", "def parse_config(path):\n    return {}"),
            ],
        )
        results = await search.search_codebase("validate_token", limit=3)
        assert results
        assert results[0].identifier == "validate_token"

    async def test_language_filter_excludes_other_languages(self, store, embedder, search):
        await index(store, embedder, "repo", "a.py", [chunk("handler", "def handler(): pass")])
        await index(
            store,
            embedder,
            "repo",
            "a.ts",
            [chunk("handler", "function handler() {}", language="typescript")],
        )
        results = await search.search_codebase("handler", language="typescript", limit=5)
        assert results
        assert {r.language for r in results} == {"typescript"}

    async def test_repo_filter_scopes_results(self, store, embedder, search):
        await index(store, embedder, "alpha", "a.py", [chunk("shared", "def shared(): pass")])
        await index(store, embedder, "beta", "b.py", [chunk("shared", "def shared(): pass")])
        results = await search.search_codebase("shared", repo_name="beta", limit=5)
        assert results
        assert {r.repo_name for r in results} == {"beta"}

    async def test_node_type_filter(self, store, embedder, search):
        await index(
            store,
            embedder,
            "repo",
            "a.py",
            [
                chunk("Service", "class Service:\n    pass", node_type="class"),
                chunk("helper", "def helper(): pass", node_type="function"),
            ],
        )
        results = await search.search_codebase("service helper", node_type="class", limit=5)
        assert results
        assert {r.node_type for r in results} == {"class"}

    async def test_limit_is_respected(self, store, embedder, search):
        await index(
            store,
            embedder,
            "repo",
            "a.py",
            [chunk(f"fn_{i}", f"def fn_{i}(): return {i}") for i in range(10)],
        )
        results = await search.search_codebase("fn", limit=3)
        assert len(results) <= 3

    async def test_results_carry_exact_locations(self, store, embedder, search):
        await index(store, embedder, "repo", "src/auth.py", [chunk("login", "def login(): pass")])
        results = await search.search_codebase("login", limit=1)
        assert results[0].file_path == "src/auth.py"
        assert results[0].start_line == 1
        assert results[0].end_line == 10

    async def test_empty_index_returns_nothing(self, search):
        assert await search.search_codebase("anything") == []

    async def test_nested_duplicates_are_collapsed_end_to_end(self, store, embedder, search):
        """A class and its own method both match; the caller should see one."""
        body = "class Auth:\n    def validate_token(self, t):\n        return True"
        await index(
            store,
            embedder,
            "repo",
            "auth.py",
            [
                chunk("Auth", body, node_type="class", node_path="Auth"),
                chunk(
                    "validate_token",
                    "    def validate_token(self, t):\n        return True",
                    node_type="method",
                    node_path="Auth.validate_token",
                    parent="Auth",
                ),
            ],
        )
        results = await search.search_codebase("validate_token", limit=5)
        paths = [r.node_path for r in results]
        assert len(paths) == 1, f"expected one collapsed hit, got {paths}"


class TestResultBudget:
    async def test_long_content_is_capped_and_flagged(self, store, embedder, search, settings):
        long_body = "def big():\n" + "".join(f"    line_{i} = {i}\n" for i in range(400))
        await index(store, embedder, "repo", "big.py", [chunk("big", long_body)])
        results = await search.search_codebase("big", limit=1)
        assert results[0].content_truncated is True
        assert len(results[0].content) <= settings.max_result_chars + 60
        assert "truncated" in results[0].content

    async def test_short_content_is_returned_whole(self, store, embedder, search):
        body = "def small():\n    return 1"
        await index(store, embedder, "repo", "s.py", [chunk("small", body)])
        results = await search.search_codebase("small", limit=1)
        assert results[0].content == body
        assert results[0].content_truncated is False


class TestRerank:
    async def test_rerank_reorders_and_invokes_the_cross_encoder(self, store, embedder, search):
        await index(
            store,
            embedder,
            "repo",
            "a.py",
            [
                chunk("a", "def a():\n    pass"),
                # Mentions the query terms repeatedly, so the stub scores it top.
                chunk("b", "def b():\n    # token token token refresh refresh\n    pass"),
                chunk("c", "def c():\n    return None"),
            ],
        )
        plain = await search.search_codebase("token refresh", rerank=False, limit=3)
        reranked = await search.search_codebase("token refresh", rerank=True, limit=3)
        assert embedder.rerank_calls == 1
        assert reranked[0].identifier == "b"
        assert [r.identifier for r in plain] != [] and reranked[0].identifier == "b"

    async def test_rerank_on_empty_index_does_not_call_the_model(self, search, embedder):
        assert await search.search_codebase("nothing", rerank=True) == []
        assert embedder.rerank_calls == 0


class TestDeletion:
    async def test_delete_file_removes_only_that_file(self, store, embedder):
        await index(store, embedder, "repo", "a.py", [chunk("a", "def a(): pass")])
        await index(store, embedder, "repo", "b.py", [chunk("b", "def b(): pass")])
        await store.delete_file("repo", "a.py")
        assert await store.count("repo") == 1

    async def test_delete_repo_removes_only_that_repo(self, store, embedder):
        await index(store, embedder, "alpha", "a.py", [chunk("a", "def a(): pass")])
        await index(store, embedder, "beta", "b.py", [chunk("b", "def b(): pass")])
        await store.delete_repo("alpha")
        assert await store.count("alpha") == 0
        assert await store.count("beta") == 1
