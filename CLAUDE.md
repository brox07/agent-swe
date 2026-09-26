# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Context MCP Engine: a FastAPI service exposing MCP tools (Streamable HTTP at `/mcp`) for hybrid semantic + BM25 retrieval over bind-mounted code repositories and ingested documentation (books, reference docs, an Obsidian vault). Qdrant holds vectors; Postgres holds file hashes, doc sources and job state. Runs in Docker Compose on a homelab host, reached by clients over Tailscale.

Document roles: `README.md` (setup/usage), `TODO.md` (remaining work, priority order), `docs/design/context-mcp-engine.md` (authoritative design and measured results — wins where it disagrees with the spec), `docs/spec/` (original spec, preserved as written; do not edit).

## Commands

```bash
uv sync --all-extras
uv run pytest                                   # full suite; no network or containers
uv run pytest tests/test_retrieval.py::test_name -v
uv run ruff check src tests                     # line length 100, rules E,F,I,UP,B
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_models.py -v   # real ONNX models, ~1GB download

docker compose up -d --build                    # engine + qdrant + postgres
curl http://localhost:8000/health               # 503 until models are loaded

uv run alembic revision --autogenerate -m "description"   # migrations run automatically from docker/entrypoint.sh
uv run python eval/run.py [--suite docs] [--misses]        # retrieval eval; needs a running, populated engine
uv run python scripts/refresh.py [--vault]                 # re-sync vault + mounted repos via the MCP client
```

The server is served as `uvicorn src.main:create_app --factory`. There is deliberately no module-level `app`: importing `src.main` must not construct a Qdrant client.

## Architecture

`src/main.py` builds an `AppState` (QdrantStore, FastEmbedder, SearchService, SyncService, DocIngestService) and passes the services to `build_mcp_server` in `src/mcp_server/tools.py`. `AppState` and `create_app(state=...)` are injectable so tests can substitute an in-memory Qdrant and a stub embedder. The package is `mcp_server/`, not `mcp/`, because the latter shadows the installed `mcp` SDK.

Two parallel pipelines share the embedder and store, each with its own Qdrant collection (`codebase_index`, `best_practices_docs`):

- **Code**: `ingest/git_sync.py` hashes the *working tree* (not commits, so uncommitted edits are indexed), re-parses only changed files, and prunes vectors for deleted ones. Parsing: `parser/tree_sitter_ast.py` does nested chunking for Python/TypeScript (a class chunk *and* per-method chunks); `parser/generic_chunker.py` does Dockerfile stages / YAML blocks; `parser/base.py` sub-splits anything over the dense model's window (header repeated, `is_truncated` flagged) rather than letting it be silently truncated.
- **Docs**: `docs/sources.py` resolves a target (preset, path under `DOCS_ROOT`, allowlisted https URL, or `vault`) → `docs/loaders.py` (HTML archive, EPUB, PDF, Markdown/GitHub, vault) produces sections → `docs/sections.py` packs sections into prose-sized chunks, keeping code blocks whole → `docs/ingest.py` runs the job. Sources are content-hashed; unchanged ones are skipped unless `force=true`.

Both sync and ingest are background jobs: tools return a `job_id` immediately and progress is read via `get_sync_status` (jobs are `SyncJob` rows). On startup, jobs left running by a restart are marked failed.

Retrieval (`vector/search.py`): dense + sparse prefetch fused with RRF inside Qdrant in a single request, duplicate collapsing (nested chunks overlap), optional cross-encoder rerank (off by default — measured as a wash on both suites for ~15x latency), and a per-hit content budget (`max_result_chars`). Point ids are deterministic UUIDv5 (`vector/qdrant.py`). Payload indexes are required for filtered search to not be a full scan.

`vector/embedder.py`: ONNX inference is blocking, so it runs in a thread pool; bulk embedding is serialized process-wide by a semaphore. Batches are bounded by both count (`embed_batch_size`) and total characters (`embed_batch_chars`) because memory scales with the longest text in a batch — this has caused OOM kills of the container before. Treat changes to batching, model choice or concurrency as memory-sensitive.

## Constraints to preserve

- Repos are mounted read-only and never cloned; no git credentials in the container.
- `ingest_document` reads local files only beneath `docs_root` and fetches only from `DOCS_ALLOWED_HOSTS` (re-checked on every redirect) — it must not become a file-read or SSRF primitive.
- DNS-rebinding protection is configured explicitly from `ALLOWED_HOSTS` (the SDK's localhost-only default would 421 every Tailscale client). `MCP_AUTH_TOKEN`, when set, gates `/mcp` only; `/health` stays open for the Compose probe.
- `config.py` comments record *why* defaults are what they are (often measured regressions). Keep that rationale in sync when changing a value.

## Tests

`tests/conftest.py` provides a real `AsyncQdrantClient(":memory:")`, SQLite via aiosqlite instead of Postgres, a deterministic hashing `StubEmbedder` (so ranking assertions are exact), and a `git_repo` fixture that builds a real git repository. The `settings` fixture passes `_env_file=None` so the developer's `.env` never influences tests. `asyncio_mode = "auto"`.
