# Context MCP Engine

Semantic, AST-aware retrieval over your own codebases, exposed to Claude Code and
other MCP clients over a Tailscale network.

Design decisions and the reasoning behind each departure from the original
specification are in [`docs/design/context-mcp-engine.md`](docs/design/context-mcp-engine.md).
The specification itself is preserved in `docs/spec/`.

## Status

**Milestone 1 (this release): the code path.** `sync_repository`,
`get_sync_status`, and `search_codebase`, with hybrid retrieval and incremental
indexing.

**Milestone 2:** documentation ingestion — `ingest_document` and
`get_best_practices`, over the `best_practices_docs` collection. The collection
and its table are already created; the tools are not yet registered.

## How it works

```
  Claude Code / IDE agent
          │  MCP Streamable HTTP  →  http://<tailnet-ip>:8000/mcp
          ▼
  FastAPI + MCPServer
          │
          ├── Tree-sitter        nested class/method chunking (Python, TypeScript)
          ├── line chunker       Dockerfile stages, Compose services
          └── FastEmbed (ONNX)   jina-code dense 768d · BM25 sparse · bge reranker
          │
          ▼
  Qdrant (named dense+sparse vectors, RRF fusion)  +  PostgreSQL (hashes, jobs)
```

Retrieval fuses dense vector similarity with BM25 sparse matching using
Reciprocal Rank Fusion inside Qdrant, in a single request. Dense similarity alone
is weak at exact identifier lookup, which is the most common query against a code
index; sparse alone misses conceptual queries. `rerank=true` additionally scores a
wider candidate set with a cross-encoder.

## Setup

Repositories are **bind-mounted read-only and never cloned**, so no git
credential exists inside the container.

```bash
cp .env.example .env          # then set REPOS_HOST_PATH
mkdir -p repos
ln -s /path/to/your/project repos/your-project   # or mount a parent directory

docker compose up -d --build
```

First boot downloads roughly 1GB of ONNX models into the `model-cache` volume.
`/health` returns 503 until they are loaded, so the container is only reported
healthy once it can actually serve a query. Subsequent starts reuse the cache.

```bash
curl http://localhost:8000/health
```

### Connecting Claude Code

```bash
claude mcp add --transport http context-engine http://<tailnet-ip>:8000/mcp
```

If you set `ALLOWED_HOSTS`, it must include the exact `host:port` the client
uses, or every request is answered with HTTP 421. Left empty, any Host header is
accepted and a warning is logged at startup.

## Tools

| Tool | Purpose |
|------|---------|
| `search_codebase` | Hybrid semantic + lexical search. Returns file paths with exact line ranges. `rerank=true` for cross-encoder precision. |
| `sync_repository` | Index or re-index a mounted repository. Returns a `job_id` immediately. |
| `get_sync_status` | Progress and outcome of a sync job. |

Indexing is incremental: only files whose content hash changed are re-parsed,
files removed from the working tree have their vectors deleted, and re-syncing an
unchanged repository is a no-op. Because hashing reads the working tree rather
than diffing commits, uncommitted edits are indexed too.

## Security posture

The tailnet is the trust boundary, by explicit decision. Qdrant (6333) and
Postgres (5432) are published as the specification describes, which means any
device on the tailnet reaches them **with no credential**. Two switches tighten
this without code changes:

- `INFRA_BIND=127.0.0.1` — Qdrant and Postgres become reachable only from the
  host itself. The engine talks to both over the internal compose network
  regardless, so nothing breaks.
- `MCP_AUTH_TOKEN=<secret>` — requires `Authorization: Bearer <secret>` on
  `/mcp`. `/health` stays open so Compose can probe it.

## Development

```bash
uv sync --all-extras
uv run pytest                 # 69 tests, no network or containers needed
uv run ruff check src tests
```

The suite runs against a real in-memory Qdrant with a deterministic stub
embedder, so retrieval ordering is asserted exactly rather than depending on a
downloaded model. Tests that exercise the real ONNX models are opt-in, because
they need roughly 1GB of downloads:

```bash
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_models.py -v
```

### Layout

```
src/
├── main.py              FastAPI bootstrap, MCP mount at /mcp, /health
├── config.py            settings
├── db/                  SQLAlchemy models and async sessions
├── vector/
│   ├── embedder.py      FastEmbed ONNX wrapper, off the event loop
│   ├── qdrant.py        collections, payload indexes, deterministic ids
│   └── search.py        RRF fusion, duplicate collapsing, rerank, budget
├── parser/
│   ├── tree_sitter_ast.py   nested Python/TypeScript chunking
│   ├── generic_chunker.py   Dockerfile and YAML blocks
│   └── base.py              chunk model, oversize splitting
├── ingest/git_sync.py   working-tree hashing, prune path, job tracking
└── mcp_server/tools.py  tool registrations
```

`mcp_server/` rather than the specification's `mcp/`: that name shadows the
installed `mcp` SDK package as soon as `src/` lands on `sys.path`.

## Migrations

Alembic runs automatically from the container entrypoint before the server
starts. To add a revision:

```bash
uv run alembic revision --autogenerate -m "description"
```
