# Context MCP Engine — Design Decision Record

Status: agreed, pending implementation
Source: `Technical Specification: Codebase & Best Practices Context MCP Engine`

This record captures the decisions made while reviewing the original technical
specification, and is the authoritative design where the two disagree. Sections
are numbered to match the source spec.

---

## 1. Decisions

| # | Area | Decision | Spec said |
|---|------|----------|-----------|
| 1 | MCP transport | Streamable HTTP at `/mcp` | HTTP+SSE at `/sse` |
| 2 | Dense embedding | `jinaai/jina-embeddings-v2-base-code`, 768d, 8k context | `BAAI/bge-small-en-v1.5`, 384d, 512 ctx |
| 3 | Retrieval | Hybrid dense + BM25 sparse, fused server-side with RRF | Dense cosine only |
| 4 | Reranking | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder behind `rerank` flag (was `bge-reranker-base`; see §9) | `BAAI/bge-reranker-base` |
| 5 | State store | PostgreSQL 16 + SQLAlchemy 2.0 async + Alembic | same |
| 6 | Milestone 1 | Code path only: sync + search, end to end | All four tools at once |
| 7 | Host | Homelab, 16GB+ RAM — both models resident, no quantization | unspecified |
| 8 | Auth | Tailnet is the trust boundary; 6333 and 5432 published | same |
| 9 | Repo access | Read-only bind-mounted local checkouts | Clone from `repo_url` |
| 10 | Sync trigger | Manual tool call, async job handle | Blocking tool call |
| 11 | Chunking | Nested — class chunk *and* per-method chunks | Leaf nodes only |
| 12 | Result size | ~1500 chars per hit, truncation marked, line refs always | Full chunk content |
| 13 | Languages | Python + TypeScript, line-based fallback for YAML/Dockerfile | same |
| 14 | Tests | Unit + integration against `QdrantClient(":memory:")` | Two thin test files |

---

## 2. Rationale for the deltas

### 2.1 Transport (decision 1)

The MCP specification superseded HTTP+SSE with Streamable HTTP in the
2025-03-26 protocol revision. SSE remains supported by current clients but is
the deprecated path; a new server should not ship on it. The FastMCP ASGI app
is mounted into FastAPI at `/mcp`, with the session manager driven from the
FastAPI lifespan so startup and shutdown are ordered correctly.

### 2.2 Embedding model (decision 2)

`bge-small-en-v1.5` has a 512-token window. The spec's own chunking strategy
(§4.1) emits whole `class_definition` nodes, which routinely exceed that — a
300-line class is several thousand tokens. The excess is **silently
truncated**, so the stored vector represents only the head of the class and the
rest of the file is unsearchable while appearing to be indexed. This was the
most serious defect in the original design.

`jina-embeddings-v2-base-code` has an 8k window, removing truncation for
realistic code units, and is trained on code rather than prose. Costs: 768
dimensions (2x storage per vector) and a larger ONNX download. Acceptable on
the chosen host.

Chunks are still guarded: anything exceeding the model window is sub-split with
the enclosing signature and docstring repeated as a header on each part, and
flagged `is_truncated` in the payload. The guard should never fire in practice —
it exists so that a pathological file degrades instead of lying.

Note that the **reranker** window is 512 tokens. Rerank input is therefore the
capped, display-sized text (decision 12), not the full chunk.

### 2.3 Hybrid retrieval (decision 3)

The highest-frequency query against a code index is lexical — "where is
`validate_token`" — and dense vectors are weak at exact identifier match.
Qdrant stores sparse vectors natively and fuses result sets with Reciprocal
Rank Fusion in a single `query_points` call, so hybrid costs one request, not
two plus client-side merging. BM25 sparse vectors come from
`fastembed.SparseTextEmbedding`.

This is the reason the decision had to be made before any code: named vectors
are part of the collection schema, and adding a sparse vector later means a
full reindex.

Query pipeline:

1. Two prefetch branches — dense kNN and sparse BM25 — each pulling
   `limit * 5` candidates.
2. RRF fusion in Qdrant.
3. If `rerank=false`: collapse near-duplicates, return top `limit`. Target ~10-15ms.
4. If `rerank=true`: take 10 fused candidates (originally 25; see §9), score with the cross-encoder,
   return top `limit`. Target ~50-60ms.

### 2.4 Bind-mounted repositories (decision 9)

Repositories are mounted read-only into the container; nothing is cloned and no
git credential ever enters it. Consequences:

- `sync_repository` takes a registered repo **name or mounted path**, not a
  clone URL. The spec's `repo_url` argument is renamed.
- `repositories.git_url` becomes the mount path; the `origin` remote is
  recorded separately when present.
- Change detection is by **file content hash on the working tree**, not by git
  diff between commits, so uncommitted edits are indexed. This is desirable for
  a live dev loop, but it means `commit_sha` in the payload records HEAD plus a
  `dirty` flag rather than identifying the indexed content exactly.
- Files present in Postgres but absent from the working tree have their vectors
  deleted by filter. The spec had no deletion path at all.

### 2.5 Async sync (decision 10)

Parsing and embedding a real repository takes minutes; MCP clients do not wait
that long for a tool result. `sync_repository` therefore validates its
arguments, starts a background task, and returns a `job_id` immediately. A
fifth tool, `get_sync_status`, reports phase, files processed, chunks upserted,
and terminal errors. Jobs are tracked in Postgres so status survives a restart.

### 2.6 Nested chunking (decision 11)

Each class is indexed both as a whole and once per method, with
`parent_identifier` and `node_path` in the payload. This is the only way both
"the auth service" and "how tokens get refreshed" retrieve well. The cost is
~2x vectors and overlapping hits, handled by collapsing a parent and its own
child in the result set — the higher-scoring one wins, the other is dropped.

### 2.7 Result budget (decision 12)

Every hit carries `file_path`, `start_line`, `end_line` and at most ~1500
characters of content, with truncation explicitly marked. A search returning
five hits therefore has a predictable ceiling of roughly 2k tokens instead of an
unbounded one. The agent reads the file directly when it needs the rest, which
it can always do because the location is exact.

---

## 3. Accepted risk: authentication

Confirmed, with the tradeoff understood and accepted: the endpoint has no
application-level authentication, and Qdrant (6333) and Postgres (5432) are
published to the host as in the spec. The tailnet is the entire trust boundary.
Any device on the tailnet reaches Qdrant and Postgres with no credential.

Two lower-cost mitigations are built in but left off by default, so tightening
this later is configuration rather than a rewrite:

- `MCP_AUTH_TOKEN` — when set, required as a bearer token on `/mcp`. Unset by
  default. `/health` stays open so Compose can probe it.
- `INFRA_BIND` — `0.0.0.0` by default, publishing Qdrant and Postgres as the
  specification describes. Setting it to `127.0.0.1` publishes them on the host's
  loopback only, so other tailnet devices cannot reach them, while the engine
  still talks to both over the internal Compose network. (See §8: an earlier plan
  to drop the port mappings through a Compose override does not work, because
  Compose appends `ports` entries rather than replacing them.)

Separately, `ingest_document` (milestone 2) accepts an arbitrary local path or
URL, which is a file-read and SSRF primitive exposed as a tool. Local paths will
be constrained to the mounted `/app/data` tree; URL fetching will be
allowlisted by scheme and host.

---

## 4. Revised storage schema

### 4.1 Qdrant — `codebase_index`

Named vectors: `dense` (768d, cosine) and `sparse` (BM25). `on_disk_payload=True`.

Payload, extending the spec:

```json
{
  "repo_name": "string",
  "file_path": "string",
  "commit_sha": "string",
  "dirty": false,
  "language": "python | typescript | dockerfile | yaml",
  "node_type": "function | class | method | block",
  "identifier": "string",
  "parent_identifier": "string | null",
  "node_path": "Class.method",
  "start_line": 0,
  "end_line": 0,
  "chunk_index": 0,
  "content_hash": "string",
  "is_truncated": false,
  "content": "string"
}
```

Additions over the spec and why:

- `parent_identifier`, `node_path` — required by nested chunking to collapse duplicates and show context.
- `chunk_index`, `is_truncated` — oversized-node sub-splitting.
- `content_hash`, `dirty` — incremental sync against a working tree.

**Payload indexes** on `repo_name`, `file_path`, `language`, and `node_type`.
The spec omitted these; without them every filtered search degrades to a scan.

**Point IDs** are deterministic: `uuid5(NAMESPACE, f"{repo_name}:{file_path}:{node_path}:{chunk_index}")`.
The spec left IDs unspecified, which means re-syncing a file duplicates its
chunks instead of upserting them.

### 4.2 Qdrant — `best_practices_docs`

As specced, with the same named-vector and payload-index treatment. Milestone 2.

### 4.3 PostgreSQL

The spec's three tables stand, with these amendments:

- `repositories.git_url` → mount path, plus nullable `origin_url`, plus `head_sha`.
- New `sync_jobs` table: `id`, `repo_id`, `status`, `phase`, `files_total`,
  `files_done`, `chunks_upserted`, `error`, `started_at`, `finished_at`.
- Managed by Alembic from the first revision — not `create_all` — so the schema
  can evolve without dropping the volume.

---

## 5. MCP tool surface

Milestone 1:

- `search_codebase(query, language?, repo_name?, rerank=false, limit=5)` — as specced.
- `sync_repository(repo, force=false)` — returns `{job_id}`. Renamed argument per decision 9.
- `get_sync_status(job_id)` — new, required by decision 10.

Milestone 2:

- `get_best_practices(topic, framework?, version?, rerank=false, limit=3)` — as specced.
- `ingest_document(source_url, source_type, framework?, version?)` — as specced, with the path/URL constraints from §3.

---

## 6. Decisions made without asking

Reasonable defaults, all cheap to change:

- **Ignored paths**: `.git`, `node_modules`, `.venv`, `__pycache__`, `dist`,
  `build`, `target`, `.next`, lockfiles, minified bundles, anything matching
  `.gitignore`, files over 1MB, and files failing a UTF-8 decode.
- **Blocking inference**: ONNX calls run in a thread executor. Running them
  directly on the event loop would stall every concurrent request, including
  the MCP session keepalive.
- **Model warm-up**: both models load and run one dummy inference during
  startup, before the health check reports ready. First boot downloads ~800MB
  to the `model-cache` volume, so a cold `docker compose up` is slow once.
- **Compose**: drop the obsolete `version: "3.8"` key.
- **Health check**: `/health` reports liveness plus model-loaded and
  Qdrant/Postgres reachability, so `docker compose` can gate on readiness
  rather than process start.

---

## 7. Verification outcomes

The five load-bearing claims were checked against the actually-installed
packages. Four held; one was wrong.

| # | Claim | Outcome |
|---|-------|---------|
| 1 | `jinaai/jina-embeddings-v2-base-code` is served by the pinned fastembed | **Confirmed.** fastembed 0.8.0, `dim=768`, 0.64GB. No fallback needed. |
| 2 | `BAAI/bge-reranker-base` is a supported cross-encoder | **Confirmed.** Present in `TextCrossEncoder.list_supported_models()`. |
| 3 | RRF fusion over named dense+sparse vectors, with a payload filter | **Confirmed** on qdrant-client 1.19.1, exercised in the test suite. |
| 4 | Tree-sitter construction idiom for the pinned grammars | **Confirmed** on tree-sitter 0.26.0: `Language(ts_python.language())` then `Parser(lang)`. `tree_sitter_typescript` exports `language_typescript()` and `language_tsx()`. |
| 5 | FastMCP's Streamable HTTP app mounts inside FastAPI via its session manager | **Corrected.** See below. |

### 7.1 Correction: FastMCP no longer exists under that name

`mcp` resolved to 2.2.0, which renamed `FastMCP` to `MCPServer`
(`mcp.server.mcpserver.MCPServer`); importing `mcp.server.fastmcp` raises with a
pointer to the migration guide. The 2.x API is adopted rather than pinning
`mcp<2`. The session manager is created lazily by `streamable_http_app()` and is
unreachable before that call, so the app factory runs in a fixed order:
build the server, call `streamable_http_app()`, then run `session_manager` from
the FastAPI lifespan.

Every reference to "FastMCP" in the source specification should be read as
`MCPServer`.

### 7.2 New finding: the SDK would have rejected every Tailscale client

`streamable_http_app()` defaults to `host="127.0.0.1"` and, seeing a localhost
bind address, **auto-enables DNS-rebinding protection with only localhost Host
headers allowed**. Clients reach this service on a tailnet IP, so every request
would have been answered with `HTTP 421 Misdirected Request` — a failure that
looks like a client bug and is invisible until something tries to connect from
another machine.

The policy is therefore always passed explicitly. `ALLOWED_HOSTS` is empty by
default, which disables the check and logs a warning at startup; setting it to
the `host:port` clients actually use re-enables protection. Both paths are
covered by tests.

## 8. Further departures decided during implementation

- **`src/mcp_server/` instead of the spec's `src/mcp/`.** That directory name
  shadows the installed `mcp` SDK package the moment `src/` is on `sys.path`.
- **`chunk_total` added; `is_truncated` narrowed.** Splitting an oversized node
  loses nothing, so `chunk_index`/`chunk_total` describe parts and `is_truncated`
  is reserved for content genuinely dropped — which now happens only when a
  single line exceeds the whole budget, as in a minified file.
- **`EXPOSE_INFRA_PORTS` replaced by `INFRA_BIND`.** §3 originally promised an
  env var that dropped the 6333/5432 mappings through a Compose override. Compose
  *appends* `ports` entries on override and cannot remove them, so that would not
  have worked. `INFRA_BIND=127.0.0.1` achieves the same end by publishing those
  ports on the host's loopback only, out of reach of other tailnet devices.
- **No module-level `app`.** Building the app at import time constructed a Qdrant
  client and attempted a version handshake on mere import. The service runs as
  `uvicorn src.main:create_app --factory`.
- **`node_type` filter added to `search_codebase`,** and `interface` added to the
  payload's node-type vocabulary, which the spec's enum omitted despite §4.1
  calling for TypeScript interface extraction.
- **Test-only dependencies:** `aiosqlite`, `httpx`, `asgi-lifespan`, so the
  incremental-sync and MCP-protocol paths are testable without containers.

### 8.1 Bug found by the test suite

`git ls-files` still reports a tracked file after it is deleted from the working
tree but before the deletion is committed. The first implementation therefore
counted such a file as present, failed to read it, and left its vectors in Qdrant
forever — the exact dev-loop case that motivated working-tree hashing. Files are
now filtered on existence, and a file that cannot be read is deliberately not
marked as seen so the prune path reclaims it.

## 9. Verification status

Passing locally, with no network and no containers required:

- **69 tests green**, `ruff` clean across `src/`, `tests/`, and `alembic/`.
  Coverage spans nested AST chunking and span boundaries, oversize splitting,
  the Dockerfile/YAML fallbacks, hybrid RRF retrieval against a real in-memory
  Qdrant, duplicate collapsing, the result budget, rerank ordering, incremental
  sync (no-op re-sync, single-file change, deletion, shrink, new file), path
  traversal refusal, job state transitions, and a full MCP handshake with
  `tools/list` and `tools/call` over the real Streamable HTTP transport.
- **Migration applies** and produces the expected schema.
- **`docker compose config`** validates; `uv sync --frozen --no-dev` (the image's
  build step) resolves; the `uvicorn ... --factory` target imports.

Not verifiable in the development container. All but the last were since
verified on a WSL2 host (6 cores, 23GB RAM, Docker 29.7) on 2026-09-19:

- **`docker compose up` reaching a healthy `/health`** — verified. Healthy in
  ~70s on first boot including the model download, ~20s after.
- **Real model load and inference** — verified. `RUN_MODEL_TESTS=1` passes all
  four tests.
- **Qdrant payload index creation** — verified. Both collections carry their
  keyword indexes on a real server.
- **Claude Code connecting over the tailnet** — still open.

The live run found a defect the suite could not: **every sync crashed on a real
deployment.** The engine runs as root while bind-mounted checkouts belong to the
host user, and git refuses a repository owned by someone else ("dubious
ownership"). Tests create their repositories as the user running them, so the
check never fired. Mounted repositories are now opened with `safe.directory`
scoped to that one path, passed as command-line config so the container's git
config is never loosened; `GIT_TEST_ASSUME_DIFFERENT_OWNER` reproduces the check
in the suite. The same run exposed that any git failure in file listing fell
back silently to a filesystem walk, which ignores `.gitignore`; git failures now
fail the job instead, and only a directory that is not a repository is walked.

Measured on that host, against this repository (29 files, 234 chunks):

| Operation | Measured | Target (§2.3) |
|-----------|----------|---------------|
| First full index | 6m14s | — |
| Re-sync, unchanged | 0.1s, 0 chunks written | no-op |
| Search, `rerank=false` | 42–113ms | 10–15ms |
| Search, `rerank=true` | 8–11s | 50–60ms |
| Engine resident memory, just after indexing | 10.3GiB | — |

**The rerank target is not reachable on CPU with `bge-reranker-base`.** The
latency is pure inference — the same 25-pair batch takes ~12s on the host
outside Docker — at roughly 0.5s per pair. Indexing cost has the same cause: the
8k window that decision 2 bought is paid for on every long chunk.

### 9.1 Reranker replaced

Five supported cross-encoders were compared on 12 hand-labelled queries against
this repository, each reranking the same 25 fused candidates:

| Reranker | Params | MRR | Latency / query |
|----------|--------|-----|-----------------|
| *fusion only, no rerank* | — | 0.637 | — |
| `BAAI/bge-reranker-base` | 278M | 0.561 | 13.7s |
| `jinaai/jina-reranker-v1-turbo-en` | 38M | 0.549 | 2.9s |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 22M | 0.531 | 2.5s |
| `jinaai/jina-reranker-v1-tiny-en` | 33M | 0.528 | 2.0s |
| `Xenova/ms-marco-MiniLM-L-12-v2` | 33M | 0.521 | 5.0s |

The quality differences between rerankers are within noise at 12 queries;
the latency differences are not. `ms-marco-MiniLM-L-6-v2` was chosen as the
smallest, with the most consistent worst case (no correct hit ranked below 5th).
Fusion placed every correct hit it found within its top 7, so candidates were
cut from 25 to 10. Measured live afterwards: **0.66–0.88s** per reranked query,
down from 8–11s.

**No reranker beat fusion alone on this set.** `rerank` stays off by default.
Whether it earns its place should be settled against a larger labelled set over
a real target repository, not this one; see `TODO.md`.

The same run found that `docker-compose.yml` passed none of the model or
retrieval settings in `.env` to the engine — `RERANKER_MODEL`, `DENSE_MODEL`,
`RERANK_CANDIDATES` and the rest were documented but silently ignored. The
engine now reads `.env` through `env_file`.

## 10. Milestone 1 definition of done

- [x] A bind-mounted repository indexes via `sync_repository`, with progress
      observable through `get_sync_status`.
- [x] `search_codebase` returns correct hits with exact line references in both
      `rerank=false` and `rerank=true` modes.
- [x] Re-syncing an unchanged repo is a no-op; changing one file re-indexes that
      file only; deleting a file removes its vectors.
- [x] Parser tests over fixtures and retrieval tests over in-memory Qdrant pass.
- [x] `docker compose up` brings the stack to a healthy `/health`.
- [ ] Claude Code connects to `/mcp` over the tailnet and calls the tools.
      *(Stack verified over
      localhost; the tailnet leg is still open.)*

## 11. Milestone 2 — documentation

### 11.1 Sources

Reference documentation comes from each project's own published artifact, never
from crawling: docs.python.org's HTML archive, Read the Docs' htmlzip for pytest,
docs.sqlalchemy.org's zip, and the Markdown sources on GitHub at a release tag
for FastAPI and Pydantic (both publish MkDocs sites with no downloadable build).
Presets pin each version so a re-ingest is reproducible.

### 11.2 One sectioner for all HTML

Sphinx pages and EPUB chapters are both HTML, so one walker handles both.
Headings open sections; so do Sphinx API entries (`<dl class="py function">`
and kin), one level below any heading, so `asyncio.open_connection` is its own
chunk rather than a paragraph in a page-long "Streams" section. Heading level is
`max(tag level, <section> nesting depth)`: O'Reilly EPUBs mark every level
`<h1>` and nest `<section>`s instead, which by tag alone flattened "Chapter 5 >
Infinite Recursion" to "Infinite Recursion".

### 11.3 Chunking

Sections pack into ~2000-character chunks; a chunk never spans two sections, so
each has one heading trail, and that trail is embedded with the text. Code blocks
are kept whole up to 6000 characters. Prose blocks over the target are split on
lines — a PDF page has no paragraph breaks and would otherwise be one block.

### 11.4 Defects found against real sources

Each loader was run against the real source before being trusted:

| Source | Defect | Effect had it shipped |
|--------|--------|-----------------------|
| pytest | Single-page build inlines every page inside `div.toctree-wrapper`, which was dropped as navigation | 8 sections from 1.3M characters |
| FastAPI | Includes (`{* ../../docs_src/... *}`) resolve against the `mkdocs.yml` directory, not the page | 443 code examples replaced by placeholders |
| FastAPI | `release-notes.md` | 42% of the corpus was changelog |
| O'Reilly EPUB | Every heading level is `<h1>` | Chapter titles missing from every section trail |

### 11.5 Throughput

The dense model pads each batch to its longest text, so one long listing among
short chunks made the whole batch pay for its length. Embedding in length order
measured 1.5x faster on a real book (1.1k → 1.7k chars/s). Point ids derive from
chunk position, so processing order does not affect what is stored; a real-model
test asserts each vector still returns aligned with its text.

### 11.6 Jobs and restarts

Ingest reuses the sync job table and `get_sync_status`. Jobs are in-process
tasks, so a restart ends them without a terminal state; startup now marks any
`pending`/`running` job failed with a restart message, rather than leaving a
poller waiting forever. One ingest runs at a time: embedding saturates the CPU.

### 11.7 Security

Local paths must resolve under `DOCS_ROOT` (`/app/data`). URLs must be https on
a `DOCS_ALLOWED_HOSTS` entry or its subdomain, checked on every redirect hop.
Downloads are capped at 300MB.

