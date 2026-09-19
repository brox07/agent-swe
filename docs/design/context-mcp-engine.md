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
| 4 | Reranking | `BAAI/bge-reranker-base` cross-encoder behind `rerank` flag | same |
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
4. If `rerank=true`: take 25 fused candidates, score with the cross-encoder,
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

- `MCP_AUTH_TOKEN` — when set, required as a bearer token on `/mcp`. Unset by default.
- `EXPOSE_INFRA_PORTS` — when false, the 6333/5432 port mappings are dropped via a compose override. True by default.

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

## 7. To verify against pinned versions at implementation time

Claims below are from knowledge, not from a running environment, and are load-bearing:

1. `jinaai/jina-embeddings-v2-base-code` appears in `TextEmbedding.list_supported_models()` for the pinned fastembed. If not, fall back to `nomic-ai/nomic-embed-text-v1.5` (768d, long context) and note the change.
2. `BAAI/bge-reranker-base` appears in `TextCrossEncoder.list_supported_models()`.
3. RRF fusion via `prefetch` + `FusionQuery` requires Qdrant >= 1.10; the spec's pin of v1.12.1 satisfies this.
4. Tree-sitter's Python binding changed its `Language`/`Parser` construction across 0.22-0.24. Grammar packages and the core library are pinned together and the construction idiom matched to the pinned version.
5. Mounting FastMCP's Streamable HTTP ASGI app inside FastAPI requires running its session manager from the host app's lifespan.

---

## 8. Milestone 1 definition of done

- `docker compose up` brings the stack to a healthy `/health`.
- A bind-mounted repository indexes via `sync_repository`, with progress observable through `get_sync_status`.
- `search_codebase` returns correct hits with exact line references, in both
  `rerank=false` and `rerank=true` modes, and identifier queries beat the
  dense-only baseline.
- Re-syncing an unchanged repo is a no-op; changing one file re-indexes that
  file only; deleting a file removes its vectors.
- Claude Code connects to `/mcp` over the tailnet and calls the tools.
- Parser tests over fixtures and retrieval tests over in-memory Qdrant pass.
