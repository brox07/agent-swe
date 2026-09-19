# Technical Specification: Codebase & Best Practices Context MCP Engine

## 1. System Overview

A high-performance semantic retrieval engine exposed via the Model Context Protocol (MCP) using Server-Sent Events (SSE). The system indexes codebases using AST-aware syntax parsing (Tree-sitter) and ingests documentation (PDFs, Markdown, Git repos) into Qdrant for vector retrieval. It supports switchable retrieval modes (fast cosine similarity vs. cross-encoder reranking) and can be accessed securely over a Tailscale private network by MCP clients such as Claude Code, IDE agents, or future bot integrations.

```text
       [ Client (Claude Code / OpenCode / Antigravity) ]
                              │
                              │ MCP via SSE (Tailscale IP:8000/sse)
                              ▼
                ┌───────────────────────────┐
                │  FastAPI + FastMCP Engine │
                └─────────────┬─────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  │  Tree-sitter  │   │  Doc Loader   │   │ FastEmbed ONNX│
  │  (Python, TS, │   │  (PDF, MD,    │   │ (BGE-Small +  │
  │  Docker/YAML) │   │  Git Repos)   │   │ BGE-Reranker) │
  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
          │                   │                   │
          └─────────────┬─────┴───────────────────┘
                        ▼
             ┌─────────────────────┐
             │ Qdrant Vector Engine│  (Port 6333)
             │ PostgreSQL Metadata │  (Port 5432)
             └─────────────────────┘
```

---

## 2. Technology Stack & Core Dependencies

* **Language & Runtime:** Python 3.11+, Docker, Docker Compose
* **API & MCP Framework:** FastAPI, `mcp` (FastMCP implementation over SSE), Uvicorn
* **Vector Database:** Qdrant (Rust-native, SIMD-accelerated, HNSW indexing)
* **Relational Storage:** PostgreSQL 16+ (State tracking, file content hashes, sync logs)
* **ORM & Migrations:** SQLAlchemy 2.0 (asyncio) + `asyncpg`, Alembic
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (384 dimensions) via `fastembed` (ONNX Runtime, sub-5ms CPU inference)
* **Cross-Encoder Reranker:** `BAAI/bge-reranker-base` via `fastembed.rerank`
* **Syntax Parsing:** `tree-sitter`, `tree-sitter-python`, `tree-sitter-typescript`
* **Doc Processing:** `pypdf` / `pymupdf`, `markdown-it-py`, `GitPython`

---

## 3. Storage & Indexing Schemas

### 3.1 Qdrant Collections

#### Collection: `codebase_index`
* **Vector Config:** 384 dimensions, Cosine distance, HNSW on-disk payloads.
* **Payload Schema:**
  ```json
  {
    "repo_name": "string",
    "file_path": "string",
    "commit_sha": "string",
    "language": "python | typescript | dockerfile | yaml",
    "node_type": "function | class | method | block",
    "identifier": "string",
    "start_line": 0,
    "end_line": 0,
    "content": "string"
  }
  ```

#### Collection: `best_practices_docs`
* **Vector Config:** 384 dimensions, Cosine distance.
* **Payload Schema:**
  ```json
  {
    "doc_title": "string",
    "source_type": "pdf | markdown | github",
    "source_url": "string",
    "framework": "string",
    "version": "string",
    "is_best_practice": true,
    "heading_hierarchy": ["string"],
    "content": "string"
  }
  ```

### 3.2 PostgreSQL Tracking Schema

```sql
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    repo_name VARCHAR(255) UNIQUE NOT NULL,
    git_url TEXT NOT NULL,
    last_synced_commit VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE indexed_files (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repo_id, file_path)
);

CREATE TABLE doc_sources (
    id SERIAL PRIMARY KEY,
    source_url TEXT UNIQUE NOT NULL,
    source_type VARCHAR(32) NOT NULL,
    framework VARCHAR(64),
    version_tag VARCHAR(32),
    content_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 4. AST & Document Chunking Strategies

### 4.1 Tree-sitter Code Parser
* **Python:** Traverses AST to extract `function_definition` and `class_definition` nodes. Collects decorators and docstrings within the scope.
* **TypeScript:** Extracts `interface_declaration`, `class_declaration`, `function_declaration`, `method_definition`, and exported arrow functions.
* **Fallback (Dockerfile, Compose, YAML):** Line-based semantic block chunking (e.g., individual `services` definitions in Compose files, multi-stage build segments in Dockerfiles).

### 4.2 Document Ingestion Parser
* **PDF:** Text extraction page-by-page, chunked using sliding recursive token windows (~500 tokens, 50 token overlap).
* **Markdown:** Header-aware chunking (splits on `##` and `###` headers), preserving breadcrumb navigation (`["Parent Section", "Child Section"]`) in the metadata.
* **GitHub Repos / Links:** Recursively clones or fetches specified doc folders, strips non-text assets, and tags records with `source_type="github"`.

---

## 5. FastMCP Tools Specification

The service exposes the following tools over SSE at `/sse`:

### 1. `search_codebase`
* **Description:** Performs semantic AST-aware search over indexed source code.
* **Arguments:**
  * `query` (string, required): Semantic search query or symbol intent.
  * `language` (string, optional): Filter by language (`python`, `typescript`, etc.).
  * `repo_name` (string, optional): Scope retrieval to a specific repository.
  * `rerank` (boolean, default `false`): When `false`, returns top 5 cosine results from Qdrant (~10ms). When `true`, retrieves top 25 candidates and scores them via `bge-reranker-base` before returning top 5 (~50ms).
  * `limit` (integer, default `5`): Max chunks to return.

### 2. `get_best_practices`
* **Description:** Retrieves framework rules, architectural patterns, and style guide snippets.
* **Arguments:**
  * `topic` (string, required): Concept or guideline being queried.
  * `framework` (string, optional): Target framework (e.g., `fastapi`, `react`, `docker`).
  * `version` (string, optional): Target version constraint.
  * `rerank` (boolean, default `false`): Enable cross-encoder reranking.
  * `limit` (integer, default `3`): Number of guidelines to return.

### 3. `sync_repository`
* **Description:** Triggers Git clone/pull, calculates diffs against PostgreSQL file hashes, re-parses modified files via Tree-sitter, and upserts vectors into Qdrant.
* **Arguments:**
  * `repo_url` (string, required): Git repository clone URL.
  * `branch` (string, default `"main"`): Target branch.

### 4. `ingest_document`
* **Description:** Ingests external documentation from a PDF path, Markdown file, or URL.
* **Arguments:**
  * `source_url` (string, required): Local file path or HTTP URL.
  * `source_type` (string, required): `"pdf"`, `"markdown"`, or `"github"`.
  * `framework` (string, optional): Associated framework.
  * `version` (string, optional): Version tag.

---

## 6. Project Layout

```text
context-mcp-engine/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI bootstrap & FastMCP SSE mount
│   ├── config.py                # Pydantic BaseSettings
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py          # SQLAlchemy async session manager
│   │   └── models.py            # Relational models
│   ├── vector/
│   │   ├── __init__.py
│   │   ├── qdrant.py            # Qdrant client & collection management
│   │   └── embedder.py          # FastEmbed ONNX wrapper (dense + reranker)
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── tree_sitter_ast.py   # Tree-sitter Python & TypeScript chunker
│   │   ├── generic_chunker.py   # YAML / Dockerfile chunking
│   │   └── doc_loader.py        # PDF and Markdown header-aware parsers
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── git_sync.py          # Repo clone, hash diffing, AST orchestration
│   │   └── doc_sync.py          # Ingestion coordinator for docs
│   └── mcp/
│       ├── __init__.py
│       └── tools.py             # FastMCP tool registrations
└── tests/
    ├── test_parser.py
    └── test_retrieval.py
```

---

## 7. Baseline Docker Compose Configuration

```yaml
version: "3.8"

services:
  mcp-engine:
    build: .
    container_name: context-mcp-engine
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/context_engine
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - FASTCACHE_DIR=/app/.cache
    volumes:
      - ./data:/app/data
      - model-cache:/app/.cache
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started

  qdrant:
    image: qdrant/qdrant:v1.12.1
    container_name: context-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  postgres:
    image: postgres:16-alpine
    container_name: context-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=context_engine
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  qdrant_storage:
  pg_data:
  model-cache:
```

---

## 8. Implementation Steps for Claude Code

1. **Bootstrap Project Environment:** Initialize `pyproject.toml` with `fastapi`, `uvicorn`, `mcp`, `qdrant-client`, `fastembed`, `tree-sitter`, `tree-sitter-python`, `tree-sitter-typescript`, `sqlalchemy`, `asyncpg`, and `pypdf`.
2. **Database & Vector Clients:** Create `src/db/postgres.py` and `src/vector/qdrant.py` ensuring collections and database tables initialize on startup.
3. **Embedder Wrapper:** Implement `src/vector/embedder.py` using `fastembed.TextEmbedding` (`bge-small-en-v1.5`) and `fastembed.TextCrossEncoder` (`bge-reranker-base`).
4. **AST Chunker:** Implement `src/parser/tree_sitter_ast.py` to extract functions and classes with start/end line bounds and node metadata.
5. **FastMCP Server:** Implement `src/mcp/tools.py` with `@mcp.tool()` decorators for `search_codebase`, `get_best_practices`, `sync_repository`, and `ingest_document`.
6. **Application Entrypoint:** Wire the FastMCP SSE handler into `src/main.py` under the route `/sse`. Expose health check and webhook endpoints on FastAPI.