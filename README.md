# Context MCP Engine

Semantic, AST-aware retrieval over your own codebases, exposed to Claude Code and
other MCP clients over a Tailscale network.

| Document | What it's for |
|----------|---------------|
| This README | Setting it up and using it |
| [`TODO.md`](TODO.md) | What's left, in priority order |
| [`docs/design/context-mcp-engine.md`](docs/design/context-mcp-engine.md) | Why it's built this way; measured results |
| `docs/spec/` | The original specification, preserved as written |

## Status

**Milestone 1: the code path — done.** `sync_repository`, `get_sync_status`, and
`search_codebase`, with hybrid retrieval and incremental indexing. Verified end to
end against a real Docker deployment, including an MCP session over the tailnet
address.

**Milestone 2: documentation — done.** `ingest_document`, `get_best_practices`,
`list_doc_sources`, and `forget_document`, over the `best_practices_docs`
collection. Indexed on the host it was built for: 19 EPUB/PDF books, the Python
3.14 docs, FastAPI, Pydantic, SQLAlchemy 2.0 and pytest, an Obsidian vault, and
10 repositories in the code collection.

Retrieval quality is measured, not assumed — see [`eval/`](eval/README.md):
MRR@10 of 0.759 on documentation and 0.718 on code, with reranking a wash on
both.

## How it works

```
  Claude Code / IDE agent
          │  MCP Streamable HTTP  →  http://<tailnet-ip>:8000/mcp
          ▼
  FastAPI + MCPServer
          │
          ├── Tree-sitter        nested class/method chunking (Python, TypeScript)
          ├── line chunker       Dockerfile stages, Compose services
          └── FastEmbed (ONNX)   jina-code dense 768d · BM25 sparse · MiniLM reranker
          │
          ▼
  Qdrant (named dense+sparse vectors, RRF fusion)  +  PostgreSQL (hashes, jobs)
```

Retrieval fuses dense vector similarity with BM25 sparse matching using
Reciprocal Rank Fusion inside Qdrant, in a single request. Dense similarity alone
is weak at exact identifier lookup, which is the most common query against a code
index; sparse alone misses conceptual queries. `rerank=true` additionally scores a
wider candidate set with a cross-encoder. On the hardware measured so far it
costs ~0.7s per query and has not beaten fusion alone, so leave it off unless a
query's top hits look wrong.

## Setup

Repositories are **bind-mounted read-only and never cloned**, so no git
credential exists inside the container.

```bash
cp .env.example .env
mkdir -p data/books      # before the first `up`, or Docker creates data/ as root
```

Then edit `.env`:

- `REPOS_HOST_PATH` — a directory whose **subdirectories** are the repositories
  to index. Pointing it at the folder that holds all your checkouts (e.g.
  `~/code`) is simplest: each one is then addressable by its directory name.
  Symlinks don't work here — a symlink inside the mount points at a path that
  doesn't exist inside the container.
- `INFRA_BIND=127.0.0.1` — recommended. See [Security posture](#security-posture).
- `POSTGRES_HOST_PORT` / `QDRANT_HOST_PORT` — change these if something on the
  host already uses 5432 or 6333. Only external tools see these ports.

```bash
docker compose up -d --build
```

First boot downloads roughly 700MB of ONNX models into the `model-cache` volume.
`/health` returns 503 until they are loaded, so the container is only reported
healthy once it can actually serve a query. Subsequent starts reuse the cache.

```bash
curl http://localhost:8000/health
```

### Connecting Claude Code on the same machine

```bash
claude mcp add --scope user --transport http context-engine http://localhost:8000/mcp
```

`--scope user` makes it available in every project, not just the current
directory. Run `/mcp` inside Claude Code to confirm it shows as connected.

### Connecting from another machine over Tailscale

Both machines join the same tailnet; the laptop then reaches the engine by the
host's tailnet name. Nothing is exposed to the internet.

**On the host** (the machine running Docker):

1. Install Tailscale and sign in. On Windows, open Tailscale from the Start menu
   or tray and choose **Log in**. On Linux: `sudo tailscale up`.
2. On Windows, turn on **Preferences → Run unattended**, so the host stays on
   the tailnet when you're signed out of Windows.
3. Note the host's tailnet name and address:
   ```bash
   tailscale status        # first line: 100.x.y.z  <host-name>  ...
   ```
4. Keep the host reachable: set Windows sleep to **Never** while plugged in, and
   enable Docker Desktop's **Start Docker Desktop when you sign in**. The
   containers restart on their own (`restart: unless-stopped`).

With Docker Desktop on Windows, published ports listen on Windows itself, and
Docker Desktop installs a firewall rule allowing inbound traffic on them. You do
not need Tailscale inside WSL.

**On a new laptop, start to finish:**

1. Install Tailscale, sign in with the same account, and confirm the host is
   listed: `tailscale status` should show `broxworx-tuf` (or your host's name).
2. Check the engine answers — no token needed for this:
   ```bash
   curl http://<host-name>:8000/health      # expect "status":"ready"
   ```
3. Register it with Claude Code, with the token from the host's `.env`:
   ```bash
   claude mcp add --scope user --transport http context-engine \
     http://<host-name>:8000/mcp --header "Authorization: Bearer <token>"
   ```
4. Run `/mcp` in Claude Code and confirm `context-engine` is connected.

Nothing else is needed: no checkout of this repository, no models, no API key.
Search results name files and line ranges on the **host's** copy, so to open a
file you still need that repository checked out locally.

**On the laptop:**

5. Install Tailscale and sign in with **the same account**.
6. Check the engine is reachable (MagicDNS is on by default; use the `100.x`
   address if the name doesn't resolve):
   ```bash
   curl http://<host-name>:8000/health       # expect "status":"ready"
   ```
7. Register it with Claude Code:
   ```bash
   claude mcp add --scope user --transport http context-engine http://<host-name>:8000/mcp
   ```
8. Start Claude Code, run `/mcp`, and confirm `context-engine` is connected.

**Authentication.** Anyone on your tailnet can otherwise use the engine, which
matters once personal notes are indexed. Set a token in the host's `.env`:

```bash
MCP_AUTH_TOKEN=$(openssl rand -hex 32)      # then: docker compose up -d
```

and add it on the laptop:

```bash
claude mcp add --scope user --transport http context-engine http://<host-name>:8000/mcp \
  --header "Authorization: Bearer <token>"
```

**Keeping the host reachable.** Docker Desktop starts at sign-in, so the host
must be signed in to Windows — locked is fine, signed out is not. Tailscale needs
**Run unattended** for the same reason. A third-party VPN client on either
machine can capture the routes Tailscale needs; that is the first thing to
check if the tailnet works everywhere except here.

**If it doesn't connect:**

| Symptom | Cause |
|---------|-------|
| `curl` times out | Host asleep, Tailscale signed out on one side, or a firewall. On the host, `tailscale ping <laptop-name>` should succeed. |
| HTTP 421 | `ALLOWED_HOSTS` is set and doesn't include `<host-name>:*`. Leave it empty or add the name. |
| HTTP 401 | `MCP_AUTH_TOKEN` is set and the laptop isn't sending it. |
| `/health` returns 503 | Models still loading. Normal for ~20s after start, longer on first boot. |

## Using it

Index a repository once, then search it. In Claude Code you can simply ask —
*"sync the agent-swe repository"*, *"search the codebase for where tokens are
validated"* — and it will call the tools. Syncing is incremental, so re-run it
whenever you want recent edits picked up; unchanged files cost nothing.

Search results carry `repo_name`, a `file_path` relative to that repository, and
exact `start_line`/`end_line`. Paths refer to the **host's** checkout: from a
laptop, you need your own checkout of the repository to open the file.

To make Claude Code reach for the engine without being asked, add a line to the
project's `CLAUDE.md`:

```markdown
This repository is indexed in the `context-engine` MCP server as `<repo-name>`.
Use `search_codebase` to locate code before reading files.
```

### Documentation and books

Ask in plain language — *"what do the docs say about cancelling an asyncio
TaskGroup"*, *"search my Rust books for atomics ordering"* — and Claude Code calls
`get_best_practices`. Each result names the document, the section trail (e.g.
*Chapter 5. Conditionals and Recursion > Infinite Recursion*), a location (a page,
chapter file, or anchor), and for the Python and SQLAlchemy docs a link.

Filters: `framework` (`python`, `rust`, `fastapi`, `pydantic`, `sqlalchemy`,
`pytest`, `security`, or whatever you tagged at ingest), `version`, and
`source_type` (`epub`/`pdf` for books, `html_archive`/`github` for reference docs).

**Adding documentation.** `ingest_document` takes one of:

- **A preset name** — `python`, `fastapi`, `pydantic`, `sqlalchemy`, `pytest`.
  Each is pinned to a version and downloaded from the official source: the
  archive python.org publishes, the Read the Docs download, or the project's
  Markdown docs on GitHub at a release tag.
- **A path under `data/`** — a single book (`books/OReillys/effectiverust.epub`)
  or a whole directory (`books`). Where a title exists as both EPUB and PDF, the
  EPUB is used: it keeps chapter structure and code listings as markup, while
  PDF text extraction loses both.
- **An https URL** on a host in `DOCS_ALLOWED_HOSTS`.
- **An Obsidian vault** — set `VAULT_HOST_PATH` in `.env` and ingest `vault`.
  The whole vault is one source, so notes deleted since the last run lose their
  chunks on the next one; unchanged vaults are skipped. Notes are tagged
  `framework=notes`, each note's filename titles its sections (most notes have
  no H1), frontmatter `tags`/`type` stay searchable, wikilinks become their
  display text, and `.obsidian` and `.trash` are skipped. `VAULT_EXCLUDE` drops
  further folders — an archive folder of superseded notes otherwise competes
  with current ones on every query.

Pass `framework` so searches can be filtered by it. Ingestion is a background
job — poll `get_sync_status` — and unchanged sources are skipped on a re-run;
`force=true` re-embeds anyway.

Books go in `data/books/` on the host. From Windows, paste
`\\wsl$\Ubuntu\home\<you>\...\agent-swe\data\books` into Explorer.

**How long it takes.** Embedding runs on CPU at very roughly 1–3k characters a
second, slower for code-heavy text. A typical 400-page book takes 5–10 minutes;
the Python docs about an hour; everything configured here, several hours. Jobs
run one at a time. A restart kills running jobs; they're marked failed on the
next start, so re-issue them.

### Tools

| Tool | Purpose |
|------|---------|
| `search_codebase` | Hybrid semantic + lexical search over code. Filters: `repo_name`, `language`, `node_type`. Returns file paths with exact line ranges. `rerank=true` for cross-encoder rescoring. |
| `sync_repository` | Index or re-index a mounted repository, by directory name. Returns a `job_id` immediately. |
| `get_best_practices` | Hybrid search over documentation and books. Filters: `framework`, `version`, `source_type`. |
| `ingest_document` | Index a preset, a book or directory of books, or an allowlisted URL. Returns a `job_id` immediately. |
| `list_doc_sources` | What documentation is indexed, with versions and sizes. |
| `forget_document` | Drop one indexed source entirely: its vectors and its record. |
| `get_sync_status` | Progress and outcome of any background job. |

A first sync of a small repository (~30 files) takes a few minutes on CPU; the
dense model's 8k-token window is the cost. Later syncs only touch changed files.

Indexing is incremental: only files whose content hash changed are re-parsed,
files removed from the working tree have their vectors deleted, and re-syncing an
unchanged repository is a no-op. Because hashing reads the working tree rather
than diffing commits, uncommitted edits are indexed too.

## Security posture

The tailnet is the trust boundary, by explicit decision. With the default
`INFRA_BIND=0.0.0.0`, Qdrant (6333) and Postgres (5432) are published as the
specification describes, which means any device on the tailnet reaches them
**with no credential**. Two switches tighten
this without code changes:

- `INFRA_BIND=127.0.0.1` — Qdrant and Postgres become reachable only from the
  host itself. The engine talks to both over the internal compose network
  regardless, so nothing breaks.
- `MCP_AUTH_TOKEN=<secret>` — requires `Authorization: Bearer <secret>` on
  `/mcp`. `/health` stays open so Compose can probe it.

## Keeping the index current

Nothing watches the filesystem. `scripts/refresh.py` re-ingests the vault and
re-syncs every mounted repository; both are content-hashed, so unchanged sources
cost a scan and no embedding.

```bash
uv run python scripts/refresh.py            # vault and repositories
uv run python scripts/refresh.py --vault    # notes only
```

To run it nightly, add a Windows Task Scheduler task (Docker Desktop must be
running, so schedule it for a time the host is signed in):

```
wsl.exe -d Ubuntu -- bash -lc "cd ~/code/broxworx/agent-swe && uv run python scripts/refresh.py"
```

## Development

```bash
uv sync --all-extras
uv run pytest                 # 73 tests, no network or containers needed
uv run ruff check src tests
```

The suite runs against a real in-memory Qdrant with a deterministic stub
embedder, so retrieval ordering is asserted exactly rather than depending on a
downloaded model. Tests that exercise the real ONNX models are opt-in, because
they need roughly 700MB of downloads:

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
├── docs/
│   ├── sources.py       presets, path containment, host allowlist, download
│   ├── loaders.py       HTML archive, EPUB, Markdown/GitHub, PDF → sections
│   ├── html.py          heading- and API-entry-aware HTML sectioning
│   ├── sections.py      section → chunk packing; code blocks kept whole
│   └── ingest.py        ingest jobs
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
