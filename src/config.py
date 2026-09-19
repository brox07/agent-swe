"""Runtime configuration, sourced from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Storage -------------------------------------------------------------
    postgres_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/context_engine"
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    codebase_collection: str = "codebase_index"
    docs_collection: str = "best_practices_docs"

    # --- Models --------------------------------------------------------------
    # Verified present in fastembed 0.8.0's supported-model lists.
    dense_model: str = "jinaai/jina-embeddings-v2-base-code"
    dense_dim: int = 768
    sparse_model: str = "Qdrant/bm25"
    # 22M parameters. bge-reranker-base (278M) measured 8-11s per query on a
    # 6-core CPU host with no quality gain over it; see design doc section 9.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    fastcache_dir: Path = Path("/app/.cache")

    # The dense model's context window. Nodes exceeding it are sub-split rather
    # than silently truncated, which is what the original spec's 512-token
    # model would have done to every large class.
    dense_max_tokens: int = 8192
    # Conservative characters-per-token ratio for code.
    chars_per_token: int = 3

    # --- Repositories --------------------------------------------------------
    # Repositories are bind-mounted read-only. Nothing is ever cloned, so no
    # git credential exists inside the container.
    repos_root: Path = Path("/app/repos")
    max_file_bytes: int = 1_048_576

    # --- Retrieval -----------------------------------------------------------
    default_limit: int = 5
    # Each hybrid branch pulls limit * this many candidates before fusion.
    prefetch_multiplier: int = 5
    # Rerank cost is linear in candidates. Fusion placed every correct hit in
    # its top 7 when measured, so 25 paid for candidates that never won.
    rerank_candidates: int = 10
    # Ceiling on content returned per hit, so one search cannot consume an
    # unbounded slice of the client's context window.
    max_result_chars: int = 1500

    # --- Documentation -------------------------------------------------------
    # ingest_document reads local files only beneath this root: accepting any
    # path from an MCP client would make the tool a file-read primitive.
    docs_root: Path = Path("/app/data")
    # And fetches only from these hosts (subdomains included), re-checked on every
    # redirect, so it cannot be pointed at the internal network.
    docs_allowed_hosts: str = (
        "docs.python.org,docs.pytest.org,docs.sqlalchemy.org,github.com,readthedocs.io"
    )
    # Docs chunks are prose-sized rather than code-unit-sized: ~500 tokens keeps
    # one idea per chunk, so a hit is an answer rather than a chapter.
    doc_chunk_chars: int = 2000
    # Code listings may run past the target up to this before they are split.
    doc_max_chunk_chars: int = 6000
    docs_default_limit: int = 3

    @property
    def docs_allowed_host_list(self) -> list[str]:
        return [h.strip().lower() for h in self.docs_allowed_hosts.split(",") if h.strip()]

    # --- Execution -----------------------------------------------------------
    # ONNX inference is blocking and would stall the event loop, including the
    # MCP session keepalive, if run inline.
    embed_batch_size: int = 32
    inference_workers: int = 2
    eager_model_load: bool = True

    # --- Service -------------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 - bound inside the container
    port: int = 8000
    # Unset by default: the tailnet is the trust boundary by explicit decision.
    # Setting this requires a bearer token on /mcp.
    mcp_auth_token: str | None = Field(default=None)
    log_level: str = "INFO"

    # The MCP SDK auto-enables DNS-rebinding protection allowing only localhost
    # Host headers, which would reject every client connecting over a Tailscale
    # IP with HTTP 421. Left empty, protection is disabled and a warning is
    # logged; set it (e.g. "100.90.1.5:*,context-engine:*") to re-enable with the
    # hosts clients actually use.
    allowed_hosts: str = ""
    allowed_origins: str = ""

    @property
    def allowed_host_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def allowed_origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_chunk_chars(self) -> int:
        return self.dense_max_tokens * self.chars_per_token


@lru_cache
def get_settings() -> Settings:
    return Settings()
