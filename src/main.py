"""FastAPI bootstrap and MCP Streamable HTTP mount.

Transport note: the spec specified HTTP+SSE at /sse, which the MCP specification
superseded with Streamable HTTP in its 2025-03-26 revision. This serves
Streamable HTTP at /mcp. The SSE app remains available in the SDK should a legacy
client ever need it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings

from src.config import Settings, get_settings
from src.db.postgres import check_connection, dispose_engine, init_engine
from src.ingest.git_sync import SyncService
from src.mcp_server.tools import build_mcp_server
from src.vector.embedder import FastEmbedder
from src.vector.qdrant import QdrantStore
from src.vector.search import SearchService

logger = logging.getLogger(__name__)


class AppState:
    """Container for the long-lived objects, so tests can build one directly."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = QdrantStore(settings)
        self.embedder = FastEmbedder(settings)
        self.search = SearchService(settings, self.store, self.embedder)
        self.sync = SyncService(settings, self.store, self.embedder)
        self.models_ready = False


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application.

    ``state`` is injectable so the test suite can supply an in-memory Qdrant and
    a stub embedder without reaching for the network.
    """
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    state = state or AppState(settings)
    mcp = build_mcp_server(settings, state.search, state.sync)

    # The SDK auto-enables DNS-rebinding protection when it sees a localhost bind
    # address, permitting only localhost Host headers. Clients reach this service
    # on a Tailscale IP, which would then be rejected with HTTP 421, so the policy
    # is always passed explicitly rather than inferred.
    if settings.allowed_host_list:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.allowed_host_list,
            allowed_origins=settings.allowed_origin_list,
        )
    else:
        logger.warning(
            "DNS-rebinding protection disabled: ALLOWED_HOSTS is unset, so any Host "
            "header is accepted. Set ALLOWED_HOSTS to the host:port clients use."
        )
        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    # streamable_http_app() must be called before session_manager is reachable:
    # the manager is created lazily by that call.
    mcp_app = mcp.streamable_http_app(
        host=settings.host, transport_security=transport_security
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        init_engine(settings.postgres_url)
        await state.store.ensure_collections()
        if settings.eager_model_load:
            # Warm before serving, so the first real query is not paying for a
            # ~1GB model download.
            logger.info("warming embedding models")
            await state.embedder.warmup()
        state.models_ready = True
        logger.info("mcp streamable http ready on /mcp")
        async with mcp.session_manager.run():
            yield
        await state.embedder.aclose()
        await state.store.aclose()
        await dispose_engine()

    app = FastAPI(title="Context MCP Engine", version="0.1.0", lifespan=lifespan)
    app.state.engine_state = state

    @app.get("/health")
    async def health() -> JSONResponse:
        """Readiness, not just liveness: compose gates on models being loaded."""
        qdrant_ok = await state.store.healthy()
        postgres_ok = await check_connection()
        ready = qdrant_ok and postgres_ok and state.models_ready
        body = {
            "status": "ready" if ready else "degraded",
            "qdrant": qdrant_ok,
            "postgres": postgres_ok,
            "models_loaded": state.models_ready,
            "dense_model": settings.dense_model,
            "transport": "streamable-http",
            "mcp_endpoint": "/mcp",
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "context-mcp-engine", "mcp_endpoint": "/mcp"}

    if settings.mcp_auth_token:
        # Off by default: the tailnet is the trust boundary by explicit decision.
        # Setting MCP_AUTH_TOKEN turns this on without any other change.
        @app.middleware("http")
        async def require_bearer(request: Request, call_next):
            if request.url.path.startswith("/mcp"):
                header = request.headers.get("authorization", "")
                expected = f"Bearer {settings.mcp_auth_token}"
                if header != expected:
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    # The MCP app already routes /mcp internally, so it mounts at the root. The
    # /health and / routes are registered first and therefore match first.
    app.mount("/", mcp_app)
    return app


# Deliberately no module-level `app = create_app()`: that would build a Qdrant
# client (and attempt a version handshake) on mere import. The service is served
# with `uvicorn src.main:create_app --factory`.
