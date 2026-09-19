"""End-to-end MCP protocol checks over the mounted Streamable HTTP transport.

Covers the transport decision itself: that /mcp speaks Streamable HTTP, that the
three milestone-1 tools are advertised with usable schemas, and that a real
tools/call round trip returns indexed results. The Host-header policy is checked
too, because the SDK's default would have rejected every Tailscale client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.db.models import Base
from src.db.postgres import dispose_engine, init_engine
from src.ingest.git_sync import SyncService
from src.main import AppState, create_app
from src.vector.qdrant import QdrantStore
from src.vector.search import SearchService

from .conftest import StubEmbedder

PROTOCOL_VERSION = "2025-06-18"
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _parse(response: httpx.Response) -> dict:
    """Streamable HTTP may answer as JSON or as a single SSE event."""
    body = response.text
    if body.lstrip().startswith("{"):
        return json.loads(body)
    for line in body.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise AssertionError(f"no JSON-RPC payload in response: {body!r}")


class Client:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http
        self._session_id: str | None = None

    async def initialize(self) -> dict:
        response = await self._http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
            headers=HEADERS,
        )
        assert response.status_code == 200, response.text
        self._session_id = response.headers.get("mcp-session-id")
        await self._http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._headers(),
        )
        return _parse(response)

    def _headers(self) -> dict[str, str]:
        headers = dict(HEADERS)
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    async def call(self, method: str, params: dict | None = None) -> dict:
        response = await self._http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 99, "method": method, "params": params or {}},
            headers=self._headers(),
        )
        assert response.status_code == 200, response.text
        return _parse(response)


@pytest_asyncio.fixture
async def app_state(settings: Settings, embedder: StubEmbedder) -> AsyncIterator[AppState]:
    state = AppState.__new__(AppState)
    state.settings = settings
    state.store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    state.embedder = embedder
    state.search = SearchService(settings, state.store, embedder)
    state.sync = SyncService(settings, state.store, embedder)
    state.models_ready = False

    engine = init_engine(settings.postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield state
    await dispose_engine()


@pytest_asyncio.fixture
async def client(settings: Settings, app_state: AppState) -> AsyncIterator[Client]:
    app = create_app(settings, state=app_state)
    async with LifespanManager(app) as managed:
        transport = httpx.ASGITransport(app=managed.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8000"
        ) as http:
            yield Client(http)


class TestTransport:
    async def test_initialize_negotiates_streamable_http(self, client: Client):
        result = await client.initialize()
        assert result["result"]["serverInfo"]["name"] == "context-mcp-engine"
        assert result["result"]["protocolVersion"]

    async def test_any_host_is_accepted_when_allowed_hosts_is_unset(
        self, settings: Settings, app_state: AppState
    ):
        """The SDK's localhost-only default would 421 every Tailscale client."""
        app = create_app(settings, state=app_state)
        async with LifespanManager(app) as managed:
            transport = httpx.ASGITransport(app=managed.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://100.90.1.5:8000"
            ) as http:
                response = await http.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {"name": "tailnet", "version": "0"},
                        },
                    },
                    headers=HEADERS,
                )
                assert response.status_code == 200, response.text

    async def test_configured_allowed_hosts_rejects_others(
        self, settings: Settings, app_state: AppState
    ):
        settings.allowed_hosts = "100.90.1.5:*"
        app = create_app(settings, state=app_state)
        async with LifespanManager(app) as managed:
            transport = httpx.ASGITransport(app=managed.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://evil.example:8000"
            ) as http:
                response = await http.post("/mcp", json={}, headers=HEADERS)
                assert response.status_code == 421


class TestToolSurface:
    async def test_all_three_milestone_tools_are_advertised(self, client: Client):
        await client.initialize()
        listed = await client.call("tools/list")
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert names == {"search_codebase", "sync_repository", "get_sync_status"}

    async def test_search_schema_exposes_the_documented_arguments(self, client: Client):
        await client.initialize()
        listed = await client.call("tools/list")
        tool = next(t for t in listed["result"]["tools"] if t["name"] == "search_codebase")
        properties = tool["inputSchema"]["properties"]
        assert {"query", "language", "repo_name", "node_type", "rerank", "limit"} <= set(
            properties
        )
        assert tool["inputSchema"]["required"] == ["query"]


class TestToolCalls:
    async def test_sync_then_search_round_trip(self, client: Client, git_repo):
        """The milestone-1 path a client actually exercises."""
        await client.initialize()

        started = await client.call(
            "tools/call", {"name": "sync_repository", "arguments": {"repo": "demo"}}
        )
        payload = json.loads(started["result"]["content"][0]["text"])
        job_id = payload["job_id"]
        assert job_id

        for _ in range(200):
            status = await client.call(
                "tools/call", {"name": "get_sync_status", "arguments": {"job_id": job_id}}
            )
            state = json.loads(status["result"]["content"][0]["text"])
            if state["status"] in ("succeeded", "failed"):
                break
        assert state["status"] == "succeeded", state

        found = await client.call(
            "tools/call",
            {"name": "search_codebase", "arguments": {"query": "validate_token", "limit": 3}},
        )
        results = json.loads(found["result"]["content"][0]["text"])
        assert results["count"] > 0
        top = results["results"][0]
        assert top["file_path"] == "src/auth.py"
        assert top["start_line"] > 0
        assert "validate_token" in top["node_path"]

    async def test_unknown_job_reports_an_error_payload(self, client: Client):
        await client.initialize()
        response = await client.call(
            "tools/call", {"name": "get_sync_status", "arguments": {"job_id": "nope"}}
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        assert "error" in payload

    async def test_repo_outside_the_mount_root_is_refused_not_crashed(
        self, client: Client, git_repo
    ):
        await client.initialize()
        response = await client.call(
            "tools/call", {"name": "sync_repository", "arguments": {"repo": "../../etc"}}
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        assert "outside" in payload["error"]


class TestHealth:
    async def test_health_reports_readiness_detail(self, client: Client):
        response = await client._http.get("/health")
        body = response.json()
        assert body["transport"] == "streamable-http"
        assert body["mcp_endpoint"] == "/mcp"
        assert "qdrant" in body and "postgres" in body

    async def test_root_advertises_the_endpoint(self, client: Client):
        response = await client._http.get("/")
        assert response.json()["mcp_endpoint"] == "/mcp"


@pytest.mark.usefixtures("git_repo")
class TestAuthToken:
    async def test_bearer_token_is_enforced_when_configured(
        self, settings: Settings, app_state: AppState
    ):
        settings.mcp_auth_token = "s3cret"
        app = create_app(settings, state=app_state)
        async with LifespanManager(app) as managed:
            transport = httpx.ASGITransport(app=managed.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://localhost:8000"
            ) as http:
                unauthorised = await http.post("/mcp", json={}, headers=HEADERS)
                assert unauthorised.status_code == 401
                # /health stays open so compose can probe it.
                assert (await http.get("/health")).status_code in (200, 503)
