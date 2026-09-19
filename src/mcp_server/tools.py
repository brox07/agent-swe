"""MCP tool registrations.

Note on the package name: the spec's layout put this at ``src/mcp/``, which
shadows the installed ``mcp`` SDK package the moment ``src/`` ends up on
sys.path. Renamed to ``src/mcp_server/`` to remove that footgun.

Milestone 1 exposes the code path only. ``get_best_practices`` and
``ingest_document`` arrive with milestone 2.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.config import Settings
from src.ingest.git_sync import SyncError, SyncService
from src.vector.search import SearchService

logger = logging.getLogger(__name__)


def build_mcp_server(
    settings: Settings, search: SearchService, sync: SyncService
) -> MCPServer:
    mcp = MCPServer(
        name="context-mcp-engine",
        instructions=(
            "Semantic, AST-aware retrieval over indexed source code. Results carry "
            "exact file paths and line ranges; open the file directly when you need "
            "more than the returned excerpt."
        ),
    )

    @mcp.tool(
        name="search_codebase",
        description=(
            "Semantic and lexical search over indexed source code. Hybrid retrieval "
            "fuses dense vector similarity with BM25, so both conceptual queries "
            "('how are tokens refreshed') and exact identifiers ('validate_token') "
            "work. Every hit includes file_path with start_line and end_line."
        ),
    )
    async def search_codebase(
        query: Annotated[str, Field(description="Semantic query or symbol name.")],
        language: Annotated[
            str | None,
            Field(
                description="Filter by language: python, typescript, yaml, dockerfile."
            ),
        ] = None,
        repo_name: Annotated[
            str | None, Field(description="Restrict the search to one repository.")
        ] = None,
        node_type: Annotated[
            str | None,
            Field(description="Filter by kind: function, class, method, interface, block."),
        ] = None,
        rerank: Annotated[
            bool,
            Field(
                description=(
                    "False returns fused hybrid results (~15ms). True scores a wider "
                    "candidate set with a cross-encoder for better precision (~60ms)."
                )
            ),
        ] = False,
        limit: Annotated[int, Field(description="Maximum hits to return.", ge=1, le=25)] = 5,
    ) -> dict[str, Any]:
        results = await search.search_codebase(
            query,
            language=language,
            repo_name=repo_name,
            node_type=node_type,
            rerank=rerank,
            limit=limit,
        )
        return {
            "query": query,
            "mode": "rerank" if rerank else "hybrid-rrf",
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

    @mcp.tool(
        name="sync_repository",
        description=(
            "Index or re-index a bind-mounted repository. Returns immediately with a "
            "job_id because indexing outlasts a tool-call timeout; poll get_sync_status. "
            "Only files whose content hash changed are re-parsed, and files removed from "
            "the working tree have their vectors deleted."
        ),
    )
    async def sync_repository(
        repo: Annotated[
            str,
            Field(
                description=(
                    "Registered repository name, or a path beneath the mounted "
                    "repository root. Repositories are mounted, never cloned."
                )
            ),
        ],
        force: Annotated[
            bool, Field(description="Re-index every file, ignoring unchanged hashes.")
        ] = False,
    ) -> dict[str, Any]:
        try:
            return await sync.start(repo, force=force)
        except SyncError as exc:
            # A caller-fixable mistake, not a server fault.
            return {"error": str(exc)}

    @mcp.tool(
        name="get_sync_status",
        description="Progress and outcome of a sync job started by sync_repository.",
    )
    async def get_sync_status(
        job_id: Annotated[str, Field(description="Job id returned by sync_repository.")],
    ) -> dict[str, Any]:
        status = await sync.status(job_id)
        if status is None:
            return {"error": f"no such job: {job_id}"}
        return status

    return mcp
