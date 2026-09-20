"""MCP tool registrations.

Note on the package name: the spec's layout put this at ``src/mcp/``, which
shadows the installed ``mcp`` SDK package the moment ``src/`` ends up on
sys.path. Renamed to ``src/mcp_server/`` to remove that footgun.

Code: ``search_codebase``, ``sync_repository``. Documentation:
``get_best_practices``, ``ingest_document``, ``list_doc_sources``. Both kinds of
background job report through ``get_sync_status``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.config import Settings
from src.docs.ingest import DocIngestService
from src.docs.sources import PRESETS, SOURCE_TYPES, SourceError
from src.ingest.git_sync import SyncError, SyncService
from src.vector.search import SearchService

logger = logging.getLogger(__name__)


def build_mcp_server(
    settings: Settings, search: SearchService, sync: SyncService, docs: DocIngestService
) -> MCPServer:
    mcp = MCPServer(
        name="context-mcp-engine",
        instructions=(
            "Semantic, AST-aware retrieval over indexed source code, plus indexed "
            "reference documentation and books. Code results carry exact file paths "
            "and line ranges; documentation results carry the section and location."
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
            Field(description="Filter by language: python, typescript, yaml, dockerfile."),
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
                    "False returns fused hybrid results (tens of ms). True rescores a "
                    "wider candidate set with a cross-encoder (~1s on CPU); use it when "
                    "the default top hits look wrong."
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
        description=(
            "Progress and outcome of a background job started by sync_repository or "
            "ingest_document."
        ),
    )
    async def get_sync_status(
        job_id: Annotated[
            str, Field(description="Job id returned by sync_repository or ingest_document.")
        ],
    ) -> dict[str, Any]:
        status = await sync.status(job_id)
        if status is None:
            return {"error": f"no such job: {job_id}"}
        return status

    @mcp.tool(
        name="get_best_practices",
        description=(
            "Search indexed documentation and books: language and library reference, "
            "tutorials, and guidance. Hybrid retrieval, so both questions ('how do I "
            "cancel a task group') and exact names ('asyncio.TaskGroup') work. Each "
            "result names its document, section, and location."
        ),
    )
    async def get_best_practices(
        topic: Annotated[str, Field(description="Question, concept, or API name.")],
        framework: Annotated[
            str | None,
            Field(
                description=(
                    "Restrict to one framework, e.g. python, fastapi, pydantic, "
                    "sqlalchemy, pytest. Call list_doc_sources to see what is indexed."
                )
            ),
        ] = None,
        version: Annotated[str | None, Field(description="Exact version tag, e.g. 3.14.")] = None,
        source_type: Annotated[
            str | None,
            Field(
                description=(
                    "Restrict by format: epub or pdf for books, html_archive or github "
                    "for reference documentation."
                )
            ),
        ] = None,
        rerank: Annotated[
            bool, Field(description="Rescore candidates with a cross-encoder (~1s on CPU).")
        ] = False,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=15)] = 3,
    ) -> dict[str, Any]:
        results = await search.search_docs(
            topic,
            framework=framework,
            version=version,
            source_type=source_type,
            rerank=rerank,
            limit=limit,
        )
        return {
            "topic": topic,
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

    @mcp.tool(
        name="ingest_document",
        description=(
            "Index documentation into the engine. Returns a job_id immediately; poll "
            "get_sync_status. Accepts a preset name ("
            + ", ".join(sorted(PRESETS))
            + "), a path under the documents root (a single EPUB/PDF/Markdown file, or "
            "a directory of books — EPUB is preferred where a title exists in both "
            "formats), or an https URL on an allowlisted host. Unchanged sources are "
            "skipped."
        ),
    )
    async def ingest_document(
        source_url: Annotated[
            str,
            Field(description="Preset name, path under the documents root, or https URL."),
        ],
        source_type: Annotated[
            str | None,
            Field(description="One of " + ", ".join(SOURCE_TYPES) + ". Inferred when omitted."),
        ] = None,
        framework: Annotated[
            str | None, Field(description="Framework tag for filtering, e.g. python.")
        ] = None,
        version: Annotated[str | None, Field(description="Version tag, e.g. 3.14.")] = None,
        title: Annotated[
            str | None, Field(description="Display title; books use their own metadata.")
        ] = None,
        force: Annotated[bool, Field(description="Re-index even if unchanged.")] = False,
    ) -> dict[str, Any]:
        try:
            return await docs.start(
                source_url,
                source_type=source_type,
                framework=framework,
                version=version,
                title=title,
                force=force,
            )
        except SourceError as exc:
            return {"error": str(exc)}

    @mcp.tool(
        name="forget_document",
        description=(
            "Remove an indexed documentation source: its vectors and its record. "
            "Takes a preset name, a path under the documents root, or the exact "
            "source_url shown by list_doc_sources. Re-ingesting replaces a source "
            "in place, so this is only for dropping one entirely."
        ),
    )
    async def forget_document(
        source_url: Annotated[
            str, Field(description="Preset name, path under the documents root, or source_url.")
        ],
    ) -> dict[str, Any]:
        return await docs.forget(source_url)

    @mcp.tool(
        name="list_doc_sources",
        description="Every indexed documentation source with its framework, version, and size.",
    )
    async def list_doc_sources() -> dict[str, Any]:
        sources = await docs.list_sources()
        return {"count": len(sources), "sources": sources, "presets": sorted(PRESETS)}

    return mcp
