#!/usr/bin/env python
"""Bring the index up to date: the vault, then every mounted repository.

    uv run python scripts/refresh.py            # everything
    uv run python scripts/refresh.py --vault    # notes only

Everything is content-hashed, so an unchanged vault or repository costs a scan
and no embedding. Jobs run one at a time on purpose: embedding is serialized in
the engine anyway, and a queue of parallel requests only makes the log harder to
read.

Intended for a scheduler. From Windows Task Scheduler, daily:

    wsl.exe -d Ubuntu -- bash -lc "cd ~/code/broxworx/agent-swe && uv run python scripts/refresh.py"

Exits non-zero if any job failed, so a scheduler can surface it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent.parent
POLL_SECONDS = 10


def env(name: str, default: str = "") -> str:
    """Prefer the real environment, fall back to the .env beside the compose file."""
    if name in os.environ:
        return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and not line.lstrip().startswith("#"):
                return value.strip()
    return default


def repositories() -> list[str]:
    """Directory names under the mounted repository root that are git checkouts."""
    root = Path(env("REPOS_HOST_PATH", str(ROOT / "repos")))
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / ".git").exists())


def result_of(response) -> dict:
    if response.structured_content:
        return response.structured_content
    return json.loads(response.content[0].text)


async def run_job(session: ClientSession, tool: str, arguments: dict, label: str) -> bool:
    started = result_of(await session.call_tool(tool, arguments))
    if "error" in started:
        print(f"  {label}: refused — {started['error']}")
        return False
    job_id = started["job_id"]
    while True:
        status = result_of(await session.call_tool("get_sync_status", {"job_id": job_id}))
        if status["status"] in ("succeeded", "failed"):
            break
        await asyncio.sleep(POLL_SECONDS)
    ok = status["status"] == "succeeded"
    detail = f"{status['chunks_upserted']} chunks"
    if status["files_skipped"]:
        detail += f", {status['files_skipped']} unchanged"
    if status["error"]:
        detail += f" — {status['error']}"
    print(f"  {label}: {status['status']} ({detail})")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=env("ENGINE_URL", "http://localhost:8000/mcp"))
    parser.add_argument("--vault", action="store_true", help="notes only")
    parser.add_argument("--repos", action="store_true", help="repositories only")
    args = parser.parse_args()

    token = env("MCP_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    failures = 0

    async with httpx2.AsyncClient(headers=headers, timeout=120) as http:
        async with streamable_http_client(args.url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if not args.repos:
                    print("vault:")
                    arguments = {"source_url": "vault"}
                    if env("VAULT_TITLE"):
                        arguments["title"] = env("VAULT_TITLE")
                    if not await run_job(session, "ingest_document", arguments, "notes"):
                        failures += 1
                if not args.vault:
                    print("repositories:")
                    for name in repositories():
                        if not await run_job(
                            session, "sync_repository", {"repo": name}, name
                        ):
                            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
