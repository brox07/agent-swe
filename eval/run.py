#!/usr/bin/env python
"""Retrieval quality against a labelled query set.

    uv run python eval/run.py                      # both suites, rerank on and off
    uv run python eval/run.py --suite docs         # documentation only
    uv run python eval/run.py --misses             # list what each suite got wrong

Labels are deliberately loose about position and strict about identity:

- A code query names the file and the qualified node path it should return.
  Line numbers drift with every edit, so labelling by span rots silently; the
  node path survives. Because chunking is nested and overlapping hits are
  collapsed, an ancestor or descendant of that node counts as the answer.
- A documentation query names the document and a section-trail fragment, both
  of which survive re-ingest. An empty ``section`` accepts any section of that
  document, for questions where the whole book is the answer.

Reported: hit@1, recall@5, MRR@10, and median latency. MRR is the headline —
it distinguishes "right answer first" from "right answer fourth", which
recall@5 cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

QUERIES = Path(__file__).with_name("queries.json")
DEPTH = 10


def code_hit(label: dict, result: dict) -> bool:
    """The labelled node, or the chunk that contains it.

    Chunking is nested and overlapping hits are collapsed, so a query about a
    method legitimately returns its enclosing class instead. An ancestor counts;
    so does a descendant, for a query naming the class.
    """
    if not result.get("file_path", "").endswith(label["file"]):
        return False
    want, got = label["symbol"], result.get("node_path", "")
    return got == want or want.startswith(got + ".") or got.startswith(want + ".")


def doc_hit(label: dict, result: dict) -> bool:
    if label["title"].lower() not in result.get("doc_title", "").lower():
        return False
    return label["section"].lower() in result.get("section", "").lower()


async def run_suite(
    session: ClientSession, suite: str, labels: list[dict], rerank: bool
) -> dict[str, Any]:
    tool = "search_codebase" if suite == "code" else "get_best_practices"
    argument = "query" if suite == "code" else "topic"
    matches = code_hit if suite == "code" else doc_hit

    ranks: list[int | None] = []
    latencies: list[float] = []
    for label in labels:
        started = time.perf_counter()
        response = await session.call_tool(
            tool, {argument: label["q"], "rerank": rerank, "limit": DEPTH}
        )
        latencies.append((time.perf_counter() - started) * 1000)
        results = (response.structured_content or {}).get("results", [])
        rank = next((i + 1 for i, r in enumerate(results) if matches(label, r)), None)
        ranks.append(rank)

    found = [r for r in ranks if r]
    return {
        "suite": suite,
        "rerank": rerank,
        "queries": len(labels),
        "hit@1": sum(1 for r in found if r == 1) / len(labels),
        "recall@5": sum(1 for r in found if r <= 5) / len(labels),
        "mrr@10": sum(1 / r for r in found) / len(labels),
        "median_ms": statistics.median(latencies),
        "misses": [label["q"] for label, rank in zip(labels, ranks, strict=True) if not rank],
        "ranks": ranks,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/mcp")
    parser.add_argument("--suite", choices=("code", "docs", "both"), default="both")
    parser.add_argument("--rerank", choices=("off", "on", "both"), default="both")
    parser.add_argument("--misses", action="store_true", help="print unanswered queries")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()

    labels = json.loads(QUERIES.read_text())
    suites = ["code", "docs"] if args.suite == "both" else [args.suite]
    modes = [False, True] if args.rerank == "both" else [args.rerank == "on"]

    reports = []
    async with streamable_http_client(args.url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for suite in suites:
                for rerank in modes:
                    reports.append(await run_suite(session, suite, labels[suite], rerank))

    header = (
        f"{'suite':6} {'rerank':7} {'n':>3} {'hit@1':>6} "
        f"{'recall@5':>9} {'MRR@10':>7} {'median':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r['suite']:6} {str(r['rerank']):7} {r['queries']:3} {r['hit@1']:6.2f} "
            f"{r['recall@5']:9.2f} {r['mrr@10']:7.3f} {r['median_ms']:7.0f}ms"
        )
    if args.misses:
        for r in reports:
            if r["misses"]:
                print(f"\nnot found in top {DEPTH} — {r['suite']}, rerank={r['rerank']}:")
                for query in r["misses"]:
                    print(f"  - {query}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
