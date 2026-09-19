# TODO

Status as of 2026-09-19. Priority order. Design rationale for anything here lives
in `docs/design/context-mcp-engine.md`.

## Milestone 1 — close out

- [ ] **Confirm Claude Code connects over the tailnet.** The stack is verified
      over localhost; the tailnet leg is not. Steps are in the README under
      *Connecting from another machine over Tailscale*. Tailscale is installed on
      the Windows host but was signed out as of this date.

## Milestone 2 — documentation ingestion

- [ ] `ingest_document` and `get_best_practices` over `best_practices_docs`
      (collection and table already exist).
- [ ] Sources: local EPUB and PDF books, and the official Python documentation.
- [ ] A labelled query set for docs retrieval, built before tuning anything.

## Retrieval quality

- [ ] **Build a real evaluation set.** The reranker choice rests on 12 queries
      against this repository, and no reranker beat fusion alone (design doc
      §9.1). 50+ labelled queries over a repository you actually work in would
      settle whether `rerank` earns its place, and would catch regressions
      from any future model change.
- [ ] Label results by line span, not by a substring in the returned content:
      content is capped at `MAX_RESULT_CHARS`, so a correct hit whose matching
      text sits past the cap is scored as a miss.

## Performance

- [ ] **First-index time.** 6m14s for 29 files / 234 chunks on a 6-core CPU.
      The dense model's 8k window is the cost. Worth measuring before indexing
      anything large: batch size, `INFERENCE_WORKERS`, and whether most chunks
      need anywhere near 8k tokens.
- [ ] Plain search is 42–169ms against a 10–15ms target. Profile where the time
      goes (query embedding vs Qdrant) before deciding whether it matters.

## Housekeeping

- [ ] `ruff format` would rewrite 13 files. Run it once in its own commit so
      later diffs stay readable.
- [ ] The model cache volume still holds `bge-reranker-base` (1.1GB) from before
      the switch. `docker compose down && docker volume rm agent-swe_model-cache`
      reclaims it at the cost of a ~700MB re-download.
