# TODO

Status as of 2026-09-19. Priority order. Design rationale for anything here lives
in `docs/design/context-mcp-engine.md`.

## Milestone 1 — close out

- [ ] **Confirm Claude Code connects over the tailnet.** The stack is verified
      over localhost; the tailnet leg is not. Steps are in the README under
      *Connecting from another machine over Tailscale*. As of this date the
      Windows Tailscale service was stuck in "starting" (`NoState`) with the
      network itself fine; restarting the service is the next step.

## Milestone 2 — documentation

- [x] `ingest_document`, `get_best_practices`, `list_doc_sources`.
- [x] Loaders: Sphinx HTML archives, EPUB, PDF (bookmarks), Markdown from GitHub.
- [x] Presets: Python 3.14, FastAPI, Pydantic, SQLAlchemy 2.0, pytest.
- [ ] **First full ingest** of the 19 books and 5 presets — re-queued
      2026-09-19 after the EPUB front/back-matter fix.
      Check `list_doc_sources` and the job statuses for failures.
- [ ] **A labelled query set for docs retrieval**, before tuning anything:
      chunk size, whether `rerank` helps here, whether books crowd out the
      reference docs for API questions.
- [ ] PDF-only books lose code formatting (text extraction flattens layout).
      Only one book is PDF-only today; revisit if more arrive.
- [ ] pytest's docs come from `stable`, so the version tag is `stable`, not a
      number. Pin to a versioned download if the version filter matters.

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

- [ ] **Ingest throughput.** ~1.1k chars/s measured on a code-heavy book
      against 3.6k for uniform prose; length-sorted batching recovered 1.5x.
      Code tokenizes denser than prose, so some of the gap is real work. A
      smaller or quantized dense model for docs is the next lever — but it means
      a separate collection with its own vector size.

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
