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

- [x] **A labelled evaluation set** — `eval/`, 34 code and 33 documentation
      queries, with a baseline in `eval/README.md`. Labels are qualified node
      paths, not spans or content substrings, both of which rot or mislead.
- [ ] **Run the documentation suite** once the first ingest finishes, and
      decide from it whether `rerank` helps on prose even though it is a wash
      on code.
- [ ] Extend the code suite to a second repository. Everything here is labelled
      against the engine itself, which its own docstrings describe well; a
      repository with worse comments is the harder test.

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

- [ ] **Serialize embedding work across job types.** Doc ingests take a global
      lock, but repository syncs take only a per-repo one, so N syncs run at
      once. They share a 2-worker inference pool, so nothing goes faster — but
      each running job holds its own chunks and batches, and engine memory went
      from 1.5GB to 7.2GB with six syncs and one ingest in flight (2026-09-19).
      A single semaphore around embedding would bound that.

## Housekeeping

- [x] `ruff format` over the tree, in its own commit.
- [ ] The model cache volume still holds `bge-reranker-base` (1.1GB) from before
      the switch. `docker compose down && docker volume rm agent-swe_model-cache`
      reclaims it at the cost of a ~700MB re-download.
