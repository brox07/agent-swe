# Retrieval evaluation

```bash
uv run python eval/run.py                 # both suites, rerank on and off
uv run python eval/run.py --suite docs --misses
```

Needs a running engine (`--url` to point elsewhere) with the repository and the
documentation already indexed. `queries.json` holds the labels; see `run.py` for
what counts as a hit and why.

Baseline, 2026-09-19. Code: 34 queries, 10 repositories indexed. Docs: 33
queries, 24 sources (19 books, 5 reference sets, 42k chunks).

| suite | rerank | hit@1 | recall@5 | MRR@10 | median |
|-------|--------|-------|----------|--------|--------|
| code  | off    | 0.59  | 0.85     | 0.718  | 47ms   |
| code  | on     | 0.62  | 0.85     | 0.728  | 757ms  |
| docs  | off    | 0.64  | 0.88     | 0.759  | 61ms   |
| docs  | on     | 0.64  | 0.85     | 0.743  | 915ms  |

Reranking is a wash on both suites. Note the code numbers fell from MRR 0.749
to 0.718 when nine more repositories were indexed: the same queries now compete
against nine other codebases, which is the honest cost of a shared index.

Earlier baseline, 34 code queries against this repository alone:

| rerank | hit@1 | recall@5 | MRR@10 | median |
|--------|-------|----------|--------|--------|
| off    | 0.62  | 0.91     | 0.749  | 114ms  |
| on     | 0.62  | 0.91     | 0.746  | 1522ms |

Reranking buys nothing here for 13x the latency, which is why it is off by
default. Two earlier readings were wrong for label reasons worth remembering:
matching a label by substring of the returned content scored correct hits as
misses when the match fell past the display cap, and requiring the exact node
counted the enclosing class — a legitimate answer under nested chunking — as a
miss.
