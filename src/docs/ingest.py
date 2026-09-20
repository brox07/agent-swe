"""Documentation ingestion as an asynchronous job.

Python's documentation alone is ~18,000 chunks and over half an hour of CPU
embedding, so ``ingest_document`` returns a job id at once and progress is read
through ``get_sync_status``, exactly like a repository sync.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select

from src.config import Settings
from src.db.models import DocSource, SyncJob, SyncStatus
from src.db.postgres import session_scope
from src.docs import loaders
from src.docs.sections import chunk_sections
from src.docs.sources import (
    PRESETS,
    DocTarget,
    SourceError,
    fetch,
    github_docs_path,
    resolve_target,
)
from src.errors import describe
from src.vector.embedder import Embedder
from src.vector.qdrant import QdrantStore

logger = logging.getLogger(__name__)

# Chunks per embed-and-upsert round: large enough to batch inference, small
# enough that progress moves visibly and a failure loses little work.
EMBED_BATCH = 64

# One ingest at a time. Embedding saturates the CPU; two in parallel would each
# take twice as long and starve search.
_ingest_lock = asyncio.Lock()
_running: set[asyncio.Task] = set()


def vault_digest(root, exclude: Sequence[str] = ()) -> str:
    """One hash over every note's path and content, so an unchanged vault skips."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part in loaders.VAULT_SKIP_DIRS for part in rel.parts) or loaders.excluded(
            rel, exclude
        ):
            continue
        digest.update(rel.as_posix().encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def load(
    target: DocTarget, data: bytes, exclude: Sequence[str] = ()
) -> loaders.LoadedDoc:
    if target.source_type == "html_archive":
        return loaders.html_archive(data, target.title)
    if target.source_type == "github":
        return loaders.github_markdown(data, github_docs_path(target), target.title, target.exclude)
    if target.source_type == "epub":
        return loaders.epub(data, target.title)
    if target.source_type == "pdf":
        return loaders.pdf(data, target.title)
    if target.source_type == "vault":
        assert target.local_path is not None
        return loaders.vault(target.local_path, target.title, exclude)
    if target.source_type == "markdown":
        page = target.local_path.name if target.local_path else target.source_url
        text = data.decode("utf-8", errors="replace")
        return loaders.LoadedDoc(target.title, loaders.markdown_sections(text, page))
    raise SourceError(f"unsupported source_type {target.source_type!r}")


class DocIngestService:
    def __init__(self, settings: Settings, store: QdrantStore, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder

    async def start(
        self,
        source: str,
        *,
        source_type: str | None = None,
        framework: str | None = None,
        version: str | None = None,
        title: str | None = None,
        force: bool = False,
    ) -> dict:
        """Validate synchronously, so a bad path or host fails the call, not the job."""
        targets = resolve_target(self._settings, source, source_type, framework, version, title)
        job_id = str(uuid.uuid4())
        label = targets[0].title if len(targets) == 1 else f"{len(targets)} documents"
        async with session_scope() as session:
            session.add(
                SyncJob(
                    id=job_id,
                    repo_name=f"docs: {label}"[:255],
                    status=SyncStatus.PENDING,
                    phase="queued",
                    files_total=len(targets),
                )
            )
        task = asyncio.create_task(self._run(job_id, targets, force))
        _running.add(task)
        task.add_done_callback(_running.discard)
        return {
            "job_id": job_id,
            "documents": [t.title for t in targets],
            "status": SyncStatus.PENDING.value,
            "hint": "Poll get_sync_status with this job_id.",
        }

    async def forget(self, source: str) -> dict:
        """Remove one indexed source: its vectors and its row.

        Re-ingesting replaces a source in place, so this exists for the other
        case — a source that should no longer be searchable at all, which
        otherwise could only be removed by editing the database by hand.
        """
        candidates = {source}
        preset = PRESETS.get(source.lower())
        if preset is not None:
            candidates.add(preset.source_url)
        if not source.startswith(("http://", "https://", "file://")):
            candidates.add(f"file://{source.lstrip('/')}")

        async with session_scope() as session:
            rows = (
                (
                    await session.execute(
                        select(DocSource).where(DocSource.source_url.in_(candidates))
                    )
                )
                .scalars()
                .all()
            )
            found = [(r.source_url, r.title, r.chunk_count) for r in rows]
            for row in rows:
                await session.delete(row)

        if not found:
            return {"error": f"no indexed source matches {source!r}", "removed": []}
        for source_url, _, _ in found:
            await self._store.delete_doc_source(source_url)
        return {
            "removed": [
                {"title": title, "source_url": url, "chunks": count} for url, title, count in found
            ]
        }

    async def list_sources(self) -> list[dict]:
        async with session_scope() as session:
            rows = (await session.execute(select(DocSource).order_by(DocSource.id))).scalars()
            return [
                {
                    "title": r.title,
                    "source_url": r.source_url,
                    "source_type": r.source_type,
                    "framework": r.framework,
                    "version": r.version_tag,
                    "chunks": r.chunk_count,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]

    # --- worker --------------------------------------------------------------

    async def _run(self, job_id: str, targets: list[DocTarget], force: bool) -> None:
        async with _ingest_lock:
            await self._update(job_id, status=SyncStatus.RUNNING)
            done = skipped = upserted = 0
            failures: list[str] = []
            for target in targets:
                try:
                    written = await self._ingest_one(job_id, target, force)
                except Exception as exc:  # noqa: BLE001 - one bad book must not sink the rest
                    logger.exception("ingest of %s failed", target.source_url)
                    failures.append(f"{target.title}: {describe(exc)}")
                    continue
                if written is None:
                    skipped += 1
                else:
                    upserted += written
                done += 1
                await self._update(
                    job_id, files_done=done, files_skipped=skipped, chunks_upserted=upserted
                )
            status = SyncStatus.FAILED if failures and not done else SyncStatus.SUCCEEDED
            await self._update(
                job_id,
                status=status,
                phase="done" if status is SyncStatus.SUCCEEDED else "failed",
                error="; ".join(failures)[:4000] or None,
                finished_at=datetime.now(UTC),
            )

    async def _ingest_one(self, job_id: str, target: DocTarget, force: bool) -> int | None:
        """Returns chunks written, or None when the source was unchanged."""
        name = target.title[:40]
        await self._update(job_id, phase=f"fetching {name}")
        if target.source_type == "vault":
            assert target.local_path is not None
            data = b""  # the loader reads the tree itself
            digest = await asyncio.to_thread(
                vault_digest, target.local_path, self._settings.vault_exclude_list
            )
        else:
            data = await fetch(self._settings, target)
            digest = hashlib.sha256(data).hexdigest()

        async with session_scope() as session:
            existing = (
                await session.execute(
                    select(DocSource).where(DocSource.source_url == target.source_url)
                )
            ).scalar_one_or_none()
        if (
            not force
            and existing is not None
            and existing.content_hash == digest
            and await self._store.count_docs(target.source_url) == existing.chunk_count
        ):
            return None

        await self._update(job_id, phase=f"parsing {name}")
        # Parsing Python's archive takes about a minute of CPU; off the loop.
        loaded = await asyncio.to_thread(
            load, target, data, self._settings.vault_exclude_list
        )
        title = loaded.title if target.source_type in ("epub", "pdf") else target.title
        chunks = chunk_sections(
            loaded.sections,
            target_chars=self._settings.doc_chunk_chars,
            max_chars=self._settings.doc_max_chunk_chars,
        )

        # Clear first: a new edition that yields fewer chunks must not leave the
        # old edition's tail behind.
        await self._store.delete_doc_source(target.source_url)
        written = 0
        # Length order keeps each inference batch uniform (see embed_documents);
        # point ids come from chunk_index, so storage order is unaffected.
        by_length = sorted(chunks, key=lambda c: len(c.content))
        for start in range(0, len(by_length), EMBED_BATCH):
            batch = by_length[start : start + EMBED_BATCH]
            dense, sparse = await self._embedder.embed_documents(
                [c.embed_text(title) for c in batch]
            )
            written += await self._store.upsert_doc_chunks(
                source_url=target.source_url,
                source_type=target.source_type,
                doc_title=title,
                framework=target.framework,
                version=target.version,
                link_base=target.link_base,
                chunks=batch,
                dense=dense,
                sparse=sparse,
            )
            await self._update(job_id, phase=f"embedding {name} {written}/{len(chunks)}"[:64])

        async with session_scope() as session:
            row = (
                await session.execute(
                    select(DocSource).where(DocSource.source_url == target.source_url)
                )
            ).scalar_one_or_none()
            if row is None:
                row = DocSource(source_url=target.source_url, source_type=target.source_type)
                session.add(row)
            row.source_type = target.source_type
            row.title = title
            row.framework = target.framework
            row.version_tag = target.version
            row.content_hash = digest
            row.chunk_count = written
        return written

    async def _update(self, job_id: str, **fields) -> None:
        async with session_scope() as session:
            job = await session.get(SyncJob, job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
