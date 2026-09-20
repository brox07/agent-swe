"""Incremental indexing of bind-mounted working trees.

Repositories are mounted read-only; nothing is cloned, so no git credential ever
exists inside the container. Change detection hashes the *working tree* rather
than diffing commits, which means uncommitted edits are indexed too — desirable
for a live development loop, and the reason ``commit_sha`` is recorded alongside
a ``dirty`` flag instead of being treated as identifying the indexed content.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, Repo
from sqlalchemy import delete, select, update

from src.config import Settings
from src.db.models import IndexedFile, Repository, SyncJob, SyncStatus
from src.db.postgres import session_scope
from src.parser.base import CodeChunk, detect_language
from src.parser.generic_chunker import parse_generic
from src.parser.tree_sitter_ast import parse_code
from src.vector.embedder import Embedder
from src.vector.qdrant import QdrantStore

logger = logging.getLogger(__name__)

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "site-packages",
        "vendor",
        "coverage",
        "htmlcov",
    }
)

# Background tasks need a strong reference or the event loop may collect them
# mid-flight.
_running: set[asyncio.Task] = set()
_repo_locks: dict[str, asyncio.Lock] = {}


class SyncError(Exception):
    """Raised for caller-fixable problems: unknown repo, path outside the mount root."""


@dataclass(slots=True)
class RepoTarget:
    name: str
    path: Path
    head_sha: str | None
    dirty: bool
    origin_url: str | None


def resolve_repo(settings: Settings, repo: str) -> RepoTarget:
    """Resolve a repo name or mounted path, refusing anything outside the mount root.

    Accepting a bare name keeps the common call short; the containment check is
    what stops ``../../etc`` from being handed to the indexer.
    """
    root = settings.repos_root.resolve()
    candidate = Path(repo)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()

    if target != root and not target.is_relative_to(root):
        raise SyncError(f"{repo!r} resolves outside the mounted repository root {root}")
    if not target.is_dir():
        raise SyncError(f"{repo!r} is not a directory at {target}")

    head_sha: str | None = None
    dirty = False
    origin_url: str | None = None
    try:
        git_repo = open_repo(target)
        head_sha = git_repo.head.commit.hexsha if git_repo.head.is_valid() else None
        dirty = git_repo.is_dirty(untracked_files=False)
        if "origin" in {r.name for r in git_repo.remotes}:
            origin_url = next(iter(git_repo.remote("origin").urls), None)
    except InvalidGitRepositoryError:
        logger.info("%s is not a git repository; indexing as a plain directory", target)
    except GitCommandError as exc:
        raise SyncError(f"git failed reading {target}: {exc.stderr.strip() or exc}") from exc

    return RepoTarget(
        name=target.name, path=target, head_sha=head_sha, dirty=dirty, origin_url=origin_url
    )


def open_repo(path: Path) -> Repo:
    """Open a mounted repository, trusting it despite its foreign owner.

    The engine runs as root while bind-mounted checkouts belong to the host user,
    and git refuses to touch a repository owned by someone else. The exception is
    scoped to this one path and passed as command-line config, so the container's
    global git config is never loosened.
    """
    repo = Repo(path)
    repo.git.update_environment(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="safe.directory",
        GIT_CONFIG_VALUE_0=str(path),
    )
    return repo


def list_source_files(target: RepoTarget) -> list[Path]:
    """Indexable files in the working tree, relative to the repo root.

    In a git repository the file list comes from git itself, so ``.gitignore`` is
    honoured for free rather than reimplemented.
    """
    paths: set[Path] = set()
    try:
        git_repo = open_repo(target.path)
    except InvalidGitRepositoryError:
        paths = {path.relative_to(target.path) for path in target.path.rglob("*") if path.is_file()}
    else:
        # A git failure propagates rather than falling back to a walk: the walk
        # ignores .gitignore, so it would silently index everything git excludes.
        tracked = git_repo.git.ls_files().splitlines()
        untracked = git_repo.git.ls_files("--others", "--exclude-standard").splitlines()
        paths = {Path(rel) for rel in (*tracked, *untracked) if rel}

    keep: list[Path] = []
    for rel in sorted(paths):
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if ".min." in rel.name:
            continue
        if detect_language(rel) is None:
            continue
        # `git ls-files` still reports a tracked file that has been deleted from
        # the working tree but not yet committed. Indexing works from the tree,
        # so such a file must be treated as gone and its vectors pruned.
        if not (target.path / rel).is_file():
            continue
        keep.append(rel)
    return keep


def chunk_source(text: str, language: str, rel_path: str, max_chunk_chars: int) -> list[CodeChunk]:
    if language in ("python", "typescript"):
        return parse_code(text, language, rel_path, max_chunk_chars)
    return parse_generic(text, language, max_chunk_chars)


class SyncService:
    def __init__(self, settings: Settings, store: QdrantStore, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder

    async def start(self, repo: str, force: bool = False) -> dict:
        """Register the job and return immediately.

        Indexing a real repository outlasts any MCP client timeout, so the tool
        hands back a job id and the caller polls ``get_sync_status``.
        """
        target = resolve_repo(self._settings, repo)
        job_id = str(uuid.uuid4())

        async with session_scope() as session:
            session.add(
                SyncJob(
                    id=job_id,
                    repo_name=target.name,
                    status=SyncStatus.PENDING,
                    phase="queued",
                )
            )

        task = asyncio.create_task(self._run(job_id, target, force))
        _running.add(task)
        task.add_done_callback(_running.discard)
        return {"job_id": job_id, "repo_name": target.name, "status": SyncStatus.PENDING.value}

    async def status(self, job_id: str) -> dict | None:
        async with session_scope() as session:
            job = await session.get(SyncJob, job_id)
            if job is None:
                return None
            return {
                "job_id": job.id,
                "repo_name": job.repo_name,
                "status": job.status.value,
                "phase": job.phase,
                "files_total": job.files_total,
                "files_done": job.files_done,
                "files_skipped": job.files_skipped,
                "chunks_upserted": job.chunks_upserted,
                "chunks_deleted": job.chunks_deleted,
                "error": job.error,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            }

    async def fail_interrupted(self) -> int:
        """Close out jobs a restart killed mid-run.

        Jobs run as in-process tasks, so a restart ends them without a terminal
        state; left alone they would report "running" forever and a client
        polling get_sync_status would wait forever. Called once at startup,
        before any new job can exist.
        """
        async with session_scope() as session:
            result = await session.execute(
                update(SyncJob)
                .where(SyncJob.status.in_([SyncStatus.PENDING, SyncStatus.RUNNING]))
                .values(
                    status=SyncStatus.FAILED,
                    phase="failed",
                    error="interrupted by a server restart; start it again",
                    finished_at=datetime.now(UTC),
                )
            )
            return result.rowcount or 0

    # --- worker --------------------------------------------------------------

    async def _run(self, job_id: str, target: RepoTarget, force: bool) -> None:
        lock = _repo_locks.setdefault(target.name, asyncio.Lock())
        async with lock:
            try:
                await self._sync(job_id, target, force)
            except Exception as exc:  # noqa: BLE001 - terminal state must be recorded
                logger.exception("sync job %s failed", job_id)
                await self._finish(job_id, SyncStatus.FAILED, error=str(exc))

    async def _sync(self, job_id: str, target: RepoTarget, force: bool) -> None:
        await self._update(job_id, status=SyncStatus.RUNNING, phase="scanning")

        repo_id = await self._upsert_repository(target)
        files = list_source_files(target)
        await self._update(job_id, files_total=len(files))

        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(IndexedFile.file_path, IndexedFile.content_hash).where(
                        IndexedFile.repo_id == repo_id
                    )
                )
            ).all()
        known: dict[str, str] = {path: digest for path, digest in rows}

        seen: set[str] = set()
        upserted = 0
        deleted = 0
        done = 0
        skipped = 0

        await self._update(job_id, phase="indexing")
        for rel in files:
            rel_str = rel.as_posix()
            absolute = target.path / rel
            try:
                if absolute.stat().st_size > self._settings.max_file_bytes:
                    skipped += 1
                    continue
                raw = absolute.read_bytes()
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                # Binary or unreadable: nothing useful to embed. Deliberately not
                # marked as seen, so a file that was indexed before and has since
                # become unreadable has its stale vectors pruned below.
                skipped += 1
                continue

            seen.add(rel_str)
            digest = hashlib.sha256(raw).hexdigest()
            if not force and known.get(rel_str) == digest:
                done += 1
                continue

            language = detect_language(rel)
            assert language is not None  # list_source_files filtered on this
            chunks = chunk_source(text, language, rel_str, self._settings.max_chunk_chars)

            # Always clear first: a file that now yields fewer chunks must not
            # leave orphaned points behind.
            await self._store.delete_file(target.name, rel_str)

            if chunks:
                dense, sparse = await self._embedder.embed_documents([c.content for c in chunks])
                upserted += await self._store.upsert_chunks(
                    repo_name=target.name,
                    file_path=rel_str,
                    commit_sha=target.head_sha,
                    dirty=target.dirty,
                    chunks=chunks,
                    dense=dense,
                    sparse=sparse,
                )

            await self._record_file(repo_id, rel_str, digest, len(chunks))
            done += 1
            if done % 25 == 0:
                await self._update(
                    job_id, files_done=done, files_skipped=skipped, chunks_upserted=upserted
                )

        # Files indexed previously but no longer present in the tree.
        await self._update(job_id, phase="pruning")
        for stale in set(known) - seen:
            await self._store.delete_file(target.name, stale)
            deleted += 1
            async with session_scope() as session:
                await session.execute(
                    delete(IndexedFile).where(
                        IndexedFile.repo_id == repo_id, IndexedFile.file_path == stale
                    )
                )

        async with session_scope() as session:
            repo = await session.get(Repository, repo_id)
            if repo is not None:
                repo.last_synced_commit = target.head_sha

        await self._finish(
            job_id,
            SyncStatus.SUCCEEDED,
            files_done=done,
            files_skipped=skipped,
            chunks_upserted=upserted,
            chunks_deleted=deleted,
        )

    # --- persistence helpers -------------------------------------------------

    async def _upsert_repository(self, target: RepoTarget) -> int:
        async with session_scope() as session:
            repo = (
                await session.execute(select(Repository).where(Repository.repo_name == target.name))
            ).scalar_one_or_none()
            if repo is None:
                repo = Repository(repo_name=target.name)
                session.add(repo)
            repo.mount_path = str(target.path)
            repo.origin_url = target.origin_url
            repo.head_sha = target.head_sha
            await session.flush()
            return repo.id

    async def _record_file(
        self, repo_id: int, file_path: str, digest: str, chunk_count: int
    ) -> None:
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(IndexedFile).where(
                        IndexedFile.repo_id == repo_id, IndexedFile.file_path == file_path
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                session.add(
                    IndexedFile(
                        repo_id=repo_id,
                        file_path=file_path,
                        content_hash=digest,
                        chunk_count=chunk_count,
                    )
                )
            else:
                row.content_hash = digest
                row.chunk_count = chunk_count

    async def _update(self, job_id: str, **fields) -> None:
        async with session_scope() as session:
            job = await session.get(SyncJob, job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    async def _finish(self, job_id: str, status: SyncStatus, **fields) -> None:
        await self._update(
            job_id,
            status=status,
            phase="done" if status is SyncStatus.SUCCEEDED else "failed",
            finished_at=datetime.now(UTC),
            **fields,
        )
