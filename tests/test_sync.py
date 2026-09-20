"""Incremental sync: hash diffing, the delete path, job tracking, path containment.

These are the definition-of-done behaviours for milestone 1, and the ones the
original spec left unverified — it had no deletion path and no job model at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.db.models import SyncStatus
from src.ingest.git_sync import SyncError, list_source_files, resolve_repo
from src.vector.qdrant import QdrantStore

from .conftest import StubEmbedder


async def run_sync(sync, repo: str = "demo", force: bool = False) -> dict:
    """Start a sync and wait for it to reach a terminal state."""
    started = await sync.start(repo, force=force)
    job_id = started["job_id"]
    for _ in range(200):
        status = await sync.status(job_id)
        assert status is not None
        if status["status"] in (SyncStatus.SUCCEEDED.value, SyncStatus.FAILED.value):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("sync did not finish")


class TestRepoResolution:
    def test_bare_name_resolves_under_the_mount_root(self, settings, git_repo):
        target = resolve_repo(settings, "demo")
        assert target.path == git_repo.resolve()
        assert target.name == "demo"

    def test_git_metadata_is_captured(self, settings, git_repo):
        target = resolve_repo(settings, "demo")
        assert target.head_sha is not None
        assert len(target.head_sha) == 40
        assert target.dirty is False

    def test_dirty_tree_is_flagged(self, settings, git_repo):
        """Working-tree hashing indexes uncommitted edits, so this must be visible."""
        (git_repo / "src" / "auth.py").write_text("def changed():\n    return 1\n")
        assert resolve_repo(settings, "demo").dirty is True

    def test_path_traversal_is_refused(self, settings, git_repo):
        with pytest.raises(SyncError, match="outside"):
            resolve_repo(settings, "../../etc")

    def test_absolute_path_outside_the_root_is_refused(self, settings, git_repo):
        with pytest.raises(SyncError, match="outside"):
            resolve_repo(settings, "/etc")

    def test_missing_repo_is_refused(self, settings):
        settings.repos_root.mkdir(parents=True, exist_ok=True)
        with pytest.raises(SyncError, match="not a directory"):
            resolve_repo(settings, "nope")


class TestFileDiscovery:
    def test_only_indexable_files_are_listed(self, settings, git_repo):
        files = {p.as_posix() for p in list_source_files(resolve_repo(settings, "demo"))}
        assert "src/auth.py" in files
        assert "src/api.ts" in files
        # Markdown is not indexable in milestone 1 (docs arrive in milestone 2).
        assert "README.md" not in files

    def test_gitignored_files_are_excluded(self, settings, git_repo):
        """.gitignore is honoured by asking git, rather than reimplementing it."""
        files = {p.as_posix() for p in list_source_files(resolve_repo(settings, "demo"))}
        assert "ignored/secret.py" not in files

    def test_ignored_directories_are_excluded(self, settings, git_repo):
        vendored = git_repo / "node_modules" / "pkg"
        vendored.mkdir(parents=True)
        (vendored / "index.ts").write_text("export const x = 1;\n")
        files = {p.as_posix() for p in list_source_files(resolve_repo(settings, "demo"))}
        assert not any("node_modules" in f for f in files)


class TestForeignOwnership:
    """The engine runs as root; bind-mounted checkouts belong to the host user.

    git refuses to operate on a repository owned by another user ("dubious
    ownership"), so without an explicit exception every sync on a real deployment
    crashed. ``GIT_TEST_ASSUME_DIFFERENT_OWNER`` is git's own switch for
    reproducing that check without root.
    """

    @pytest.fixture(autouse=True)
    def foreign_owner(self, monkeypatch):
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")

    def test_a_repo_owned_by_another_user_resolves(self, settings, git_repo):
        target = resolve_repo(settings, "demo")
        assert target.head_sha is not None

    def test_a_repo_owned_by_another_user_still_honours_gitignore(self, settings, git_repo):
        files = {p.as_posix() for p in list_source_files(resolve_repo(settings, "demo"))}
        assert "src/auth.py" in files
        assert "ignored/secret.py" not in files


class TestGitFailure:
    def test_a_git_failure_is_not_masked_by_a_filesystem_walk(
        self, settings, git_repo, monkeypatch
    ):
        """A walk would ignore .gitignore and index whatever it excludes."""
        from git import GitCommandError
        from git.cmd import Git

        def broken(self, *args, **kwargs):
            raise GitCommandError("ls-files", 128)

        target = resolve_repo(settings, "demo")
        monkeypatch.setattr(Git, "ls_files", broken, raising=False)
        with pytest.raises(GitCommandError):
            list_source_files(target)

    def test_a_plain_directory_is_still_walked(self, settings):
        plain = settings.repos_root / "plain"
        plain.mkdir(parents=True)
        (plain / "app.py").write_text("def main():\n    return 1\n")
        files = {p.as_posix() for p in list_source_files(resolve_repo(settings, "plain"))}
        assert files == {"app.py"}


@pytest.mark.usefixtures("database")
class TestIncrementalSync:
    async def test_first_sync_indexes_the_tree(self, sync, store: QdrantStore, git_repo):
        status = await run_sync(sync)
        assert status["status"] == SyncStatus.SUCCEEDED.value
        assert status["files_total"] == 2
        assert status["files_done"] == 2
        assert status["chunks_upserted"] > 0
        assert await store.count("demo") > 0

    async def test_resync_without_changes_is_a_no_op(
        self, sync, store: QdrantStore, embedder: StubEmbedder, git_repo
    ):
        await run_sync(sync)
        count_after_first = await store.count("demo")
        calls_after_first = embedder.embed_calls

        status = await run_sync(sync)
        assert status["status"] == SyncStatus.SUCCEEDED.value
        # Nothing re-embedded, and the index is unchanged.
        assert embedder.embed_calls == calls_after_first
        assert status["chunks_upserted"] == 0
        assert await store.count("demo") == count_after_first

    async def test_force_reindexes_everything(self, sync, embedder: StubEmbedder, git_repo):
        await run_sync(sync)
        calls_after_first = embedder.embed_calls
        status = await run_sync(sync, force=True)
        assert embedder.embed_calls > calls_after_first
        assert status["chunks_upserted"] > 0

    async def test_changing_one_file_reindexes_only_that_file(
        self, sync, embedder: StubEmbedder, git_repo: Path
    ):
        await run_sync(sync)
        embedder.embedded_texts.clear()

        (git_repo / "src" / "api.ts").write_text(
            "export function createSession(id: string) {\n  return { id };\n}\n"
            "export function destroySession(id: string) {\n  return null;\n}\n"
        )
        status = await run_sync(sync)
        assert status["chunks_upserted"] == 2
        # Only TypeScript content was re-embedded; the Python file was skipped.
        assert all("export function" in t for t in embedder.embedded_texts)

    async def test_deleting_a_file_removes_its_vectors(
        self, sync, store: QdrantStore, git_repo: Path
    ):
        await run_sync(sync)
        before = await store.count("demo")

        (git_repo / "src" / "api.ts").unlink()
        status = await run_sync(sync)

        assert status["chunks_deleted"] == 1
        after = await store.count("demo")
        assert after < before
        # Nothing from the deleted file survives.
        hits = await store.client.scroll(collection_name=store.collection_name, limit=100)
        assert all(p.payload["file_path"] != "src/api.ts" for p in hits[0])

    async def test_shrinking_a_file_leaves_no_orphan_chunks(
        self, sync, store: QdrantStore, git_repo: Path
    ):
        """A file that now yields fewer chunks must not leave stale points behind."""
        await run_sync(sync)
        before = await store.count("demo")

        (git_repo / "src" / "auth.py").write_text("def only_one():\n    return 1\n")
        await run_sync(sync)

        assert await store.count("demo") < before

    async def test_new_file_is_picked_up(self, sync, store: QdrantStore, git_repo: Path):
        await run_sync(sync)
        before = await store.count("demo")
        (git_repo / "src" / "extra.py").write_text("def extra():\n    return 42\n")
        status = await run_sync(sync)
        assert status["chunks_upserted"] == 1
        assert await store.count("demo") == before + 1


class TestEmbeddingIsSerialized:
    async def test_two_jobs_do_not_embed_at_once(self):
        """Concurrent jobs multiplied memory until the kernel killed the engine."""
        from src.vector.embedder import FastEmbedder

        embedder = FastEmbedder.__new__(FastEmbedder)
        live, peak = 0, 0

        async def fake(texts):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return [], []

        embedder._embed_documents = fake
        await asyncio.gather(*(FastEmbedder.embed_documents(embedder, ["x"]) for _ in range(5)))
        assert peak == 1


@pytest.mark.usefixtures("database")
class TestJobTracking:
    async def test_start_returns_a_job_handle_immediately(self, sync, git_repo):
        started = await sync.start("demo")
        assert "job_id" in started
        assert started["repo_name"] == "demo"
        # Let the background task settle so the event loop closes cleanly.
        await run_sync(sync)

    async def test_unknown_job_id_returns_none(self, sync):
        assert await sync.status("does-not-exist") is None

    async def test_failure_is_recorded_as_a_terminal_state(self, sync, git_repo, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("qdrant exploded")

        monkeypatch.setattr(sync._store, "delete_file", boom)
        status = await run_sync(sync)
        assert status["status"] == SyncStatus.FAILED.value
        assert "qdrant exploded" in status["error"]
        assert status["finished_at"] is not None

    async def test_jobs_left_running_by_a_restart_are_marked_failed(self, sync):
        from src.db.models import SyncJob
        from src.db.postgres import session_scope

        async with session_scope() as session:
            session.add(SyncJob(id="stale", repo_name="demo", status=SyncStatus.RUNNING))
            session.add(SyncJob(id="done", repo_name="demo", status=SyncStatus.SUCCEEDED))
        assert await sync.fail_interrupted() == 1
        stale = await sync.status("stale")
        assert stale["status"] == SyncStatus.FAILED.value
        assert "restart" in stale["error"]
        assert (await sync.status("done"))["status"] == SyncStatus.SUCCEEDED.value

    async def test_unresolvable_repo_raises_before_a_job_is_created(self, sync):
        with pytest.raises(SyncError):
            await sync.start("../escape")
