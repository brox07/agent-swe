"""Async SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(url: str | None = None) -> AsyncEngine:
    """Create the process-wide engine. Idempotent."""
    global _engine, _sessionmaker
    if _engine is None:
        settings = get_settings()
        target = url or settings.postgres_url
        # SQLite (used by the test suite) does not accept server-pool sizing.
        pool_kwargs = (
            {} if target.startswith("sqlite") else {"pool_size": 5, "max_overflow": 5}
        )
        _engine = create_async_engine(target, pool_pre_ping=True, **pool_kwargs)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session. Commits on success, rolls back on exception."""
    if _sessionmaker is None:
        init_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_connection() -> bool:
    """Used by /health, so the container reports ready rather than merely alive."""
    from sqlalchemy import text

    try:
        engine = init_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
