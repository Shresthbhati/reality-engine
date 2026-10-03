"""Database setup for the Reality Engine application API.

DATABASE_URL selects the backend:
  sqlite+aiosqlite:///path (development default)
  postgresql+asyncpg://... (production, PostGIS-ready)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apps.api.models import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/app.db")


def _ensure_sqlite_dir(url: str) -> None:
    """A clean machine has no ./data: SQLite cannot create the file's directory, so the default URL made the API refuse to
    start. Create the parent directory of a file-backed sqlite URL (``:memory:`` and server databases are untouched)."""
    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix):
        path = url[len(prefix):].split("?", 1)[0]
        if path and path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        url = database_url()
        _ensure_sqlite_dir(url)
        _engine = create_async_engine(url, echo=False, future=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def init_db() -> None:
    """Create tables. In production this is replaced by migrations; for the
    dev slice, metadata.create_all is deterministic and idempotent."""
    get_engine()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


async def get_db() -> AsyncIterator[AsyncSession]:
    maker = get_sessionmaker()
    async with maker() as session:
        yield session