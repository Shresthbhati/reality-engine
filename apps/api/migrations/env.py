"""Alembic environment. The URL is, in order: an explicit ``sqlalchemy.url`` main option (tests set it), else the
``DATABASE_URL`` env var the API uses, else the API's SQLite default. Async driver suffixes are dropped because
migrations run on a plain synchronous engine."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from apps.api.models import Base
from apps.api import routes_assistant  # noqa: F401  registers AssistantPlan on Base.metadata (as the API does)

config = context.config
target_metadata = Base.metadata


def _sync_url() -> str:
    url = config.get_main_option("sqlalchemy.url") or os.environ.get(
        "DATABASE_URL", "sqlite+aiosqlite:///./data/app.db")
    for async_suffix in ("+aiosqlite", "+asyncpg"):
        url = url.replace(async_suffix, "")
    if url.startswith("sqlite:///") and ":memory:" not in url:
        from pathlib import Path
        Path(url[len("sqlite:///"):].split("?", 1)[0]).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return url


def run_migrations_offline() -> None:
    context.configure(url=_sync_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
