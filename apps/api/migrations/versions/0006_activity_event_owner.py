"""activity_events.team_id: audit events carry their own owner.

Revision ID: 0006
Revises: 0005

Until now an activity event was only visible through its subject, so an event about a deleted or purged entity was
hidden from the team that owned it. The column is added nullable and backfilled ONLY where the owner can be proven:
the event's subject row still exists and has an owner. Every other row stays NULL (unowned: hidden from tenants,
visible to system administrators); no owner is ever assigned without evidence. Checks before it acts, like 0002-0005,
because init_db adds the same column on dev databases. Reversible: downgrade drops the index and the column.
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None

#: entity_type -> table whose ``team_id`` proves the event's owner.
_SUBJECT_TABLES = {
    "world": "worlds", "session": "sessions", "evidence": "evidence", "upload": "uploads", "site": "sites",
    "job": "jobs", "asset": "assets",
}


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "team_id" not in {c["name"] for c in insp.get_columns("activity_events")}:
        op.add_column("activity_events", sa.Column("team_id", sa.String(length=32), nullable=True))
    if "ix_activity_events_team_id" not in {i["name"] for i in sa.inspect(bind).get_indexes("activity_events")}:
        op.create_index("ix_activity_events_team_id", "activity_events", ["team_id"])
    present = set(sa.inspect(bind).get_table_names())
    for entity_type, table in _SUBJECT_TABLES.items():
        if table not in present:
            continue
        op.execute(sa.text(
            f"UPDATE activity_events SET team_id = (SELECT {table}.team_id FROM {table} "
            f"WHERE {table}.id = activity_events.entity_id) "
            "WHERE team_id IS NULL AND entity_type = :t AND entity_id IS NOT NULL"
        ).bindparams(t=entity_type))


def downgrade() -> None:
    with op.batch_alter_table("activity_events") as batch:
        batch.drop_index("ix_activity_events_team_id")
        batch.drop_column("team_id")
