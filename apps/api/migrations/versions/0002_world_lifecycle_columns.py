"""doc 13 columns: worlds lifecycle, world_versions label/job, jobs.stage_counts, capture_requests

Revision ID: 0002
Revises: 0001

Every step checks before it acts, so this applies cleanly both on a database that went through 0001 and on a dev
database that the API's init_db already brought up to date (init_db adds the same columns).
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

_COLUMNS = {
    "worlds": [
        ("archived_at", sa.DateTime(timezone=True)),
        ("deleted_at", sa.DateTime(timezone=True)),
        ("thumbnail_uri", sa.String(length=500)),
        ("scale_state", sa.String(length=16)),
    ],
    "world_versions": [
        ("label", sa.String(length=200)),
        ("created_by_job_id", sa.String(length=32)),
    ],
    "jobs": [("stage_counts", sa.JSON())],
    # pre-D1 dev databases lack these two; the baseline assumed them
    "world_measurements": [("state", sa.String(length=16)), ("verified_value", sa.Float())],
}


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    for table, cols in _COLUMNS.items():
        have = {c["name"] for c in insp.get_columns(table)}
        for name, type_ in cols:
            if name not in have:
                op.add_column(table, sa.Column(name, type_, nullable=True))
    if "capture_requests" not in insp.get_table_names():
        op.create_table(
            "capture_requests",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("world_id", sa.String(length=32), nullable=False),
            sa.Column("reason", sa.String(length=300), nullable=False),
            sa.Column("target_region", sa.JSON(), nullable=True),
            sa.Column("target_ids", sa.JSON(), nullable=False),
            sa.Column("suggested_poses", sa.JSON(), nullable=False),
            sa.Column("expected_gain", sa.Float(), nullable=True),
            sa.Column("instruction", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("created_version", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["world_id"], ["worlds.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_capture_requests_world_id", "capture_requests", ["world_id"])
        op.create_index("ix_capture_requests_status", "capture_requests", ["status"])


def downgrade() -> None:
    op.drop_table("capture_requests")
    for table, cols in _COLUMNS.items():
        if table == "world_measurements":
            continue   # belongs to the baseline contract; never dropped here
        with op.batch_alter_table(table) as batch:
            for name, _ in cols:
                batch.drop_column(name)
