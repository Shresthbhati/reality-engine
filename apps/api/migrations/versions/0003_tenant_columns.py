"""tenant ownership: uploads.team_id, evidence.team_id, sites.team_id

Revision ID: 0003
Revises: 0002

Worlds and sessions already carry team_id. Uploads, evidence and sites now carry it too so the shared
``enforce_project_isolation`` check applies to them. Existing rows stay NULL (unowned); there is no data to infer
an owner from. Checks before it acts, like 0002, because init_db adds the same columns on dev databases.
"""
from alembic import op
import sqlalchemy as sa

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

_TABLES = ("uploads", "evidence", "sites")


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    for table in _TABLES:
        if "team_id" not in {c["name"] for c in insp.get_columns(table)}:
            op.add_column(table, sa.Column("team_id", sa.String(length=32), nullable=True))


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "team_id")
