"""named design options (plan T-4b.3)

Revision ID: 0006
Revises: 0005

Checks before it creates, like the earlier revisions, because init_db's create_all makes the same table on dev databases.
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "world_options" not in set(sa.inspect(op.get_bind()).get_table_names()):
        op.create_table(
            "world_options",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("world_id", sa.String(length=32), sa.ForeignKey("worlds.id"), nullable=False),
            sa.Column("name", sa.String(length=80), nullable=False),
            sa.Column("version_id", sa.String(length=64), nullable=False),
            sa.Column("created_by", sa.String(length=32), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_world_options_world_id", "world_options", ["world_id"])


def downgrade() -> None:
    op.drop_table("world_options")
