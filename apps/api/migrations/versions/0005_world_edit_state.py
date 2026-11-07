"""per-world redo list (plan T-6.7)

Revision ID: 0005
Revises: 0004

Checks before it creates, like the earlier revisions, because init_db's create_all makes the same table on dev databases.
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "world_edit_state" not in set(sa.inspect(op.get_bind()).get_table_names()):
        op.create_table(
            "world_edit_state",
            sa.Column("world_id", sa.String(length=32), sa.ForeignKey("worlds.id"), primary_key=True),
            sa.Column("redo", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    op.drop_table("world_edit_state")
