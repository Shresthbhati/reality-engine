"""accounts and invites (plan T-6.4)

Revision ID: 0004
Revises: 0003

Checks before it creates, like 0002/0003, because init_db's create_all makes the same tables on dev databases.
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    present = set(sa.inspect(op.get_bind()).get_table_names())
    if "accounts" not in present:
        op.create_table(
            "accounts",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("email", sa.String(length=254), nullable=False),
            sa.Column("password_hash", sa.String(length=300), nullable=False),
            sa.Column("tenant_id", sa.String(length=32), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("is_owner", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_accounts_email", "accounts", ["email"], unique=True)
        op.create_index("ix_accounts_tenant_id", "accounts", ["tenant_id"])
    if "invites" not in present:
        op.create_table(
            "invites",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("tenant_id", sa.String(length=32), nullable=False),
            sa.Column("email", sa.String(length=254), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False, unique=True),
            sa.Column("created_by", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_invites_tenant_id", "invites", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("invites")
    op.drop_table("accounts")
