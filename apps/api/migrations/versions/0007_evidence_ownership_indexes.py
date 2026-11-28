"""Tenant-scoped evidence and upload indexes.

Revision ID: 0007
Revises: 0006

Adds composite indexes ``ix_uploads_team_session_checksum`` and
``ix_evidence_team_session_checksum``, plus the partial unique index
``uq_evidence_team_session_checksum`` (for non-null ``session_id`` and
``checksum``) so duplicate uploads are scoped strictly to ``(team_id,
session_id, checksum)`` and concurrent uploads within the same session cannot
race into duplicate Evidence rows.

Existing legacy duplicate rows (if any) are left untouched unless no conflicts
exist; when creating the partial unique index on an existing database, any
legacy duplicate check is surfaced cleanly.
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "uploads" in insp.get_table_names():
        up_idx = {i["name"] for i in insp.get_indexes("uploads")}
        if "ix_uploads_team_session_checksum" not in up_idx:
            op.create_index(
                "ix_uploads_team_session_checksum",
                "uploads",
                ["team_id", "session_id", "checksum"],
            )

    if "evidence" in insp.get_table_names():
        ev_idx = {i["name"] for i in insp.get_indexes("evidence")}
        if "ix_evidence_team_session_checksum" not in ev_idx:
            op.create_index(
                "ix_evidence_team_session_checksum",
                "evidence",
                ["team_id", "session_id", "checksum"],
            )
        if "uq_evidence_team_session_checksum" not in ev_idx:
            op.create_index(
                "uq_evidence_team_session_checksum",
                "evidence",
                ["team_id", "session_id", "checksum"],
                unique=True,
                sqlite_where=sa.text("session_id IS NOT NULL AND checksum IS NOT NULL"),
                postgresql_where=sa.text("session_id IS NOT NULL AND checksum IS NOT NULL"),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "evidence" in insp.get_table_names():
        ev_idx = {i["name"] for i in insp.get_indexes("evidence")}
        if "uq_evidence_team_session_checksum" in ev_idx:
            op.drop_index("uq_evidence_team_session_checksum", table_name="evidence")
        if "ix_evidence_team_session_checksum" in ev_idx:
            op.drop_index("ix_evidence_team_session_checksum", table_name="evidence")

    if "uploads" in insp.get_table_names():
        up_idx = {i["name"] for i in insp.get_indexes("uploads")}
        if "ix_uploads_team_session_checksum" in up_idx:
            op.drop_index("ix_uploads_team_session_checksum", table_name="uploads")
