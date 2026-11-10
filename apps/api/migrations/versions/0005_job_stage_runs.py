"""job_stage_runs table (P103).

Checks before it acts so it applies cleanly on a database that init_db already brought up to date.

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if 'job_stage_runs' in set(sa.inspect(op.get_bind()).get_table_names()):
        return
    op.create_table(
        'job_stage_runs',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('job_id', sa.String(length=32), nullable=False),
        sa.Column('team_id', sa.String(length=32), nullable=True),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('version', sa.String(length=32), nullable=False),
        sa.Column('inputs_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('attempt', sa.Integer(), nullable=False),
        sa.Column('outputs', sa.JSON(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('peak_rss_mb', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'name', 'inputs_hash', 'attempt', name='uq_job_stage_runs_job_name_hash_attempt'),
    )
    op.create_index('ix_job_stage_runs_job_id', 'job_stage_runs', ['job_id'], unique=False)
    op.create_index('ix_job_stage_runs_team_id', 'job_stage_runs', ['team_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_job_stage_runs_team_id', table_name='job_stage_runs')
    op.drop_index('ix_job_stage_runs_job_id', table_name='job_stage_runs')
    op.drop_table('job_stage_runs')
