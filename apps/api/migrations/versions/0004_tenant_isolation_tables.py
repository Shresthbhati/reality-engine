"""Tenant/subscription/account/isolation tables + jobs.team_id (P0-3 refinement).

Adds all model tables that existed in metadata but had no Alembic migration,
plus the jobs.team_id column that 0003 forgot. Every step checks before it
acts so this applies cleanly on a database that init_db already brought up
to date.

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    existing = set(insp.get_table_names())

    # jobs.team_id column (model has it, 0003 forgot it)
    if 'jobs' in existing and 'team_id' not in {c['name'] for c in insp.get_columns('jobs')}:
        op.add_column('jobs', sa.Column('team_id', sa.String(length=32), nullable=True))
        op.create_index('ix_jobs_team_id', 'jobs', ['team_id'], unique=False)

    # ---- tenants ----
    if 'tenants' not in existing:
        op.create_table(
            'tenants',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('name', sa.String(length=200), nullable=False),
            sa.Column('slug', sa.String(length=100), nullable=False),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_tenants_slug', 'tenants', ['slug'], unique=True)

    # ---- user_accounts ----
    if 'user_accounts' not in existing:
        op.create_table(
            'user_accounts',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('tenant_id', sa.String(length=32), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('full_name', sa.String(length=200), nullable=False),
            sa.Column('role', sa.String(length=32), nullable=False, server_default='editor'),
            sa.Column('password_hash', sa.String(length=255), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_user_accounts_email', 'user_accounts', ['email'], unique=True)
        op.create_index('ix_user_accounts_tenant_id', 'user_accounts', ['tenant_id'], unique=False)

    # ---- tenant_invites ----
    if 'tenant_invites' not in existing:
        op.create_table(
            'tenant_invites',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('tenant_id', sa.String(length=32), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('role', sa.String(length=32), nullable=False, server_default='editor'),
            sa.Column('token', sa.String(length=128), nullable=False),
            sa.Column('invited_by_user_id', sa.String(length=32), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
            sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['invited_by_user_id'], ['user_accounts.id'], ),
            sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_tenant_invites_email', 'tenant_invites', ['email'], unique=False)
        op.create_index('ix_tenant_invites_tenant_id', 'tenant_invites', ['tenant_id'], unique=False)
        op.create_index('ix_tenant_invites_token', 'tenant_invites', ['token'], unique=True)

    # ---- tenant_subscriptions ----
    if 'tenant_subscriptions' not in existing:
        op.create_table(
            'tenant_subscriptions',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('tenant_id', sa.String(length=32), nullable=False),
            sa.Column('tier', sa.String(length=32), nullable=False, server_default='free'),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
            sa.Column('stripe_customer_id', sa.String(length=128), nullable=True),
            sa.Column('stripe_subscription_id', sa.String(length=128), nullable=True),
            sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
            sa.Column('cancel_at_period_end', sa.Boolean(), nullable=False, server_default='false'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_tenant_subscriptions_tenant_id', 'tenant_subscriptions', ['tenant_id'], unique=True)
        op.create_index('ix_tenant_subscriptions_stripe_customer_id', 'tenant_subscriptions', ['stripe_customer_id'], unique=False)
        op.create_index('ix_tenant_subscriptions_stripe_subscription_id', 'tenant_subscriptions', ['stripe_subscription_id'], unique=False)

    # ---- assets ----
    if 'assets' not in existing:
        op.create_table(
            'assets',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('name', sa.String(length=200), nullable=False),
            sa.Column('category', sa.String(length=64), nullable=False, server_default='furniture'),
            sa.Column('source_world_id', sa.String(length=32), nullable=True),
            sa.Column('source_entity_id', sa.String(length=128), nullable=True),
            sa.Column('dimensions', sa.JSON(), nullable=True),
            sa.Column('provenance', sa.JSON(), nullable=True),
            sa.Column('artifact_uri', sa.String(length=500), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.Column('owner_id', sa.String(length=32), nullable=True),
            sa.Column('team_id', sa.String(length=32), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['source_world_id'], ['worlds.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_assets_source_world_id', 'assets', ['source_world_id'], unique=False)
        op.create_index('ix_assets_source_entity_id', 'assets', ['source_entity_id'], unique=False)
        op.create_index('ix_assets_team_id', 'assets', ['team_id'], unique=False)

    # ---- export_records ----
    if 'export_records' not in existing:
        op.create_table(
            'export_records',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('tenant_id', sa.String(length=32), nullable=True),
            sa.Column('world_id', sa.String(length=32), nullable=False),
            sa.Column('version_id', sa.String(length=64), nullable=True),
            sa.Column('format', sa.String(length=32), nullable=False),
            sa.Column('size_bytes', sa.Integer(), nullable=True),
            sa.Column('artifact_uri', sa.String(length=500), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
            sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_export_records_tenant_id', 'export_records', ['tenant_id'], unique=False)
        op.create_index('ix_export_records_world_id', 'export_records', ['world_id'], unique=False)
        op.create_index('ix_export_records_format', 'export_records', ['format'], unique=False)

    # ---- job_logs ----
    if 'job_logs' not in existing:
        op.create_table(
            'job_logs',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('job_id', sa.String(length=32), nullable=False),
            sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
            sa.Column('stage', sa.String(length=64), nullable=True),
            sa.Column('level', sa.String(length=16), nullable=False, server_default='INFO'),
            sa.Column('message', sa.Text(), nullable=False),
            sa.Column('context_json', sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_job_logs_job_timestamp', 'job_logs', ['job_id', 'timestamp'], unique=False)

    # ---- world_edit_commands ----
    if 'world_edit_commands' not in existing:
        op.create_table(
            'world_edit_commands',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('world_id', sa.String(length=32), nullable=False),
            sa.Column('team_id', sa.String(length=32), nullable=True),
            sa.Column('command_type', sa.String(length=32), nullable=False),
            sa.Column('entity_id', sa.String(length=128), nullable=False),
            sa.Column('description', sa.String(length=255), nullable=False),
            sa.Column('params', sa.JSON(), nullable=False),
            sa.Column('inverse_params', sa.JSON(), nullable=False),
            sa.Column('version_id_before', sa.String(length=64), nullable=True),
            sa.Column('version_id_after', sa.String(length=64), nullable=True),
            sa.Column('undone', sa.Boolean(), nullable=False, server_default='false'),
            sa.Column('sequence', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('undone_at', sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_world_edit_commands_world_undone', 'world_edit_commands', ['world_id', 'undone'], unique=False)
        op.create_index('ix_world_edit_commands_world_seq', 'world_edit_commands', ['world_id', 'sequence'], unique=False)

    # ---- world_camera_views ----
    if 'world_camera_views' not in existing:
        op.create_table(
            'world_camera_views',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('world_id', sa.String(length=32), nullable=False),
            sa.Column('name', sa.String(length=120), nullable=False),
            sa.Column('position', sa.JSON(), nullable=False),
            sa.Column('target', sa.JSON(), nullable=False),
            sa.Column('fov_deg', sa.Float(), nullable=False, server_default='60.0'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )

    # ---- world_share_links ----
    if 'world_share_links' not in existing:
        op.create_table(
            'world_share_links',
            sa.Column('id', sa.String(length=32), nullable=False),
            sa.Column('world_id', sa.String(length=32), nullable=False),
            sa.Column('token', sa.String(length=200), nullable=False),
            sa.Column('label', sa.String(length=200), nullable=True),
            sa.Column('created_by', sa.String(length=32), nullable=True),
            sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_world_share_links_token', 'world_share_links', ['token'], unique=True)
        op.create_index('ix_world_share_links_world_id', 'world_share_links', ['world_id'], unique=False)


def downgrade() -> None:
    op.drop_table('world_share_links')
    op.drop_table('world_camera_views')
    op.drop_table('world_edit_commands')
    op.drop_table('job_logs')
    op.drop_table('assets')
    op.drop_table('tenant_subscriptions')
    op.drop_table('tenant_invites')
    op.drop_table('user_accounts')
    op.drop_table('tenants')
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_index('ix_jobs_team_id')
        batch_op.drop_column('team_id')