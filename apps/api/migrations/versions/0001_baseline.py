"""baseline: the schema as created by metadata.create_all before Alembic

Revision ID: 0001
Revises: 
"""
from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent: a dev DB built by the old metadata.create_all already has these tables (see README.md).
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if 'activity_events' not in existing:
        op.create_table('activity_events',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('entity_type', sa.String(length=32), nullable=True),
        sa.Column('entity_id', sa.String(length=32), nullable=True),
        sa.Column('actor_id', sa.String(length=32), nullable=True),
        sa.Column('summary', sa.String(length=500), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('activity_events', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_activity_events_type'), ['type'], unique=False)
    if 'jobs' not in existing:
        op.create_table('jobs',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('entity_type', sa.String(length=32), nullable=False),
        sa.Column('entity_id', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('stage', sa.String(length=128), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('max_attempts', sa.Integer(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('worker_id', sa.String(length=64), nullable=True),
        sa.Column('cancel_requested', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('jobs', schema=None) as batch_op:
            batch_op.create_index('ix_jobs_entity', ['entity_type', 'entity_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_jobs_status'), ['status'], unique=False)
            batch_op.create_index(batch_op.f('ix_jobs_type'), ['type'], unique=False)
    if 'locations' not in existing:
        op.create_table('locations',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('entity_type', sa.String(length=32), nullable=False),
        sa.Column('entity_id', sa.String(length=32), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('altitude', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('heading', sa.Float(), nullable=True),
        sa.Column('speed', sa.Float(), nullable=True),
        sa.Column('source', sa.String(length=32), nullable=False),
        sa.Column('coordinate_reference_system', sa.String(length=16), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('locations', schema=None) as batch_op:
            batch_op.create_index('ix_locations_entity', ['entity_type', 'entity_id'], unique=False)
    if 'notifications' not in existing:
        op.create_table('notifications',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('user_id', sa.String(length=32), nullable=True),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=300), nullable=False),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('entity_type', sa.String(length=32), nullable=True),
        sa.Column('entity_id', sa.String(length=32), nullable=True),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('notifications', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_notifications_user_id'), ['user_id'], unique=False)
    if 'site_anchors' not in existing:
        op.create_table('site_anchors',
        sa.Column('site_id', sa.String(length=32), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lon', sa.Float(), nullable=False),
        sa.Column('rotation_deg', sa.Float(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('site_id')
        )
    if 'sites' not in existing:
        op.create_table('sites',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('grid_cell_m', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
    if 'worlds' not in existing:
        op.create_table('worlds',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('owner_id', sa.String(length=32), nullable=True),
        sa.Column('team_id', sa.String(length=32), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('coverage', sa.JSON(), nullable=True),
        sa.Column('current_version_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
    if 'assistant_plans' not in existing:
        op.create_table('assistant_plans',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('base_version_id', sa.String(length=64), nullable=True),
        sa.Column('operation', sa.JSON(), nullable=False),
        sa.Column('preview', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('result_version_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('assistant_plans', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_assistant_plans_world_id'), ['world_id'], unique=False)
    if 'completion_proposals' not in existing:
        op.create_table('completion_proposals',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('base_version_id', sa.String(length=64), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('alt_index', sa.Integer(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False),
        sa.Column('spec', sa.JSON(), nullable=False),
        sa.Column('summary', sa.JSON(), nullable=False),
        sa.Column('accepted_version_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('completion_proposals', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_completion_proposals_world_id'), ['world_id'], unique=False)
    if 'sessions' not in existing:
        op.create_table('sessions',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('owner_id', sa.String(length=32), nullable=True),
        sa.Column('team_id', sa.String(length=32), nullable=True),
        sa.Column('world_id', sa.String(length=32), nullable=True),
        sa.Column('device_metadata', sa.JSON(), nullable=True),
        sa.Column('coordinate_reference_system', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('sessions', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_sessions_status'), ['status'], unique=False)
            batch_op.create_index(batch_op.f('ix_sessions_world_id'), ['world_id'], unique=False)
    if 'site_composites' not in existing:
        op.create_table('site_composites',
        sa.Column('site_id', sa.String(length=32), nullable=False),
        sa.Column('kind', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('site_id', 'kind', 'name')
        )
        with op.batch_alter_table('site_composites', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_site_composites_world_id'), ['world_id'], unique=False)
    if 'site_placements' not in existing:
        op.create_table('site_placements',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('site_id', sa.String(length=32), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('label', sa.String(length=120), nullable=True),
        sa.Column('cell_x', sa.Integer(), nullable=False),
        sa.Column('cell_z', sa.Integer(), nullable=False),
        sa.Column('offset_x_m', sa.Float(), nullable=False),
        sa.Column('offset_z_m', sa.Float(), nullable=False),
        sa.Column('rotation_deg', sa.Float(), nullable=False),
        sa.Column('scale', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['site_id'], ['sites.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('site_placements', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_site_placements_site_id'), ['site_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_site_placements_world_id'), ['world_id'], unique=False)
    if 'world_intents' not in existing:
        op.create_table('world_intents',
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('scale', sa.String(length=32), nullable=False),
        sa.Column('capabilities', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('world_id')
        )
    if 'world_measurements' not in existing:
        op.create_table('world_measurements',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('kind', sa.String(length=32), nullable=False),
        sa.Column('points', sa.JSON(), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(length=16), nullable=False),
        sa.Column('uncertainty', sa.Float(), nullable=True),
        sa.Column('method', sa.String(length=200), nullable=False),
        sa.Column('scale_state', sa.String(length=16), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=True),
        sa.Column('verified_value', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('world_measurements', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_world_measurements_world_id'), ['world_id'], unique=False)
    if 'world_versions' not in existing:
        op.create_table('world_versions',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('world_id', sa.String(length=32), nullable=False),
        sa.Column('parent_version_id', sa.String(length=64), nullable=True),
        sa.Column('artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('artifact_hash', sa.String(length=128), nullable=True),
        sa.Column('source_session_ids', sa.JSON(), nullable=False),
        sa.Column('changed_entity_ids', sa.JSON(), nullable=False),
        sa.Column('changed_geometry_ids', sa.JSON(), nullable=False),
        sa.Column('report', sa.JSON(), nullable=True),
        sa.Column('points_artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('cameras_artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['world_id'], ['worlds.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('world_versions', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_world_versions_world_id'), ['world_id'], unique=False)
    if 'evidence' not in existing:
        op.create_table('evidence',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=300), nullable=False),
        sa.Column('type', sa.String(length=32), nullable=False),
        sa.Column('session_id', sa.String(length=32), nullable=True),
        sa.Column('upload_id', sa.String(length=32), nullable=True),
        sa.Column('processing_state', sa.String(length=32), nullable=False),
        sa.Column('mime_type', sa.String(length=200), nullable=True),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('checksum', sa.String(length=128), nullable=True),
        sa.Column('artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('evidence', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_evidence_processing_state'), ['processing_state'], unique=False)
            batch_op.create_index(batch_op.f('ix_evidence_session_id'), ['session_id'], unique=False)
    if 'uploads' not in existing:
        op.create_table('uploads',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('session_id', sa.String(length=32), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=False),
        sa.Column('mime_type', sa.String(length=200), nullable=True),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('checksum', sa.String(length=128), nullable=True),
        sa.Column('artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('uploads', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_uploads_session_id'), ['session_id'], unique=False)
    if 'evidence_decisions' not in existing:
        op.create_table('evidence_decisions',
        sa.Column('evidence_id', sa.String(length=32), nullable=False),
        sa.Column('decision', sa.String(length=32), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['evidence_id'], ['evidence.id'], ),
        sa.PrimaryKeyConstraint('evidence_id')
        )


def downgrade() -> None:
    op.drop_table('assistant_plans')
    op.drop_table('evidence_decisions')
    with op.batch_alter_table('uploads', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_uploads_session_id'))

    op.drop_table('uploads')
    with op.batch_alter_table('evidence', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_evidence_session_id'))
        batch_op.drop_index(batch_op.f('ix_evidence_processing_state'))

    op.drop_table('evidence')
    with op.batch_alter_table('world_versions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_world_versions_world_id'))

    op.drop_table('world_versions')
    with op.batch_alter_table('world_measurements', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_world_measurements_world_id'))

    op.drop_table('world_measurements')
    op.drop_table('world_intents')
    with op.batch_alter_table('site_placements', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_site_placements_world_id'))
        batch_op.drop_index(batch_op.f('ix_site_placements_site_id'))

    op.drop_table('site_placements')
    with op.batch_alter_table('site_composites', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_site_composites_world_id'))

    op.drop_table('site_composites')
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_sessions_world_id'))
        batch_op.drop_index(batch_op.f('ix_sessions_status'))

    op.drop_table('sessions')
    with op.batch_alter_table('completion_proposals', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_completion_proposals_world_id'))

    op.drop_table('completion_proposals')
    op.drop_table('worlds')
    op.drop_table('sites')
    op.drop_table('site_anchors')
    with op.batch_alter_table('notifications', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_notifications_user_id'))

    op.drop_table('notifications')
    with op.batch_alter_table('locations', schema=None) as batch_op:
        batch_op.drop_index('ix_locations_entity')

    op.drop_table('locations')
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_jobs_type'))
        batch_op.drop_index(batch_op.f('ix_jobs_status'))
        batch_op.drop_index('ix_jobs_entity')

    op.drop_table('jobs')
    with op.batch_alter_table('activity_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_activity_events_type'))

    op.drop_table('activity_events')
