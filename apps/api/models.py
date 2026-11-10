"""Application database models for Reality Engine.

The application database stores identity, relationships, permissions,
metadata, job state, and application state. Computational artifacts
(WorldStore versions, point clouds, meshes, reports) remain in the
content-addressed artifact stores; this database references them by
URI and hash, never duplicating them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class World(Base):
    """Application World record. WorldStore remains authoritative for
    immutable computational versions; this row carries application state."""

    __tablename__ = "worlds"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    owner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: lifecycle (doc 13): status is active | archived | trashed; the timestamps record when.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    thumbnail_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    #: NULL until a scale state has been recorded for the world; never guessed.
    scale_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class WorldVersion(Base):
    """Application mirror of a WorldStore version lineage node."""

    __tablename__ = "world_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    parent_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_session_ids: Mapped[list] = mapped_column(JSON, default=list)
    changed_entity_ids: Mapped[list] = mapped_column(JSON, default=list)
    changed_geometry_ids: Mapped[list] = mapped_column(JSON, default=list)
    # Sibling compile-pipeline outputs (points.ply/cameras.json/report.json
    # from engine.pipeline.vertical_slice) -- not part of the versioned
    # WorldIR JSON itself. report is small structured data stored inline;
    # points/cameras are binary/larger, stored content-addressed via
    # apps/api/storage.py and referenced here by sha256:// URI.
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    points_artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    cameras_artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_by_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorldMeasurement(Base):
    """A saved viewport measurement. `value` is always recomputed server-side from
    `points` (world_ir.measure); `unit` is derived from `scale_state`, so a world
    whose scale is unknown can never be stored as metres. `uncertainty` is NULL
    unless the data supplied one -- never a guess."""

    __tablename__ = "world_measurements"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    points: Mapped[list] = mapped_column(JSON)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    method: Mapped[str] = mapped_column(String(200))
    scale_state: Mapped[str] = mapped_column(String(16))
    #: D1 dimension state (MEASURED/RECONSTRUCTED/ESTIMATED/VERIFIED); NULL on rows saved before D1.
    state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: user tape-confirmed value (VERIFIED override); the computed `value` is kept beside it.
    verified_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorldShareLink(Base):
    """Read-only client share link for a world (T-4b.4). Token-scoped, revocable, no account needed."""

    __tablename__ = "world_share_links"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    token: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorldCameraView(Base):
    """Saved camera view for client decks and walk presets (T-4b.6)."""

    __tablename__ = "world_camera_views"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    position: Mapped[dict] = mapped_column(JSON)
    target: Mapped[dict] = mapped_column(JSON)
    fov_deg: Mapped[float] = mapped_column(Float, default=60.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Session(Base):
    """A capture Session. First-class: may exist without a World."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    owner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    world_id: Mapped[str | None] = mapped_column(
        ForeignKey("worlds.id"), nullable=True, index=True
    )
    device_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    coordinate_reference_system: Mapped[str] = mapped_column(String(32), default="WGS84")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    uploaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class Location(Base):
    """A real location observation. Only rows with actual device-reported
    values are written — never fabricated."""

    __tablename__ = "locations"
    __table_args__ = (Index("ix_locations_entity", "entity_type", "entity_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32))  # session | evidence | world
    entity_id: Mapped[str] = mapped_column(String(32))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    altitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="gnss")
    coordinate_reference_system: Mapped[str] = mapped_column(String(16), default="WGS84")
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Upload(Base):
    """An artifact upload. Content-addressed on completion."""

    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    #: tenant that owns the upload (inherited from its session); NULL = unowned, as for worlds and sessions.
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    filename: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Evidence(Base):
    """Evidence linked to the real evidence/ingestion system."""

    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    type: Mapped[str] = mapped_column(String(32))  # photo|video|point_cloud|gnss|imu|sensor_log|dataset
    #: tenant that owns the evidence (inherited from its session); NULL = unowned.
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True, index=True
    )
    upload_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    processing_state: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    mime_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class EvidenceDecision(Base):
    """The operator's Evidence Quality Firewall decision for one evidence item.

    One row per evidence item (PK = evidence_id). Only ``rejected`` changes
    behaviour: rejected evidence is excluded from reconstruction input. The other
    decisions are recorded for the operator and are advisory."""

    __tablename__ = "evidence_decisions"

    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"), primary_key=True)
    decision: Mapped[str] = mapped_column(String(32))  # accepted|accepted_with_warning|needs_review|rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    """Durable application job.

    Lifecycle: queued -> running -> {succeeded | partial | failed |
    cancelled}. Terminal states never leave the row: succeeded/partial
    carry the measured payload, failed carries the error, cancelled
    records an operator or timeout decision. `cancel_requested` lets an
    operator stop a running job at the next stage boundary.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_entity", "entity_type", "entity_id"),
        Index("ix_jobs_team_id", "team_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    #: per-stage counters (e.g. {"images": 24, "registered": 21}); NULL until a stage reports.
    stage_counts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorldIntent(Base):
    """What the user said they are trying to construct (scale + wanted capabilities). Advisory: it records
    intent and drives the honest resource estimate; it does not change what the pipeline does."""

    __tablename__ = "world_intents"

    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), primary_key=True)
    scale: Mapped[str] = mapped_column(String(32))
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CaptureRequest(Base):
    """Listing/status mirror of a WorldIR CaptureRequest (doc 11): the extra photo the system needs.
    WorldIR stays authoritative; this row exists so requests can be listed and filtered without loading a world."""

    __tablename__ = "capture_requests"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    reason: Mapped[str] = mapped_column(String(300))
    target_region: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    target_ids: Mapped[list] = mapped_column(JSON, default=list)
    suggested_poses: Mapped[list] = mapped_column(JSON, default=list)
    expected_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open | satisfied | dismissed
    created_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ActivityEvent(Base):
    __tablename__ = "activity_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary: Mapped[str] = mapped_column(String(500))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CompletionProposal(Base):
    """A GENERATED interior proposal for a world's reconstructed shell. It is a proposal only: it lives in this
    table and changes nothing in the world until accepted (then it is committed through the procedural write path)."""

    __tablename__ = "completion_proposals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    base_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="proposed")  # proposed | accepted | rejected
    alt_index: Mapped[int] = mapped_column(Integer, default=0)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    spec: Mapped[dict] = mapped_column(JSON)       # procedural.spec_store.GeneratedSpec as a dict
    summary: Mapped[dict] = mapped_column(JSON)    # footprint + unit + program, for listing without regenerating
    accepted_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Site(Base):
    """A user-directed composition of several reconstructed worlds on a metric grid. Placement is chosen by the
    user; it is NOT automatic geo-registration."""

    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    grid_cell_m: Mapped[float] = mapped_column(Float, default=50.0)
    #: tenant that owns the site; NULL = unowned.
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SitePlacement(Base):
    """One world placed in a Site cell, with a user-chosen offset/rotation/scale."""

    __tablename__ = "site_placements"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    cell_x: Mapped[int] = mapped_column(Integer)
    cell_z: Mapped[int] = mapped_column(Integer)
    offset_x_m: Mapped[float] = mapped_column(Float, default=0.0)
    offset_z_m: Mapped[float] = mapped_column(Float, default=0.0)
    rotation_deg: Mapped[float] = mapped_column(Float, default=0.0)
    scale: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SiteComposite(Base):
    """The world a Site was composed into (one per site/kind/name; re-composing commits a new version to it)."""

    __tablename__ = "site_composites"

    site_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SiteAnchor(Base):
    """Optional real-world anchor of a Site (user-set): where its origin is on Earth and how the grid is rotated."""

    __tablename__ = "site_anchors"

    site_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    rotation_deg: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Asset(Base):
    """An independent, reusable 3D asset extracted from a world entity (T-2.6).

    Retains the original entity's geometry, materials, canonical dimensions,
    and provenance back to the source world and session, leaving the original
    world untouched.
    """

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(64), default="furniture")
    source_world_id: Mapped[str] = mapped_column(String(32), index=True)
    source_entity_id: Mapped[str] = mapped_column(String(128), index=True)
    dimensions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    owner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ExportRecord(Base):
    """Record of an export artifact generated for a tenant (T-6.5 usage metering)."""

    __tablename__ = "export_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    format: Mapped[str] = mapped_column(String(32), index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class JobLog(Base):
    """Structured log record for a background job (T-6.6 observability).

    Enables deep post-mortem analysis and operational debugging of failed
    and running jobs without exposing host server tracebacks or SSH access.
    """

    __tablename__ = "job_logs"
    __table_args__ = (
        Index("ix_job_logs_job_timestamp", "job_id", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    level: Mapped[str] = mapped_column(String(16), default="INFO")
    message: Mapped[str] = mapped_column(Text)
    context_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class WorldEditCommand(Base):
    """Persistent per-world edit command log for undo/redo (T-6.7 & T-4b.1).

    Replaces volatile client-side history with an auditable, durable command stream.
    Supports move, rotate, delete, replace, surface finish, etc. Every command is reversible
    via inverse_params and advances the world's version lineage with provenance GENERATED.
    """

    __tablename__ = "world_edit_commands"
    __table_args__ = (
        Index("ix_world_edit_commands_world_undone", "world_id", "undone"),
        Index("ix_world_edit_commands_world_seq", "world_id", "sequence"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    command_type: Mapped[str] = mapped_column(String(32))  # move, rotate, delete, replace, surface_finish, transform
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    description: Mapped[str] = mapped_column(String(255))
    params: Mapped[dict] = mapped_column(JSON)
    inverse_params: Mapped[dict] = mapped_column(JSON)
    version_id_before: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version_id_after: Mapped[str | None] = mapped_column(String(64), nullable=True)
    undone: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    undone_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Tenant(Base):
    """A customer organization / firm (T-6.4). Primary isolation boundary."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserAccount(Base):
    """An authenticated user belonging to a Tenant (T-6.4)."""

    __tablename__ = "user_accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32), default="editor")
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TenantInvite(Base):
    """Pending team invite to join a Tenant (T-6.4)."""

    __tablename__ = "tenant_invites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(32), default="editor")
    token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    invited_by_user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)  # pending | accepted | revoked | expired
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class TenantSubscription(Base):
    """Tenant subscription state and Stripe billing integration (Task T-7.2)."""

    __tablename__ = "tenant_subscriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), unique=True, index=True)
    tier: Mapped[str] = mapped_column(String(32), default="free")  # free | starter | pro | enterprise
    status: Mapped[str] = mapped_column(String(32), default="active")  # active | trialing | past_due | canceled
    stripe_customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class JobStageRun(Base):
    """One attempt of one pipeline stage for a job (P103); backs checkpoint reuse and stage diagnostics."""

    __tablename__ = "job_stage_runs"
    __table_args__ = (
        UniqueConstraint("job_id", "name", "inputs_hash", "attempt", name="uq_job_stage_runs_job_name_hash_attempt"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    team_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    inputs_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    outputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    peak_rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
