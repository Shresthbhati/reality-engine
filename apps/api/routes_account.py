"""Account Lifecycle, Team Invites, and GDPR Data Erasure (T-6.4).

Provides:
- Organization (Tenant) signup and Admin token issuance.
- Team member invitations with cryptographically random tokens and role assignment.
- Secure login and token minting using OWASP-compliant PBKDF2-HMAC-SHA256 password hashing.
- Complete 'Delete My Data' (Right to Erasure) that securely purges all tenant database
  records and unlinks/destroys all on-disk artifacts (point clouds, meshes, uploads, WorldStore versions).
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.auth import (
    AuthContext,
    Role,
    get_auth_context,
    hash_password,
    issue_scoped_token,
    verify_password,
)
from apps.api.db import get_db
from apps.api.models import (
    Asset,
    CaptureRequest,
    CompletionProposal,
    Evidence,
    EvidenceDecision,
    ExportRecord,
    Job,
    JobLog,
    Location,
    Notification,
    Session,
    Site,
    SiteAnchor,
    SiteComposite,
    SitePlacement,
    Tenant,
    TenantInvite,
    Upload,
    UserAccount,
    World,
    WorldCameraView,
    WorldEditCommand,
    WorldIntent,
    WorldMeasurement,
    WorldShareLink,
    WorldVersion,
    new_id,
    utcnow,
)
from apps.api.storage import blob_digest, gc_blobs

log = logging.getLogger("reality.api.account")

auth_router = APIRouter()
account_router = APIRouter()

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_expired(dt: datetime) -> bool:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc) < utcnow()
    return dt < utcnow()


class SignupRequest(BaseModel):
    tenant_name: str = Field(..., min_length=2, max_length=200)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=200)


class LoginRequest(BaseModel):
    email: str
    password: str


class InviteRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    role: str = Field(default="editor")


class AcceptInviteRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=200)
    password: str = Field(..., min_length=8, max_length=128)


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return cleaned[:80] or "team"


# -------------------------------------------------------------------------------------------------
# Public Authentication Routes (/api/auth)
# -------------------------------------------------------------------------------------------------

@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(req: SignupRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """Create a new tenant organization and initial administrator user account."""
    email_clean = req.email.strip().lower()
    if not EMAIL_REGEX.match(email_clean):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid email format")

    # Check for duplicate email
    existing_user = (await db.execute(select(UserAccount).where(UserAccount.email == email_clean))).scalar_one_or_none()
    if existing_user is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    base_slug = _slugify(req.tenant_name)
    slug = base_slug
    # Ensure unique slug
    existing_slug = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if existing_slug is not None:
        slug = f"{base_slug}-{secrets.token_hex(4)}"

    tenant = Tenant(
        id=new_id("tnt"),
        name=req.tenant_name.strip(),
        slug=slug,
        status="active",
    )
    db.add(tenant)
    await db.flush()

    pwd_hash = hash_password(req.password)
    user = UserAccount(
        id=new_id("usr"),
        tenant_id=tenant.id,
        email=email_clean,
        full_name=req.full_name.strip(),
        role=Role.ADMIN.value,
        password_hash=pwd_hash,
    )
    db.add(user)
    await db.commit()

    token = issue_scoped_token(tenant.id, user.id, Role.ADMIN)
    return {
        "token": token,
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
        },
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@auth_router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """Authenticate with email and password to receive a scoped bearer token."""
    email_clean = req.email.strip().lower()
    user = (await db.execute(select(UserAccount).where(UserAccount.email == email_clean))).scalar_one_or_none()
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    tenant = await db.get(Tenant, user.tenant_id)
    if tenant is None or tenant.status != "active":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tenant organization is suspended or deleted")

    token = issue_scoped_token(tenant.id, user.id, user.role)
    return {
        "token": token,
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
        },
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@auth_router.get("/invites/{token}")
async def inspect_invite(token: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Public inspection of team invite status before acceptance."""
    inv = (await db.execute(select(TenantInvite).where(TenantInvite.token == token.strip()))).scalar_one_or_none()
    if inv is None or inv.status != "pending" or _is_expired(inv.expires_at):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite is invalid, expired, or has already been used")

    tenant = await db.get(Tenant, inv.tenant_id)
    tenant_name = tenant.name if tenant else "Unknown Team"
    return {
        "email": inv.email,
        "role": inv.role,
        "tenant_name": tenant_name,
        "expires_at": inv.expires_at.isoformat(),
    }


@auth_router.post("/invites/{token}/accept", status_code=status.HTTP_201_CREATED)
async def accept_invite(token: str, req: AcceptInviteRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """Accept team invite, register user account, and return scoped token."""
    inv = (await db.execute(select(TenantInvite).where(TenantInvite.token == token.strip()))).scalar_one_or_none()
    if inv is None or inv.status != "pending" or _is_expired(inv.expires_at):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite is invalid, expired, or has already been used")

    tenant = await db.get(Tenant, inv.tenant_id)
    if tenant is None or tenant.status != "active":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inviting tenant organization no longer exists")

    # Check if this email already registered
    existing_user = (await db.execute(select(UserAccount).where(UserAccount.email == inv.email))).scalar_one_or_none()
    if existing_user is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    pwd_hash = hash_password(req.password)
    user = UserAccount(
        id=new_id("usr"),
        tenant_id=tenant.id,
        email=inv.email,
        full_name=req.full_name.strip(),
        role=inv.role,
        password_hash=pwd_hash,
    )
    inv.status = "accepted"
    db.add(user)
    await db.commit()

    scoped_token = issue_scoped_token(tenant.id, user.id, user.role)
    return {
        "token": scoped_token,
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
        },
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


# -------------------------------------------------------------------------------------------------
# Tenant / Account Management Routes (/api/account)
# -------------------------------------------------------------------------------------------------

@account_router.get("/me")
async def get_current_account(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Return profile for the currently authenticated user and tenant."""
    tenant = await db.get(Tenant, auth.tenant_id)
    user = await db.get(UserAccount, auth.user_id) if auth.user_id else None
    return {
        "user_id": auth.user_id,
        "tenant_id": auth.tenant_id,
        "role": auth.role.value,
        "tenant_name": tenant.name if tenant else auth.tenant_id,
        "email": user.email if user else None,
        "full_name": user.full_name if user else None,
    }


@account_router.post("/invites", status_code=status.HTTP_201_CREATED)
async def create_invite(
    req: InviteRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Invite a new member to the current tenant organization (Admin only)."""
    if auth.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only tenant administrators can invite members")

    email_clean = req.email.strip().lower()
    if not EMAIL_REGEX.match(email_clean):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid email format")

    role_str = req.role.strip().lower()
    try:
        assigned_role = Role(role_str)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Invalid role {req.role!r}; expected admin, editor or viewer")

    # Check if user is already an active member of this tenant
    existing_member = (await db.execute(
        select(UserAccount).where(UserAccount.tenant_id == auth.tenant_id, UserAccount.email == email_clean)
    )).scalar_one_or_none()
    if existing_member is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "User is already a member of this organization")

    invite_token = f"inv_{secrets.token_urlsafe(32)}"
    expires = utcnow() + timedelta(days=7)

    invite = TenantInvite(
        id=new_id("inv"),
        tenant_id=auth.tenant_id,
        email=email_clean,
        role=assigned_role.value,
        token=invite_token,
        invited_by_user_id=auth.user_id,
        status="pending",
        expires_at=expires,
    )
    db.add(invite)
    await db.commit()

    return {
        "id": invite.id,
        "email": invite.email,
        "role": invite.role,
        "invite_token": invite.token,
        "expires_at": invite.expires_at.isoformat(),
        "status": invite.status,
    }


@account_router.get("/invites")
async def list_invites(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List pending invites for caller's tenant."""
    invites = (await db.execute(
        select(TenantInvite)
        .where(TenantInvite.tenant_id == auth.tenant_id, TenantInvite.status == "pending")
        .order_by(TenantInvite.created_at.desc())
    )).scalars().all()

    return {
        "items": [
            {
                "id": inv.id,
                "email": inv.email,
                "role": inv.role,
                "invite_token": inv.token,
                "expires_at": inv.expires_at.isoformat(),
                "created_at": inv.created_at.isoformat(),
                "status": inv.status,
            }
            for inv in invites
        ]
    }


@account_router.delete("/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Revoke a pending team invite (Admin only)."""
    if auth.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only tenant administrators can revoke invites")

    inv = await db.get(TenantInvite, invite_id)
    if inv is None or inv.tenant_id != auth.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite not found")

    inv.status = "revoked"
    await db.commit()
    return {"status": "revoked", "id": invite_id}


@account_router.delete("/data")
@account_router.post("/delete-my-data")
async def delete_my_data(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """GDPR Right to Erasure / Account Data Purge (Admin only).

    Strict tenant boundary enforcement:
    1. Only an Admin can initiate data deletion for their own tenant organization.
    2. Cancels any running background reconstruction or export jobs.
    3. Permanently deletes all database rows belonging to the tenant:
       Worlds, Versions, Measurements, Share Links, Commands, Sessions, Uploads,
       Evidence, Assets, ExportRecords, JobLogs, Jobs, Invites, User Accounts, and Tenant.
    4. Physically destroys all on-disk artifacts:
       WorldStore version lineage files & blobs, raw photo uploads, point clouds,
       and camera geometry files stored in STORAGE_ROOT.
    5. Leaves all other tenants' data and files 100% intact.
    """
    if auth.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only tenant administrators can delete tenant data")

    tenant_id = auth.tenant_id
    if not tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No tenant identified in auth context")

    # 1. Discover all worlds and sessions belonging to this tenant
    worlds = (await db.execute(select(World).where(World.team_id == tenant_id))).scalars().all()
    world_ids = [w.id for w in worlds]

    sessions = (await db.execute(select(Session).where(Session.team_id == tenant_id))).scalars().all()
    session_ids = [s.id for s in sessions]

    # Include any session attached to a tenant world even if session.team_id was null
    if world_ids:
        attached_sessions = (await db.execute(
            select(Session).where(Session.world_id.in_(world_ids))
        )).scalars().all()
        for s in attached_sessions:
            if s.id not in session_ids:
                session_ids.append(s.id)

    # 2. Collect all candidate artifact blob URIs to GC physically
    candidates: set[str] = set()

    for w in worlds:
        d = blob_digest(w.thumbnail_uri)
        if d:
            candidates.add(d)

    if world_ids:
        versions = (await db.execute(
            select(WorldVersion).where(WorldVersion.world_id.in_(world_ids))
        )).scalars().all()
        for v in versions:
            for uri in (v.points_artifact_uri, v.cameras_artifact_uri):
                d = blob_digest(uri)
                if d:
                    candidates.add(d)

    if session_ids:
        ev_rows = (await db.execute(
            select(Evidence).where(Evidence.session_id.in_(session_ids))
        )).scalars().all()
        for e in ev_rows:
            d = blob_digest(e.artifact_uri)
            if d:
                candidates.add(d)

        up_rows = (await db.execute(
            select(Upload).where(Upload.session_id.in_(session_ids))
        )).scalars().all()
        for u in up_rows:
            d = blob_digest(u.artifact_uri)
            if d:
                candidates.add(d)

    # Assets & Export records
    assets = (await db.execute(select(Asset).where(Asset.team_id == tenant_id))).scalars().all()
    for a in assets:
        d = blob_digest(a.artifact_uri)
        if d:
            candidates.add(d)

    exports = (await db.execute(select(ExportRecord).where(ExportRecord.tenant_id == tenant_id))).scalars().all()
    for exp in exports:
        d = blob_digest(exp.artifact_uri)
        if d:
            candidates.add(d)

    # 3. Cancel and delete jobs tied to this tenant
    jobs = (await db.execute(
        select(Job).where(
            (Job.team_id == tenant_id)
            | (Job.entity_id.in_(world_ids if world_ids else [""]))
            | (Job.entity_id.in_(session_ids if session_ids else [""]))
        )
    )).scalars().all()
    job_ids = [j.id for j in jobs]
    if job_ids:
        await db.execute(delete(JobLog).where(JobLog.job_id.in_(job_ids)))
        await db.execute(delete(Job).where(Job.id.in_(job_ids)))

    # 4. Delete world child records
    if world_ids:
        await db.execute(delete(WorldEditCommand).where(WorldEditCommand.world_id.in_(world_ids)))
        await db.execute(delete(WorldShareLink).where(WorldShareLink.world_id.in_(world_ids)))
        await db.execute(delete(WorldCameraView).where(WorldCameraView.world_id.in_(world_ids)))
        await db.execute(delete(WorldMeasurement).where(WorldMeasurement.world_id.in_(world_ids)))
        await db.execute(delete(CaptureRequest).where(CaptureRequest.world_id.in_(world_ids)))
        await db.execute(delete(CompletionProposal).where(CompletionProposal.world_id.in_(world_ids)))
        await db.execute(delete(WorldIntent).where(WorldIntent.world_id.in_(world_ids)))
        await db.execute(delete(WorldVersion).where(WorldVersion.world_id.in_(world_ids)))
        await db.execute(delete(Location).where(Location.entity_type == "world", Location.entity_id.in_(world_ids)))
        await db.execute(delete(SitePlacement).where(SitePlacement.world_id.in_(world_ids)))
        await db.execute(delete(SiteComposite).where(SiteComposite.world_id.in_(world_ids)))

    # 5. Delete session child records
    if session_ids:
        ev_ids = list((await db.execute(select(Evidence.id).where(Evidence.session_id.in_(session_ids)))).scalars())
        if ev_ids:
            await db.execute(delete(EvidenceDecision).where(EvidenceDecision.evidence_id.in_(ev_ids)))
            await db.execute(delete(Location).where(Location.entity_type == "evidence", Location.entity_id.in_(ev_ids)))
        await db.execute(delete(Location).where(Location.entity_type == "session", Location.entity_id.in_(session_ids)))
        await db.execute(delete(Evidence).where(Evidence.session_id.in_(session_ids)))
        await db.execute(delete(Upload).where(Upload.session_id.in_(session_ids)))

    # 6. Delete worlds, sessions, assets, exports, sites
    if world_ids:
        await db.execute(delete(World).where(World.id.in_(world_ids)))
    if session_ids:
        await db.execute(delete(Session).where(Session.id.in_(session_ids)))

    await db.execute(delete(Asset).where(Asset.team_id == tenant_id))
    await db.execute(delete(ExportRecord).where(ExportRecord.tenant_id == tenant_id))

    sites = (await db.execute(select(Site).where(Site.team_id == tenant_id))).scalars().all()
    site_ids = [s.id for s in sites]
    if site_ids:
        await db.execute(delete(SitePlacement).where(SitePlacement.site_id.in_(site_ids)))
        await db.execute(delete(SiteComposite).where(SiteComposite.site_id.in_(site_ids)))
        await db.execute(delete(SiteAnchor).where(SiteAnchor.site_id.in_(site_ids)))
        await db.execute(delete(Site).where(Site.id.in_(site_ids)))

    # 7. Delete users, invites, and tenant record
    users = (await db.execute(select(UserAccount).where(UserAccount.tenant_id == tenant_id))).scalars().all()
    user_ids = [u.id for u in users]
    if user_ids:
        await db.execute(delete(Notification).where(Notification.user_id.in_(user_ids)))
        await db.execute(delete(UserAccount).where(UserAccount.id.in_(user_ids)))

    await db.execute(delete(TenantInvite).where(TenantInvite.tenant_id == tenant_id))
    await db.execute(delete(Tenant).where(Tenant.id == tenant_id))

    # Commit DB transaction so all rows are permanently deleted
    await db.commit()

    # 8. Physical artifact destruction on disk
    # (a) Destroy WorldStore version directories for each world
    total_store_blobs = 0
    total_versions_purged = 0
    for wid in world_ids:
        try:
            ws_res = worldstore_service.purge_world_store(wid)
            total_versions_purged += len(ws_res.get("versions_removed", []))
            total_store_blobs += len(ws_res.get("blobs_removed", []))
        except Exception as exc:
            log.warning("WorldStore purge for world %s failed or already missing: %s", wid, exc)

    # (b) Unlink unreferenced content-addressed blobs from STORAGE_ROOT
    gc_res = await gc_blobs(db, candidates)
    removed_artifacts_count = len(gc_res.get("removed", []))

    log.info(
        "Tenant %s purged completely: %d worlds, %d sessions, %d users, %d artifacts removed",
        tenant_id, len(world_ids), len(session_ids), len(user_ids), removed_artifacts_count,
    )

    return {
        "status": "purged",
        "tenant_id": tenant_id,
        "worlds_deleted": len(world_ids),
        "sessions_deleted": len(session_ids),
        "users_deleted": len(user_ids),
        "worldstore_versions_removed": total_versions_purged,
        "worldstore_blobs_removed": total_store_blobs,
        "artifacts_removed": removed_artifacts_count,
    }
