"""Account lifecycle logic (plan T-6.4): password hashing, and erasing everything a tenant owns.

Identity model: an Account belongs to one tenant; its scoped token carries (tenant_id, account id, role). Tenant isolation
is unchanged and still decides what a token can see. ADMIN bypasses isolation, so accounts are only ever editor or viewer.

Erasure: ``erase_tenant`` permanently deletes a tenant's worlds (through the existing ``purge_world``, which also removes
WorldStore versions and only the blobs nothing else references), then every session, evidence, upload and site row it still
holds, then the blobs those pointed to when no other tenant's row shares them, then the accounts and invites themselves.
Rows are committed before any file is touched, so a failed step never costs a blob that a surviving row still needs.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Account, ActivityEvent, Evidence, EvidenceDecision, Invite, Job, Location, Notification, Session, Site, SiteAnchor,
    SiteComposite, SitePlacement, Upload, World, new_id,
)
from apps.api.storage import blob_digest, gc_blobs
from apps.api.world_lifecycle import JobsStillRunning, purge_world

MIN_PASSWORD_CHARS = 10
MAX_PASSWORD_CHARS = 200
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


def normalise_email(email: str) -> str:
    e = (email or "").strip().lower()
    if len(e) > 254 or not _EMAIL.match(e):
        raise ValueError("Enter a valid email address")
    return e


def check_password_strength(password: str) -> None:
    if not isinstance(password, str) or not (MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS):
        raise ValueError(f"Password must be {MIN_PASSWORD_CHARS}-{MAX_PASSWORD_CHARS} characters")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    return f"scrypt${salt.hex()}${hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
        return hmac.compare_digest(got, bytes.fromhex(hash_hex))
    except (ValueError, TypeError):
        return False


#: Verified against when the email is unknown, so "no such account" costs the same time as "wrong password".
DUMMY_HASH = hash_password("not-a-real-password")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _digests_of(rows) -> set[str]:
    return {d for d in (blob_digest(r.artifact_uri) for r in rows) if d}


async def erase_tenant(db: AsyncSession, tenant_id: str) -> dict:
    """Permanently delete everything ``tenant_id`` owns, accounts included. Commits.

    Raises ``JobsStillRunning`` (accounts kept, so the caller can retry) while a job of the tenant is still stopping."""
    out = {"worlds": 0, "sessions": 0, "evidence": 0, "uploads": 0, "sites": 0, "blobs_removed": 0}
    world_ids = list((await db.execute(select(World.id).where(World.team_id == tenant_id))).scalars())
    for wid in world_ids:
        world = await db.get(World, wid)
        if world is not None:
            res = await purge_world(db, world)
            out["worlds"] += 1
            out["blobs_removed"] += len(res["blobs_removed"])

    sess_ids = list((await db.execute(select(Session.id).where(Session.team_id == tenant_id))).scalars())
    ev_rows = (await db.execute(select(Evidence).where(or_(Evidence.team_id == tenant_id, Evidence.session_id.in_(sess_ids or [""]))))).scalars().all()
    up_rows = (await db.execute(select(Upload).where(or_(Upload.team_id == tenant_id, Upload.session_id.in_(sess_ids or [""]))))).scalars().all()
    site_ids = list((await db.execute(select(Site.id).where(Site.team_id == tenant_id))).scalars())
    ev_ids, up_ids = [e.id for e in ev_rows], [u.id for u in up_rows]
    candidates = _digests_of((*ev_rows, *up_rows))

    subjects = (("session", sess_ids), ("evidence", ev_ids), ("upload", up_ids), ("site", site_ids), ("world", world_ids))
    job_clause = or_(*[(Job.entity_type == t) & Job.entity_id.in_(ids or [""]) for t, ids in subjects])
    live = (await db.execute(select(Job).where(job_clause, Job.status == "running"))).scalars().all()
    if live:
        from apps.api.jobs import terminate_job_procs

        for j in live:
            j.cancel_requested = True
            terminate_job_procs(j.id)
        await db.commit()
        raise JobsStillRunning(f"job(s) {[j.id for j in live]} are still stopping; retry shortly")

    await db.execute(delete(Job).where(job_clause))
    for model in (Location, Notification, ActivityEvent):
        await db.execute(delete(model).where(or_(*[(model.entity_type == t) & model.entity_id.in_(ids or [""]) for t, ids in subjects])))
    await db.execute(delete(EvidenceDecision).where(EvidenceDecision.evidence_id.in_(ev_ids or [""])))
    await db.execute(delete(Evidence).where(Evidence.id.in_(ev_ids or [""])))
    await db.execute(delete(Upload).where(Upload.id.in_(up_ids or [""])))
    await db.execute(delete(Session).where(Session.id.in_(sess_ids or [""])))
    for model in (SitePlacement, SiteComposite, SiteAnchor):
        await db.execute(delete(model).where(model.site_id.in_(site_ids or [""])))
    await db.execute(delete(Site).where(Site.id.in_(site_ids or [""])))
    await db.execute(delete(Invite).where(Invite.tenant_id == tenant_id))
    await db.execute(delete(Account).where(Account.tenant_id == tenant_id))
    db.add(ActivityEvent(id=new_id("act"), type="tenant.erased", entity_type="tenant", entity_id=tenant_id,
                         summary="All data of this tenant was permanently deleted at the account owner's request",
                         metadata_json={"worlds": out["worlds"], "sessions": len(sess_ids), "evidence": len(ev_ids)}))
    await db.commit()
    # Rows are gone and committed; only now touch files.
    blobs = await gc_blobs(db, candidates)
    out.update(sessions=len(sess_ids), evidence=len(ev_ids), uploads=len(up_ids), sites=len(site_ids))
    out["blobs_removed"] += len(blobs["removed"])
    out["blobs_kept_shared"] = len(blobs["kept_shared"])
    return out
