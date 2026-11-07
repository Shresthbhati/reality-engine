"""Account lifecycle (plan T-6.4): signup, login, invite, members, delete-my-data.

Signup is closed unless ``REALITY_SIGNUP=open`` (a hosted deployment decides who may create a team). Signup/login/accept are
reachable without a token (auth.OPEN_PATHS), so they carry their own rate limits: per client host and per email.
"""
from __future__ import annotations

import asyncio
import math
import os
import secrets
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import accounts as acct
from apps.api.auth import AuthContext, Role, get_auth_context, issue_scoped_token, rate_limiter
from apps.api.db import get_db
from apps.api.models import Account, Invite, new_id, utcnow
from apps.api.world_lifecycle import JobsStillRunning

router = APIRouter(prefix="/api/accounts", tags=["accounts"])

INVITE_TTL = timedelta(days=7)
CONFIRM_PHRASE = "delete my data"


class Credentials(BaseModel):
    email: str
    password: str


class InviteIn(BaseModel):
    email: str
    role: str = "editor"


class AcceptIn(BaseModel):
    token: str
    password: str


class EraseIn(BaseModel):
    confirm: str = ""


def _public(a: Account) -> dict:
    return {"id": a.id, "email": a.email, "tenant_id": a.tenant_id, "role": a.role, "is_owner": a.is_owner}


def _session(a: Account) -> dict:
    return {"token": issue_scoped_token(a.tenant_id, a.id, Role(a.role)), "account": _public(a)}


def _throttle(request: Request, email: str | None = None) -> None:
    host = request.client.host if request.client else "unknown"
    checks = [(f"acct:{host}", 0.5, 10.0)] + ([(f"acct-email:{email}", 0.05, 5.0)] if email else [])
    for key, rps, burst in checks:
        ok, retry = rate_limiter.check(key, rps=rps, burst=burst)
        if not ok:
            raise HTTPException(429, "Too many attempts. Try again shortly.", headers={"Retry-After": str(max(1, math.ceil(retry)))})


def _valid(email: str, password: str) -> str:
    try:
        e = acct.normalise_email(email)
        acct.check_password_strength(password)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return e


async def _hash(password: str) -> str:
    return await asyncio.to_thread(acct.hash_password, password)


async def _owner(db: AsyncSession, auth: AuthContext) -> Account:
    a = await db.get(Account, auth.user_id)
    if a is None or a.tenant_id != auth.tenant_id:
        raise HTTPException(403, "Sign in with an account to do this")
    if not a.is_owner:
        raise HTTPException(403, "Only the team owner can do this")
    return a


@router.post("/signup", status_code=201)
async def signup(body: Credentials, request: Request, db: AsyncSession = Depends(get_db)) -> dict:
    if os.environ.get("REALITY_SIGNUP", "closed").lower() != "open":
        raise HTTPException(403, "Signup is closed on this deployment")
    email = _valid(body.email, body.password)
    _throttle(request, email)
    a = Account(id=new_id("acc"), email=email, password_hash=await _hash(body.password),
                tenant_id=f"t_{uuid.uuid4().hex[:12]}", role=Role.EDITOR.value, is_owner=True)
    db.add(a)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "An account with this email already exists") from None
    return _session(a)


@router.post("/login")
async def login(body: Credentials, request: Request, db: AsyncSession = Depends(get_db)):
    email = (body.email or "").strip().lower()
    _throttle(request, email[:254])
    a = (await db.execute(select(Account).where(Account.email == email))).scalars().first()
    ok = await asyncio.to_thread(acct.verify_password, body.password or "", a.password_hash if a else acct.DUMMY_HASH)
    if a is None or not ok:
        return JSONResponse({"detail": "Wrong email or password"}, status_code=401)
    return _session(a)


@router.get("/me")
async def me(db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    a = await db.get(Account, auth.user_id)
    if a is None or a.tenant_id != auth.tenant_id:
        raise HTTPException(404, "No account for this token")
    return _public(a)


@router.post("/invites", status_code=201)
async def create_invite(body: InviteIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    owner = await _owner(db, auth)
    if body.role not in ("editor", "viewer"):
        raise HTTPException(422, "role must be editor or viewer")
    try:
        email = acct.normalise_email(body.email)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    token = secrets.token_urlsafe(32)
    inv = Invite(id=new_id("inv"), tenant_id=owner.tenant_id, email=email, role=body.role, token_hash=acct.hash_token(token),
                 created_by=owner.id, expires_at=utcnow() + INVITE_TTL)
    db.add(inv)
    await db.commit()
    # No mail provider is wired: the owner passes the token on. It is shown here once; only its hash is stored.
    return {"invite_id": inv.id, "email": email, "role": inv.role, "token": token, "expires_at": inv.expires_at.isoformat()}


@router.post("/invites/accept", status_code=201)
async def accept_invite(body: AcceptIn, request: Request, db: AsyncSession = Depends(get_db)) -> dict:
    _throttle(request)
    inv = (await db.execute(select(Invite).where(Invite.token_hash == acct.hash_token(body.token or "")))).scalars().first()
    if inv is None or inv.accepted_at is not None or inv.expires_at.replace(tzinfo=None) < utcnow().replace(tzinfo=None):
        raise HTTPException(410, "This invitation is invalid, used or expired")
    email = _valid(inv.email, body.password)
    a = Account(id=new_id("acc"), email=email, password_hash=await _hash(body.password), tenant_id=inv.tenant_id,
                role=inv.role, is_owner=False)
    inv.accepted_at = utcnow()
    db.add(a)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "An account with this email already exists") from None
    return _session(a)


@router.get("/members")
async def members(db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    owner = await _owner(db, auth)
    rows = (await db.execute(select(Account).where(Account.tenant_id == owner.tenant_id).order_by(Account.created_at))).scalars().all()
    return {"items": [_public(a) for a in rows]}


@router.delete("/members/{account_id}")
async def remove_member(account_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    owner = await _owner(db, auth)
    target = await db.get(Account, account_id)
    if target is None or target.tenant_id != owner.tenant_id:
        raise HTTPException(404, "Member not found")
    if target.id == owner.id:
        raise HTTPException(409, "The owner cannot be removed; delete the team data instead")
    await db.delete(target)
    await db.commit()
    return {"removed": account_id}


@router.delete("/me")
async def delete_my_data(body: EraseIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    """A member deletes their own account (the team's data stays: rows belong to the team). The owner deletes the whole team:
    every world, session, evidence, upload, site, artifact no one else shares, and every account."""
    a = await db.get(Account, auth.user_id)
    if a is None or a.tenant_id != auth.tenant_id:
        raise HTTPException(403, "Sign in with an account to do this")
    if body.confirm.strip().lower() != CONFIRM_PHRASE:
        raise HTTPException(422, f'Send {{"confirm": "{CONFIRM_PHRASE}"}} to confirm; this cannot be undone')
    if not a.is_owner:
        await db.delete(a)
        await db.commit()
        return {"deleted": "account", "account_id": a.id}
    try:
        summary = await acct.erase_tenant(db, a.tenant_id)
    except JobsStillRunning as exc:
        raise HTTPException(409, str(exc)) from None
    return {"deleted": "team", "tenant_id": a.tenant_id, **summary}
