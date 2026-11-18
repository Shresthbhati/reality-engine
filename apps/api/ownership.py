"""Row ownership that is derived from evidence, never guessed.

An ``ActivityEvent`` describes a subject (``entity_type``, ``entity_id``). Its owner is the subject's owner, read at
the moment the event is written -- before a delete or purge can remove the subject -- so an audit event about a
deleted entity stays visible to the team it belongs to and to nobody else.

A subject that cannot be resolved (unknown type, row already gone, subject itself unowned) leaves the event unowned
(``team_id`` NULL): hidden from every tenant, visible to system administrators. Nothing here defaults an owner.
"""
from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession

from apps.api.models import ActivityEvent, Asset, Evidence, Job, Session, Site, Upload, World

#: ``entity_type`` value -> model carrying ``team_id``.
SUBJECT_MODELS = {
    "world": World, "session": Session, "evidence": Evidence, "upload": Upload, "site": Site, "job": Job, "asset": Asset,
}


def resolve_owner(session: OrmSession, entity_type: str | None, entity_id: str | None) -> str | None:
    """The owner of the subject, or None when it cannot be established. A subject created in the same unit of work
    (or one being deleted by it, which the identity map no longer returns) is found among the pending objects;
    otherwise the identity map and then the database are consulted."""
    model = SUBJECT_MODELS.get(entity_type or "")
    if model is None or not entity_id:
        return None
    for obj in (*session.new, *session.deleted):  # created or being deleted in this same unit of work
        if isinstance(obj, model) and obj.id == entity_id:
            return obj.team_id
    row = session.get(model, entity_id)
    return row.team_id if row is not None else None


@event.listens_for(OrmSession, "before_flush")
def _stamp_activity_owner(session: OrmSession, flush_context, instances) -> None:
    for obj in list(session.new):
        if isinstance(obj, ActivityEvent) and obj.team_id is None:
            with session.no_autoflush:
                obj.team_id = resolve_owner(session, obj.entity_type, obj.entity_id)
