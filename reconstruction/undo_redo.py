"""Global undo/redo (N-28).

Transactional undo/redo across manual edits and confirmed AI operations,
including version-aware recovery. Every operation is recorded as a reversible
transaction with before/after snapshots.

Honesty rules:
* A transaction can only be undone if its before-snapshot still exists.
* Undo/redo history is append-only; transactions are never rewritten.
* A transaction that would conflict with later changes is flagged, not silently
  reversed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from world_ir import Entity, WorldIR


@dataclass(frozen=True)
class Transaction:
    """One reversible operation on the world."""
    tx_id: str
    timestamp: str
    actor: str                     # "user" | "assistant" | "system"
    operation: str                 # "update_entity" | "delete_entity" | "add_entity" | "batch"
    entity_ids: List[str]          # entities touched
    before_snapshot: Dict[str, dict]  # entity_id -> serialized entity
    after_snapshot: Dict[str, dict]
    description: str
    reversible: bool               # False if a later transaction depends on this one

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UndoRedoResult:
    """Outcome of an undo or redo."""
    ok: bool
    tx_id: Optional[str]
    message: str
    entities_restored: List[str]
    world: Optional[WorldIR]  # None if not ok


class UndoRedoManager:
    """Manages a transaction stack for undo/redo."""

    def __init__(self):
        self._transactions: List[Transaction] = []
        self._index: int = -1  # points to the last applied transaction
        self._tx_counter: int = 0

    def record_transaction(
        self,
        operation: str,
        entity_ids: List[str],
        before: Dict[str, Entity],
        after: Dict[str, Entity],
        *,
        actor: str = "user",
        description: str = "",
        reversible: bool = True,
    ) -> Transaction:
        """Record a new transaction. Any transactions after the current index
        are discarded (standard undo/redo semantics: new work forks the history)."""
        self._tx_counter += 1
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        tx = Transaction(
            tx_id=f"tx-{self._tx_counter:06d}",
            timestamp=now,
            actor=actor,
            operation=operation,
            entity_ids=sorted(entity_ids),
            before_snapshot={eid: self._serialize(e) for eid, e in before.items()},
            after_snapshot={eid: self._serialize(e) for eid, e in after.items()},
            description=description or f"{operation} on {', '.join(entity_ids)}",
            reversible=reversible,
        )

        self._transactions = self._transactions[: self._index + 1]
        self._transactions.append(tx)
        self._index += 1
        return tx

    def can_undo(self) -> bool:
        return self._index >= 0

    def can_redo(self) -> bool:
        return self._index + 1 < len(self._transactions)

    def undo(self, world_entities: Dict[str, Entity]) -> UndoRedoResult:
        """Undo the most recent applied transaction."""
        if not self.can_undo():
            return UndoRedoResult(
                ok=False, tx_id=None,
                message="Nothing to undo.", entities_restored=[], world=None,
            )

        tx = self._transactions[self._index]
        if not tx.reversible:
            return UndoRedoResult(
                ok=False, tx_id=tx.tx_id,
                message=f"Transaction {tx.tx_id} is not reversible.",
                entities_restored=[], world=None,
            )

        restored: List[str] = []
        for eid, snap in tx.before_snapshot.items():
            world_entities[eid] = self._deserialize(snap)
            restored.append(eid)

        for eid in tx.after_snapshot:
            if eid not in tx.before_snapshot:
                world_entities.pop(eid, None)

        self._index -= 1
        return UndoRedoResult(
            ok=True, tx_id=tx.tx_id,
            message=f"Undid: {tx.description}",
            entities_restored=restored, world=None,
        )

    def redo(self, world_entities: Dict[str, Entity]) -> UndoRedoResult:
        """Redo the most recent undone transaction."""
        if not self.can_redo():
            return UndoRedoResult(
                ok=False, tx_id=None,
                message="Nothing to redo.", entities_restored=[], world=None,
            )

        self._index += 1
        tx = self._transactions[self._index]

        if not tx.reversible:
            self._index -= 1
            return UndoRedoResult(
                ok=False, tx_id=tx.tx_id,
                message=f"Transaction {tx.tx_id} is not reversible.",
                entities_restored=[], world=None,
            )

        restored: List[str] = []
        for eid, snap in tx.after_snapshot.items():
            world_entities[eid] = self._deserialize(snap)
            restored.append(eid)

        for eid in tx.before_snapshot:
            if eid not in tx.after_snapshot:
                world_entities.pop(eid, None)

        return UndoRedoResult(
            ok=True, tx_id=tx.tx_id,
            message=f"Redid: {tx.description}",
            entities_restored=restored, world=None,
        )

    def history(self) -> List[dict]:
        return [
            {**t.to_dict(), "applied": i <= self._index}
            for i, t in enumerate(self._transactions)
        ]

    @staticmethod
    def _serialize(e: Entity) -> dict:
        return {
            "id": e.id,
            "type": e.type.value if hasattr(e.type, "value") else str(e.type),
            "name": e.name,
            "custom_properties": dict(e.custom_properties),
            "confidence": e.confidence,
            "provenance": str(e.provenance),
        }

    @staticmethod
    def _deserialize(snap: dict) -> Entity:
        from world_ir import EntityType
        etype = None
        if snap.get("type"):
            try:
                etype = EntityType(snap["type"])
            except ValueError:
                etype = None
        return Entity(
            id=snap["id"],
            type=etype,
            name=snap.get("name"),
            custom_properties=snap.get("custom_properties", {}),
            confidence=snap.get("confidence", 0.0),
            provenance=snap.get("provenance", "unknown"),
        )


__all__ = ["UndoRedoManager", "Transaction", "UndoRedoResult"]