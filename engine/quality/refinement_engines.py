"""Where specialist engines plug into the RERES refinement loop.

The platform owns the loop, its budget and the commit rules; a specialist (detector, SAM, association) owns what one kind of
refinement task actually does. An engine is ``engine(task, world) -> bool | None`` with the executor contract of
``AutonomousCorrectionLoop.run_refinement_loop``: True = changed the world, False = tried and changed nothing, None = could not
run. Nothing is registered by default, so ``redetect_crop`` and ``segment`` are reported unavailable until someone registers a
real engine; the loop never fakes them.
"""
from __future__ import annotations

from collections.abc import Callable

from engine.quality.autonomous_correction_loop import (
    RefinementKind,
    RefinementTask,
    builtin_refinement_executor,
)

Engine = Callable[[RefinementTask, object], "bool | None"]
_ENGINES: dict[RefinementKind, Engine] = {}


def register_engine(kind: RefinementKind, engine: Engine) -> None:
    """Register (or replace) the engine that runs ``kind`` tasks."""
    _ENGINES[RefinementKind(kind)] = engine


def unregister_engine(kind: RefinementKind) -> None:
    _ENGINES.pop(RefinementKind(kind), None)


def registered_kinds() -> list[str]:
    return sorted(k.value for k in _ENGINES)


def registered_executor(task: RefinementTask, world):
    """Registered engine for the task's kind, else the built-in executor (which answers None for kinds it cannot do)."""
    engine = _ENGINES.get(task.kind)
    return engine(task, world) if engine is not None else builtin_refinement_executor(task, world)
