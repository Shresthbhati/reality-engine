"""Perception Specialist Engines (Universal Spatial Intelligence Directive Phase 7).

Integrates specialized perception, segmentation, identity association, and active capture guidance
into the autonomous refinement loop.
"""

from __future__ import annotations

from engine.quality.autonomous_correction_loop import RefinementKind
from engine.quality.refinement_engines import register_engine, unregister_engine

from .more_views import more_views_specialist
from .reassociate import reassociate_specialist
from .redetect_crop import redetect_crop_specialist
from .segment import segment_specialist

__all__ = [
    "more_views_specialist",
    "reassociate_specialist",
    "redetect_crop_specialist",
    "segment_specialist",
    "register_all_specialist_engines",
    "unregister_all_specialist_engines",
]


def register_all_specialist_engines() -> None:
    """Register all 4 specialist perception engines in the RERES refinement loop."""
    register_engine(RefinementKind.REDETECT_CROP, redetect_crop_specialist)
    register_engine(RefinementKind.SEGMENT, segment_specialist)
    register_engine(RefinementKind.REASSOCIATE, reassociate_specialist)
    register_engine(RefinementKind.MORE_VIEWS, more_views_specialist)


def unregister_all_specialist_engines() -> None:
    """Unregister all specialist perception engines."""
    unregister_engine(RefinementKind.REDETECT_CROP)
    unregister_engine(RefinementKind.SEGMENT)
    unregister_engine(RefinementKind.REASSOCIATE)
    unregister_engine(RefinementKind.MORE_VIEWS)
