"""Multi-view intelligence module for Reality Engine.

Provides multi-view object identity, room-photo association, and cross-view spatial reasoning.
"""

from perception.multiview.room_associator import (
    CameraPose3D,
    MultiViewRoomAssociator,
    PhotoRoomAssociation,
    RoomBoundary,
)

__all__ = [
    "CameraPose3D",
    "MultiViewRoomAssociator",
    "PhotoRoomAssociation",
    "RoomBoundary",
]
