"""Fine-grained architectural fixture inspection and 3D surface localization (Task 3.4, Epic E05-1).

Provides:
- Orthorectified high-resolution surface patch extraction from planar walls and ceilings.
- Fine-grained open-vocabulary fixture detection (electrical, plumbing, mechanical).
- 3D wall-surface back-projection and spatial bounding box calculation.
- Multi-view 3D association and non-maximum suppression.
"""

from .detector import DetectedFixture, FixtureCategory, FixtureDetector
from .patch_extractor import OrthorectifiedPatchExtractor, WallPatch

__all__ = [
    "DetectedFixture",
    "FixtureCategory",
    "FixtureDetector",
    "OrthorectifiedPatchExtractor",
    "WallPatch",
]
