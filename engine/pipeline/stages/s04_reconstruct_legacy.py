"""P106 coarse stage `reconstruct_legacy`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class ReconstructLegacy(SegmentStage):
    spec = StageSpec(name="reconstruct_legacy", version="1", resource_class="gpu", required=True)
