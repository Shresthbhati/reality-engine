"""P106 coarse stage `compile_validate`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class CompileValidate(SegmentStage):
    spec = StageSpec(name="compile_validate", version="1", resource_class="cpu", required=True)
