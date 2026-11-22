"""P106 coarse stage `compile_validate`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class CompileValidate(SegmentStage):
    # a validated world always exists when this stage returns (it raises otherwise): it is stored, its hash published and REQUIRED
    spec = StageSpec(name="compile_validate", version="2", resource_class="cpu", required=True, requires_outputs=("world_ir",))
