"""P106 coarse stage `evidence_resolve`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class EvidenceResolve(SegmentStage):
    spec = StageSpec(name="evidence_resolve", version="1", resource_class="cpu", required=True)
