"""P106 coarse stage `evidence_quality`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class EvidenceQuality(SegmentStage):
    spec = StageSpec(name="evidence_quality", version="1", resource_class="cpu", required=True)
