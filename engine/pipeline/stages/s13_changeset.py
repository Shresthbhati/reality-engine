"""P106 coarse stage `changeset_commit`: thin wrapper over the host-supplied segment (see registry.SegmentStage)."""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class ChangesetCommit(SegmentStage):
    spec = StageSpec(name="changeset_commit", version="1", resource_class="cpu", required=True)
