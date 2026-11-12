"""P108 stage `reconstruct`: the progressive LADDER (multi-view reconstruction, falling back to the single-image
bootstrap), via the host-supplied segment (see registry.SegmentStage).

The fine-grained stages (pose, scale, depth, layout, perception, mesh, dense, detail, finalize) run INSIDE the ladder's
multi-view rung (``ladder.run_fine_stages``) and are reported by ``ladder.LadderReport`` right after this stage. The
ladder itself cannot be dissolved into independent stages: its rung choice (``progressive._multiview_weakness``) needs
the finished multi-view world, and a multi-view failure must fall back to a lower rung rather than fail the job.
"""
from engine.pipeline.stages.base import StageSpec
from engine.pipeline.stages.registry import SegmentStage


class Reconstruct(SegmentStage):
    spec = StageSpec(name="reconstruct", version="1", resource_class="gpu", required=True)
