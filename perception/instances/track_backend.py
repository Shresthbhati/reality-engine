"""Concrete ITrackBackend implementation using epipolar + appearance
multi-view identity (P7-01).

This backend associates regions across views into InstanceTracks -- the same
real-world object seen from multiple frames -- with the gated Hungarian
association (`association.resolve_hypotheses`; appearance only as a small score
term inside the geometric gate). The legacy union-find resolver is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from perception.instances.interface import ITrackBackend, InstanceTrack
from perception.segmentation.interface import SegmentationResult, SegmentedRegion
from perception.instances.lifting import lift_region_to_3d, ObjectHypothesis3D
from perception.instances.association import resolve_hypotheses
from perception.instances.object_resolution import MergedObjectCandidate
from perception.instances.appearance import compute_color_histogram, AppearanceDescriptor
from perception.instances.epipolar import epipolar_consistent
from reconstruction.calibration.camera import PinholeCamera
from provenance import Uncertainty
from world_ir.observations import DepthSource


@dataclass
class _TrackCandidate:
    """Internal candidate before conversion to InstanceTrack."""
    track_id: str
    regions: List[SegmentedRegion]
    label: str
    confidence: float
    merged_candidate: MergedObjectCandidate


class MultiViewIdentityTrackBackend(ITrackBackend):
    """Track backend using geometric (epipolar) + photometric (color
    histogram) multi-view identity.

    This is the same pipeline the perception stage uses for object
    merging, exposed as a tracking interface. It:
    - Lifts each region to 3D using metric depth + registered camera
    - Computes appearance descriptors for each region
    - Uses association.resolve_hypotheses (Hungarian; positional + appearance terms)
    - Converts MergedObjectCandidate -> InstanceTrack
    """

    def __init__(
        self,
        *,
        distance_threshold_m: float = 0.5,
        appearance_similarity_min: float = 0.3,
        epipolar_tolerance_px: float = 5.0,
    ):
        self.distance_threshold_m = distance_threshold_m
        self.appearance_similarity_min = appearance_similarity_min
        self.epipolar_tolerance_px = epipolar_tolerance_px

    def link_instances(
        self,
        results: List[SegmentationResult],
        metric_depth_maps: Dict[str, object],  # DepthFrame by evidence_id
        cameras: Dict[str, PinholeCamera],
        images: Dict[str, list],  # image list by evidence_id
    ) -> List[InstanceTrack]:
        """Associate SegmentedRegions across evidence items into InstanceTracks.

        Args:
            results: Segmentation results per evidence item
            metric_depth_maps: Metric depth frames keyed by evidence_id
            cameras: Registered cameras keyed by evidence_id
            images: RGB images (list of list of (r,g,b)) keyed by evidence_id

        Returns:
            List of InstanceTrack, each representing one real-world object
            observed across multiple views.
        """
        if not results:
            return []

        # Build hypotheses and appearance descriptors
        hypotheses = []
        appearance_by_region = {}
        region_to_seg = {}

        for seg in results:
            depth = metric_depth_maps.get(seg.evidence_id)
            if depth is None:
                continue  # No metric depth for this view
            camera = cameras.get(seg.evidence_id)
            if camera is None:
                continue  # Unregistered view
            img_list = images.get(seg.evidence_id)

            for region in seg.regions:
                # Compute appearance descriptor if image available
                if img_list is not None:
                    try:
                        desc = compute_color_histogram(img_list, region.mask)
                        if desc is not None:
                            appearance_by_region[region.region_id] = desc
                    except Exception:
                        pass  # Degrade gracefully

                # Lift to 3D
                hyp = lift_region_to_3d(region, depth, camera)
                if hyp is not None:
                    hypotheses.append(hyp)
                    region_to_seg[(hyp.region_id, hyp.evidence_id)] = region

        if not hypotheses:
            return []

        # Hungarian association. Uncertainty (not a fixed metre threshold) gates the distance, so
        # `distance_threshold_m` / `epipolar_tolerance_px` are accepted for compatibility but no longer used; no epipolar
        # check is run (decisions record geom_verified=None). Appearance histograms become embeddings.
        embeddings = {
            f"{h.evidence_id}|{h.region_id}": appearance_by_region[h.region_id].histogram
            for h in hypotheses if h.region_id in appearance_by_region
        }
        cam_pos = {
            ev: (cam.extrinsics.position.x, cam.extrinsics.position.y, cam.extrinsics.position.z)
            for ev, cam in cameras.items()
        }
        candidates, _ = resolve_hypotheses(hypotheses, DepthSource.METRIC, cam_pos, appearance=embeddings)

        # Convert to InstanceTrack
        tracks = []
        for i, cand in enumerate(candidates):
            # Collect the original regions for this candidate
            cand_regions = []
            for hyp in cand.source_hypotheses:
                region = region_to_seg.get((hyp.region_id, hyp.evidence_id))
                if region:
                    cand_regions.append(region)

            tracks.append(InstanceTrack(
                track_id=f"track-{cand.label}-{i:04d}",
                regions=cand_regions,
                label=cand.label,
                confidence=cand.confidence,
                uncertainty=Uncertainty(confidence=cand.confidence),
            ))

        return tracks


def build_images_dict(evidence_items, kind_filter="image") -> Dict[str, list]:
    """Load RGB images from evidence items into list-of-lists format.

    Returns dict mapping evidence_id -> [[(r,g,b), ...], ...]
    """
    images = {}
    try:
        import imageio.v3 as iio
        for item in evidence_items:
            if item.kind.value != kind_filter:
                continue
            try:
                img = iio.imread(item.source_uri)
                images[item.id] = [[tuple(pixel[:3]) for pixel in row] for row in img]
            except Exception:
                pass  # Loading failure -> no image for this evidence
    except ImportError:
        pass  # imageio not available
    return images