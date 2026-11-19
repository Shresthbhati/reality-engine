"""Engine registry: every engine's identity, needs, licence, limits and a REAL availability probe.

Availability is never inferred from "the package imports". Each engine supplies a probe that must return what it verified
(``Availability.evidence``); the six engines the repo can actually probe today reuse ``perception.availability`` (dependency
present AND checkpoint cached AND binary on PATH). A probe that raises is reported BLOCKED with the exception, never AVAILABLE.

``integration`` says how far the repository really goes with an engine: ``registry-only`` (declared, no adapter), ``adapter``
(an ``EngineAdapter`` exists) or ``wired`` (the production job path calls it). Nothing in this file creates an adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Tuple

from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    CheckpointRequirement,
    EngineSpec,
    HardwareRequirement,
    Task,
)


class RegistryError(ValueError):
    pass


class EngineRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, EngineSpec] = {}

    # ------------------------------------------------------------ registration
    def register(self, spec: EngineSpec) -> None:
        if spec.engine_id in self._specs:
            raise RegistryError(f"engine {spec.engine_id!r} is already registered")
        self._specs[spec.engine_id] = spec

    def get(self, engine_id: str) -> EngineSpec:
        try:
            return self._specs[engine_id]
        except KeyError:
            raise RegistryError(f"unknown engine {engine_id!r}") from None

    def all(self) -> List[EngineSpec]:
        return sorted(self._specs.values(), key=lambda s: (s.priority, s.engine_id))

    # ------------------------------------------------------------ availability
    def availability(self, engine_id: str) -> Availability:
        spec = self.get(engine_id)
        try:
            a = spec.probe()
        except Exception as exc:  # noqa: BLE001 -- a broken probe is a BLOCKED engine with the real error
            return Availability(
                AvailabilityStatus.BLOCKED,
                f"availability probe raised {type(exc).__name__}: {exc}",
                "fix the probe or the environment it checks",
            )
        if not isinstance(a, Availability):
            return Availability(
                AvailabilityStatus.BLOCKED, "availability probe returned no Availability record"
            )
        return a

    def candidates(self, task: Task) -> List[Tuple[EngineSpec, Availability]]:
        """Every engine that declares ``task``, with its REAL availability, in priority order."""
        return [(s, self.availability(s.engine_id)) for s in self.all() if task in s.tasks]

    def available(self, task: Task) -> List[EngineSpec]:
        return [s for s, a in self.candidates(task) if a.ok]

    def unavailable(self, task: Task) -> List[Tuple[EngineSpec, Availability]]:
        return [(s, a) for s, a in self.candidates(task) if not a.ok]

    def report(self) -> List[dict]:
        return [s.to_dict(self.availability(s.engine_id)) for s in self.all()]

    def families(self, engine_ids: Iterable[str]) -> Dict[str, List[str]]:
        """Group engines by correlation family. Engines in one family share a base model / training lineage and must not be
        counted as independent evidence by consensus. An engine with no declared family is its own family."""
        out: Dict[str, List[str]] = {}
        for eid in engine_ids:
            spec = self.get(eid)
            out.setdefault(spec.family or spec.engine_id, []).append(eid)
        return out


# ------------------------------------------------------------------ the engines this repository can really probe today


def _probe_from_perception(name: str):
    def probe() -> Availability:
        from perception.availability import real_models_by_name

        m = real_models_by_name()[name]
        if m.status == "available":
            return Availability(AvailabilityStatus.AVAILABLE, m.reason, evidence=(m.reason,))
        return Availability(AvailabilityStatus.BLOCKED, m.reason, m.remediation)

    return probe


def _licence(model_key: str) -> Tuple[str, bool | None]:
    from perception import model_registry as mr

    spec = mr.all_specs().get(model_key)
    return (spec.license, spec.commercial_ok) if spec else ("", None)


def default_registry(include_extended: bool = False) -> EngineRegistry:
    """The registry of engines declared in Reality Engine.

    When include_extended is False, returns the 6 baseline engines verified by Sonnet's
    foundation tests. When True, returns the full suite of maximum-fidelity engines.
    """
    if include_extended:
        return maximum_fidelity_registry()

    reg = EngineRegistry()
    lic = {
        k: _licence(k)
        for k in (
            "midas_dpt_hybrid",
            "depth_anything_v2",
            "owlvit_base_patch32",
            "sam_vit_b",
            "maskrcnn_resnet50_fpn",
        )
    }
    reg.register(
        EngineSpec(
            engine_id="colmap",
            name="COLMAP (SfM + patch-match MVS)",
            version="cli",
            tasks=frozenset({Task.SFM, Task.MVS}),
            input_schema="images -> camera poses, sparse points; dense fused cloud",
            output_schema="camera_pose, sparse_point, dense_point",
            probe=_probe_from_perception("colmap"),
            family="colmap",
            dependencies=("colmap binary on PATH",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="BSD-3-Clause",
            commercial_ok=True,
            redistribution="binary is not shipped by this repository",
            limitations=(
                "dense MVS needs a CUDA build of COLMAP",
                "sparse stage on 10 photos took 79.5 s and produced 115 points "
                "(datasets/south_building/runs/pipeline_measurement_20261118T044709.json)",
            ),
            tests=("tests/test_colmap_session.py", "tests/test_colmap_backend_dense.py"),
            benchmarks=("datasets/south_building/runs/pipeline_measurement_20261118T044709.json",),
            priority=10,
            integration="wired",
        )
    )
    for eid, model_key, name, task, family, integ, lims in (
        (
            "midas",
            "midas_dpt_hybrid",
            "MiDaS DPT_Hybrid",
            Task.DEPTH,
            "midas",
            "wired",
            ("relative depth: needs a scale anchor for metric use",),
        ),
        (
            "depth_anything",
            "depth_anything_v2",
            "Depth Anything V2",
            Task.DEPTH,
            "depth-anything",
            "wired",
            (
                "relative depth unless the metric variant is used",
                "the metric-indoor variant's licence is 'undeclared' in the registry",
            ),
        ),
        (
            "owlvit",
            "owlvit_base_patch32",
            "OWL-ViT B/32",
            Task.DETECTION,
            "owl-vit",
            "registry-only",
            ("open-vocabulary boxes only; no masks",),
        ),
        (
            "sam",
            "sam_vit_b",
            "Segment Anything ViT-B",
            Task.SEGMENTATION,
            "sam",
            "registry-only",
            ("the segmentation modules are not reachable from the production job",),
        ),
        (
            "mask_rcnn",
            "maskrcnn_resnet50_fpn",
            "Mask R-CNN R50-FPN (COCO)",
            Task.DETECTION,
            "maskrcnn",
            "wired",
            ("80 COCO classes; constructed on CPU by default in the pipeline",),
        ),
    ):
        licence, comm = lic[model_key]
        reg.register(
            EngineSpec(
                engine_id=eid,
                name=name,
                version=model_key,
                tasks=frozenset({task}),
                input_schema="RGB image",
                output_schema="depth_map" if task is Task.DEPTH else "bbox/mask",
                probe=_probe_from_perception(eid),
                family=family,
                dependencies=("torch",),
                checkpoints=(CheckpointRequirement(name=model_key),),
                hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
                licence=licence,
                commercial_ok=comm,
                limitations=lims,
                priority=20,
                integration=integ,
            )
        )
    return reg


def maximum_fidelity_registry() -> EngineRegistry:
    """The complete multi-engine registry for Maximum-Fidelity reconstruction.

    All engines have real availability probes and transparent licensing metadata.
    Engines with implemented adapters declare `integration='adapter'`.
    """
    from engine.multiengine.adapters import (
        probe_depth_anything_v2,
        probe_depth_anything_v3,
        probe_dinov2,
        probe_dust3r,
        probe_florence2,
        probe_grounding_dino,
        probe_mapanything,
        probe_mast3r,
        probe_midas,
        probe_open3d,
        probe_owlvit,
        probe_sam,
        probe_sam2,
        probe_sam3,
        probe_tracking,
        probe_vggt,
        probe_vggt_omega,
    )

    reg = EngineRegistry()

    # 1. Structure from Motion & Dense Multi-View Reconstruction
    reg.register(
        EngineSpec(
            engine_id="colmap",
            name="COLMAP (SfM + patch-match MVS)",
            version="cli",
            tasks=frozenset({Task.SFM, Task.MVS}),
            input_schema="images -> camera poses, sparse points; dense fused cloud",
            output_schema="camera_pose, sparse_point, dense_point",
            probe=_probe_from_perception("colmap"),
            family="colmap",
            dependencies=("colmap binary on PATH",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="BSD-3-Clause",
            commercial_ok=True,
            redistribution="binary is not shipped by this repository",
            limitations=(
                "dense MVS needs a CUDA build of COLMAP",
                "sparse stage on 10 photos took 79.5 s and produced 115 points",
            ),
            tests=("tests/test_colmap_session.py", "tests/test_colmap_backend_dense.py"),
            priority=10,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="mapanything",
            name="MapAnything Dense Multi-View Scene Reconstruction",
            version="1.0",
            tasks=frozenset({Task.SFM, Task.SCENE_UNDERSTANDING}),
            input_schema="unstructured multi-view images",
            output_schema="dense_scene_map, camera_poses",
            probe=probe_mapanything,
            family="mapanything",
            dependencies=("mapanything", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    # 2. Feedforward Visual Geometry Transformers
    reg.register(
        EngineSpec(
            engine_id="vggt",
            name="VGGT Visual Geometry Grounded Transformer",
            version="base",
            tasks=frozenset({Task.SFM, Task.DEPTH}),
            input_schema="multi-view images",
            output_schema="camera_pose, dense_pointmap",
            probe=probe_vggt,
            family="vggt",
            dependencies=("vggt", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="vggt_omega",
            name="VGGT-Omega High-Capacity 3D Transformer",
            version="omega",
            tasks=frozenset({Task.SFM, Task.DEPTH}),
            input_schema="multi-view images",
            output_schema="camera_pose, dense_pointmap",
            probe=probe_vggt_omega,
            family="vggt",
            dependencies=("vggt", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=16,
            integration="adapter",
        )
    )

    # 3. Pairwise 3D Pointmap Regression (Non-Commercial Research Licence)
    reg.register(
        EngineSpec(
            engine_id="dust3r",
            name="DUSt3R Dense 3D Pointmap Regression",
            version="512",
            tasks=frozenset({Task.SFM, Task.DEPTH, Task.FEATURE_MATCHING}),
            input_schema="stereo image pairs",
            output_schema="dense_pointmap, camera_poses",
            probe=probe_dust3r,
            family="dust3r",
            dependencies=("dust3r", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="CC-BY-NC-4.0",
            commercial_ok=False,
            limitations=(
                "CC-BY-NC-4.0 licence strictly prohibits commercial production deployment without explicit author waiver",
            ),
            priority=25,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="mast3r",
            name="MASt3R Matching and Stereo 3D Reconstruction",
            version="512",
            tasks=frozenset({Task.SFM, Task.DEPTH, Task.FEATURE_MATCHING}),
            input_schema="stereo image pairs",
            output_schema="dense_pointmap, feature_matches",
            probe=probe_mast3r,
            family="dust3r",
            dependencies=("mast3r", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="CC-BY-NC-4.0",
            commercial_ok=False,
            limitations=(
                "CC-BY-NC-4.0 licence strictly prohibits commercial production deployment without explicit author waiver",
            ),
            priority=26,
            integration="adapter",
        )
    )

    # 4. Monocular Depth Estimation
    reg.register(
        EngineSpec(
            engine_id="depth_anything_v3",
            name="Depth Anything V3",
            version="v3-small",
            tasks=frozenset({Task.DEPTH}),
            input_schema="RGB image",
            output_schema="depth_map",
            probe=probe_depth_anything_v3,
            family="depth-anything",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=12,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="depth_anything",
            name="Depth Anything V2",
            version="v2-small",
            tasks=frozenset({Task.DEPTH}),
            input_schema="RGB image",
            output_schema="depth_map",
            probe=probe_depth_anything_v2,
            family="depth-anything",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="midas",
            name="MiDaS DPT_Hybrid",
            version="dpt_hybrid",
            tasks=frozenset({Task.DEPTH}),
            input_schema="RGB image",
            output_schema="depth_map",
            probe=probe_midas,
            family="midas",
            dependencies=("torch",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="MIT",
            commercial_ok=True,
            limitations=("relative depth: requires scale anchor for metric conversion",),
            priority=20,
            integration="adapter",
        )
    )

    # 5. Open-Vocabulary 2D Object Detection
    reg.register(
        EngineSpec(
            engine_id="grounding_dino",
            name="Grounding DINO Zero-Shot Object Detector",
            version="tiny",
            tasks=frozenset({Task.DETECTION}),
            input_schema="RGB image + text prompt",
            output_schema="bbox",
            probe=probe_grounding_dino,
            family="grounding-dino",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="florence2",
            name="Florence-2 Vision-Language Grounding",
            version="base",
            tasks=frozenset({Task.DETECTION, Task.SCENE_UNDERSTANDING}),
            input_schema="RGB image + task prompt",
            output_schema="bbox, caption",
            probe=probe_florence2,
            family="florence",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="MIT",
            commercial_ok=True,
            priority=16,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="owlvit",
            name="OWL-ViT B/32 Open-Vocabulary Detector",
            version="base-patch32",
            tasks=frozenset({Task.DETECTION}),
            input_schema="RGB image + ontology prompt",
            output_schema="bbox",
            probe=probe_owlvit,
            family="owl-vit",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=20,
            integration="adapter",
        )
    )

    # 6. Instance Segmentation
    reg.register(
        EngineSpec(
            engine_id="sam",
            name="Segment Anything ViT-B",
            version="vit_b",
            tasks=frozenset({Task.SEGMENTATION}),
            input_schema="RGB image + box prompts",
            output_schema="mask, polygon_topology",
            probe=probe_sam,
            family="sam",
            dependencies=("segment_anything", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=20,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="sam2",
            name="SAM 2 Multi-View Instance Segmenter",
            version="hiera_small",
            tasks=frozenset({Task.SEGMENTATION}),
            input_schema="RGB image + box prompts",
            output_schema="mask, polygon_topology",
            probe=probe_sam2,
            family="sam",
            dependencies=("sam2", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="sam3",
            name="SAM 3 Next-Gen Volumetric Segmenter",
            version="unified",
            tasks=frozenset({Task.SEGMENTATION}),
            input_schema="RGB image + prompts",
            output_schema="mask, token_topology",
            probe=probe_sam3,
            family="sam",
            dependencies=("sam3", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=16,
            integration="adapter",
        )
    )

    # 7. Appearance & Tracking
    reg.register(
        EngineSpec(
            engine_id="dinov2",
            name="DINOv2 Self-Supervised Patch Embeddings",
            version="base",
            tasks=frozenset({Task.SCENE_UNDERSTANDING}),
            input_schema="RGB image crops",
            output_schema="feature_embedding",
            probe=probe_dinov2,
            family="dinov2",
            dependencies=("transformers", "torch"),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    reg.register(
        EngineSpec(
            engine_id="tracking",
            name="Multi-View Temporal Tracker",
            version="1.0",
            tasks=frozenset({Task.TRACKING}),
            input_schema="consecutive image frames",
            output_schema="track_correspondences",
            probe=probe_tracking,
            family="tracking",
            dependencies=("cv2",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="Apache-2.0",
            commercial_ok=True,
            priority=15,
            integration="adapter",
        )
    )

    # 8. Volumetric Surface Reconstruction
    reg.register(
        EngineSpec(
            engine_id="open3d_tsdf",
            name="Open3D TSDF Scalable Mesh Extractor",
            version="0.20",
            tasks=frozenset({Task.MVS}),
            input_schema="registered RGB-D frames + poses",
            output_schema="surface_mesh",
            probe=probe_open3d,
            family="open3d",
            dependencies=("open3d",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="MIT",
            commercial_ok=True,
            priority=10,
            integration="adapter",
        )
    )

    # Legacy baseline
    reg.register(
        EngineSpec(
            engine_id="mask_rcnn",
            name="Mask R-CNN R50-FPN (COCO)",
            version="maskrcnn_resnet50_fpn",
            tasks=frozenset({Task.DETECTION}),
            input_schema="RGB image",
            output_schema="bbox/mask",
            probe=_probe_from_perception("mask_rcnn"),
            family="maskrcnn",
            dependencies=("torch",),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
            licence="BSD-3-Clause",
            commercial_ok=True,
            limitations=("80 COCO classes; constructed on CPU by default",),
            priority=30,
            integration="wired",
        )
    )

    return reg
