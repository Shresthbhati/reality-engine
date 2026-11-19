"""Engine registry: every engine's identity, needs, licence, limits and a REAL availability probe.

Availability is never inferred from "the package imports". Each engine supplies a probe that must return what it verified
(``Availability.evidence``); the six engines the repo can actually probe today reuse ``perception.availability`` (dependency
present AND checkpoint cached AND binary on PATH). A probe that raises is reported BLOCKED with the exception, never AVAILABLE.

``integration`` says how far the repository really goes with an engine: ``registry-only`` (declared, no adapter), ``adapter``
(an ``EngineAdapter`` exists) or ``wired`` (the production job path calls it). Nothing in this file creates an adapter.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from engine.multiengine.contract import (
    Availability, AvailabilityStatus, CheckpointRequirement, EngineSpec, HardwareRequirement, Task,
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
            return Availability(AvailabilityStatus.BLOCKED, f"availability probe raised {type(exc).__name__}: {exc}",
                                "fix the probe or the environment it checks")
        if not isinstance(a, Availability):
            return Availability(AvailabilityStatus.BLOCKED, "availability probe returned no Availability record")
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


def _licence(model_key: str) -> Tuple[str, Optional[bool]]:
    from perception import model_registry as mr

    spec = mr.all_specs().get(model_key)
    return (spec.license, spec.commercial_ok) if spec else ("", None)


def default_registry() -> EngineRegistry:
    """The registry of engines the repository declares today. ``min_vram_mb`` is deliberately left unset everywhere:
    ``perception.model_registry.measured_vram_mb`` is not backed by a cited measurement (the 2026-11-06 RTX 4050 audit
    measured SAM ViT-B at ~2.8 GB allocated against a registry figure of 950 MB), so it is not imported as a requirement."""
    reg = EngineRegistry()
    lic = {k: _licence(k) for k in ("midas_dpt_hybrid", "depth_anything_v2", "owlvit_base_patch32", "sam_vit_b",
                                    "maskrcnn_resnet50_fpn")}
    reg.register(EngineSpec(
        engine_id="colmap", name="COLMAP (SfM + patch-match MVS)", version="cli", tasks=frozenset({Task.SFM, Task.MVS}),
        input_schema="images -> camera poses, sparse points; dense fused cloud", output_schema="camera_pose, sparse_point, dense_point",
        probe=_probe_from_perception("colmap"), family="colmap", dependencies=("colmap binary on PATH",),
        hardware=HardwareRequirement(gpu_required=False, cpu_supported=True),
        licence="BSD-3-Clause", commercial_ok=True, redistribution="binary is not shipped by this repository",
        limitations=("dense MVS needs a CUDA build of COLMAP", "sparse stage on 10 photos took 79.5 s and produced 115 points "
                     "(datasets/south_building/runs/pipeline_measurement_20261118T044709.json)"),
        tests=("tests/test_colmap_session.py", "tests/test_colmap_backend_dense.py"),
        benchmarks=("datasets/south_building/runs/pipeline_measurement_20261118T044709.json",),
        priority=10, integration="wired"))
    for eid, model_key, name, task, family, integ, lims in (
        ("midas", "midas_dpt_hybrid", "MiDaS DPT_Hybrid", Task.DEPTH, "midas", "wired",
         ("relative depth: needs a scale anchor for metric use",)),
        ("depth_anything", "depth_anything_v2", "Depth Anything V2", Task.DEPTH, "depth-anything", "wired",
         ("relative depth unless the metric variant is used", "the metric-indoor variant's licence is 'undeclared' in the registry")),
        ("owlvit", "owlvit_base_patch32", "OWL-ViT B/32", Task.DETECTION, "owl-vit", "registry-only",
         ("open-vocabulary boxes only; no masks",)),
        ("sam", "sam_vit_b", "Segment Anything ViT-B", Task.SEGMENTATION, "sam", "registry-only",
         ("the segmentation modules are not reachable from the production job",)),
        ("mask_rcnn", "maskrcnn_resnet50_fpn", "Mask R-CNN R50-FPN (COCO)", Task.DETECTION, "maskrcnn", "wired",
         ("80 COCO classes; constructed on CPU by default in the pipeline",)),
    ):
        licence, comm = lic[model_key]
        reg.register(EngineSpec(
            engine_id=eid, name=name, version=model_key, tasks=frozenset({task}), input_schema="RGB image",
            output_schema="depth_map" if task is Task.DEPTH else "bbox/mask", probe=_probe_from_perception(eid), family=family,
            dependencies=("torch",), checkpoints=(CheckpointRequirement(name=model_key),),
            hardware=HardwareRequirement(gpu_required=False, cpu_supported=True), licence=licence, commercial_ok=comm,
            limitations=lims, priority=20, integration=integ))
    return reg
