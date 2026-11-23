"""Unified SceneManifest: one JSON-serialisable description of a benchmark scene, whatever dataset it came from.

It preserves, per scene: the ORIGINAL image paths and sha256 (the originals are never copied or rewritten), intrinsics / extrinsics, depth,
ground-truth mesh / point cloud, semantic / instance labels, 3D boxes, part annotations, the coordinate frame and units, the split, the
licence / terms, and -- per annotation -- a *provenance class* that is kept SEPARATE from the scene's *data kinds*:

* ``provenance_class`` (how a ground-truth item was obtained):
    ``measured``  -- an independent physical instrument (laser scanner, tape, total station), not derived from the images under test;
    ``estimated`` -- produced by an algorithm or human judgement from sensor data (SfM/SLAM poses, fused RGB-D meshes, human labels);
    ``fitted``    -- a model fitted to other data (a mesh fitted to a laser cloud, boxes fitted to a mesh);
    ``synthetic`` -- authored or rendered by a tool (designer layouts, renderer outputs).
* ``data_kinds`` (what the input is): any of ``synthetic``, ``real_image``, ``rgbd``, ``physical_gt`` (the scene carries independent
  physical ground truth). Reports aggregate per data-kind combination and per provenance class, never one blended number.

Nothing in this module reads a dataset; adapters (``adapters.py``) produce manifests from LOCAL paths and never download anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1"
PROVENANCE_CLASSES = ("measured", "estimated", "fitted", "synthetic")
DATA_KINDS = ("synthetic", "real_image", "rgbd", "physical_gt")
SPLITS = ("unassigned", "development", "validation", "test")
ANNOTATION_KINDS = ("mesh", "point_cloud", "semantic", "instance", "box3d", "part", "layout", "dimension", "structure", "pose", "depth")
CAMERA_AXES = ("opencv", "opengl", "unknown")  # opencv: x right, y down, z forward.  opengl: x right, y up, z backward.
_HEX = set("0123456789abcdef")


class ManifestFormatError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CoordinateFrame:
    name: str = "native"
    unit_to_metre: float = 1.0          # multiply native coordinates by this to get metres
    units: str = "m"                    # label only ("m", "mm")
    up_axis: str = "unknown"            # "+z", "+y", ... or "unknown"
    handedness: str = "right"
    pose_convention: str = "camera_to_world"
    camera_axes: str = "unknown"        # one of CAMERA_AXES; decides whether rotation error can be reported
    note: str = ""


@dataclass
class ImageRecord:
    id: str
    path: str                            # relative to the dataset root, exactly as it is on disk
    sha256: str
    width: int | None = None
    height: int | None = None
    intrinsics: dict[str, float] | None = None       # fx, fy, cx, cy (pixels); distortion NOT applied
    camera_model: str | None = None
    pose_c2w: list[list[float]] | None = None        # 4x4 camera-to-world, native units + native camera axes
    pose_provenance: str | None = None               # PROVENANCE_CLASSES member; required when pose_c2w is set
    depth_path: str | None = None
    depth_sha256: str | None = None
    depth_unit_to_metre: float | None = None
    depth_provenance: str | None = None


@dataclass
class Annotation:
    id: str
    kind: str                            # ANNOTATION_KINDS
    provenance_class: str                # PROVENANCE_CLASSES
    provenance_note: str = ""            # who/what produced it
    path: str | None = None              # file-backed annotations: relative path + sha256
    sha256: str | None = None
    label: str | None = None
    data: dict[str, Any] = field(default_factory=dict)   # inline payload (boxes, dimensions, ...)
    uncertainty_m: float | None = None


@dataclass
class Licence:
    name: str
    url: str = ""
    commercial_use: bool | None = None   # None = UNVERIFIED -- read the terms
    redistribution: bool | None = None
    registration_required: bool = True
    user_actions: list[str] = field(default_factory=list)
    terms_accepted_by_user: bool | None = None   # we never accept terms; the user does, out of band
    verified: bool = False               # False: the entries above were not verified against the live terms text
    notes: str = ""


@dataclass
class SceneManifest:
    dataset: str
    scene_id: str
    group_id: str                        # physical space; scenes sharing a group must share a split (rescans, multiple scans)
    kind: str = "interior"               # interior | exterior (the key engine.multiengine.benchmark_manifest uses)
    split: str = "unassigned"
    data_kinds: list[str] = field(default_factory=list)
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    images: list[ImageRecord] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    licence: Licence | None = None
    layout_verified: bool = False        # True only if the on-disk layout was verified against the official documentation
    notes: list[str] = field(default_factory=list)   # e.g. "needs frame export", UNVERIFIED items
    extra: dict[str, Any] = field(default_factory=dict)   # dataset-specific (official split, rescan transform, ...)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def annotations_of(self, kind: str) -> list[Annotation]:
        return [a for a in self.annotations if a.kind == kind]

    def data_kind_key(self) -> str:
        return "+".join(sorted(self.data_kinds))


def _from_dict(cls: Any, d: dict[str, Any]) -> Any:
    names = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in d.items() if k in names})


def scene_from_dict(d: dict[str, Any]) -> SceneManifest:
    problems = schema_problems(d)
    if problems:
        raise ManifestFormatError("; ".join(problems))
    return SceneManifest(
        **{k: d[k] for k in ("dataset", "scene_id", "group_id") if k in d},
        **{k: d[k] for k in ("kind", "split", "data_kinds", "layout_verified", "notes", "extra", "schema_version") if k in d},
        frame=_from_dict(CoordinateFrame, d.get("frame") or {}),
        images=[_from_dict(ImageRecord, i) for i in d.get("images", [])],
        annotations=[_from_dict(Annotation, a) for a in d.get("annotations", [])],
        licence=_from_dict(Licence, d["licence"]) if d.get("licence") else None,
    )


def save_scene_manifest(sm: SceneManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(sm.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_scene_manifest(path: str | Path) -> SceneManifest:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestFormatError(f"scene manifest unreadable: {exc}") from exc
    return scene_from_dict(d)


def with_split(sm: SceneManifest, split: str) -> SceneManifest:
    return replace(sm, split=split)


# --------------------------------------------------------------------------- JSON schema
_ENUM = lambda vals: {"type": "string", "enum": list(vals)}  # noqa: E731
_SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
SCENE_MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SceneManifest",
    "type": "object",
    "required": ["schema_version", "dataset", "scene_id", "group_id", "kind", "split", "data_kinds", "frame", "images", "annotations"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "dataset": {"type": "string", "minLength": 1}, "scene_id": {"type": "string", "minLength": 1}, "group_id": {"type": "string", "minLength": 1},
        "kind": _ENUM(("interior", "exterior")), "split": _ENUM(SPLITS),
        "data_kinds": {"type": "array", "items": _ENUM(DATA_KINDS), "minItems": 1, "uniqueItems": True},
        "frame": {"type": "object", "required": ["unit_to_metre", "camera_axes"],
                  "properties": {"unit_to_metre": {"type": "number", "exclusiveMinimum": 0}, "camera_axes": _ENUM(CAMERA_AXES),
                                 "units": {"type": "string"}, "name": {"type": "string"}, "up_axis": {"type": "string"},
                                 "handedness": {"type": "string"}, "pose_convention": {"const": "camera_to_world"}, "note": {"type": "string"}}},
        "images": {"type": "array", "items": {
            "type": "object", "required": ["id", "path", "sha256"],
            "properties": {"id": {"type": "string", "minLength": 1}, "path": {"type": "string", "minLength": 1}, "sha256": _SHA,
                           "width": {"type": ["integer", "null"]}, "height": {"type": ["integer", "null"]},
                           "intrinsics": {"type": ["object", "null"], "required": ["fx", "fy", "cx", "cy"]},
                           "camera_model": {"type": ["string", "null"]},
                           "pose_c2w": {"type": ["array", "null"], "minItems": 4, "maxItems": 4,
                                        "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}},
                           "pose_provenance": {"anyOf": [_ENUM(PROVENANCE_CLASSES), {"type": "null"}]},
                           "depth_path": {"type": ["string", "null"]}, "depth_sha256": {"anyOf": [_SHA, {"type": "null"}]},
                           "depth_unit_to_metre": {"type": ["number", "null"]},
                           "depth_provenance": {"anyOf": [_ENUM(PROVENANCE_CLASSES), {"type": "null"}]}}}},
        "annotations": {"type": "array", "items": {
            "type": "object", "required": ["id", "kind", "provenance_class"],
            "properties": {"id": {"type": "string", "minLength": 1}, "kind": _ENUM(ANNOTATION_KINDS), "provenance_class": _ENUM(PROVENANCE_CLASSES),
                           "provenance_note": {"type": "string"}, "path": {"type": ["string", "null"]},
                           "sha256": {"anyOf": [_SHA, {"type": "null"}]}, "label": {"type": ["string", "null"]},
                           "data": {"type": "object"}, "uncertainty_m": {"type": ["number", "null"]}}}},
        "licence": {"type": ["object", "null"], "required": ["name"]},
        "layout_verified": {"type": "boolean"}, "notes": {"type": "array", "items": {"type": "string"}}, "extra": {"type": "object"},
    },
}


def schema_problems(d: Any) -> list[str]:
    """Structural problems against the JSON schema (jsonschema is a dev dependency; a minimal check runs without it)."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is in the dev extra
        miss = [k for k in SCENE_MANIFEST_SCHEMA["required"] if not isinstance(d, dict) or k not in d]
        return [f"missing key {k!r}" for k in miss]
    v = jsonschema.Draft202012Validator(SCENE_MANIFEST_SCHEMA)
    return sorted(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(d))


# ------------------------------------------------------------------------------ semantic validation
def _finite_pose(m: Any) -> bool:
    a = np.asarray(m, float)
    return a.shape == (4, 4) and bool(np.isfinite(a).all()) and abs(np.linalg.det(a[:3, :3]) - 1.0) < 1e-2


def validate_scene(sm: SceneManifest, root: str | Path, *, verify_hashes: bool = True) -> list[str]:
    """Everything wrong with ``sm`` (empty list = consistent). Checks schema, enums, files, hashes and provenance/data-kind consistency."""
    root = Path(root)
    p = list(schema_problems(sm.to_dict()))
    if p:
        return p
    sid = sm.scene_id
    seen: set[str] = set()
    for im in sm.images:
        if im.id in seen:
            p.append(f"{sid}: duplicate image id {im.id!r}")
        seen.add(im.id)
        if im.pose_c2w is not None:
            if not _finite_pose(im.pose_c2w):
                p.append(f"{sid}: image {im.id!r} pose is not a finite rigid 4x4")
            if im.pose_provenance is None:
                p.append(f"{sid}: image {im.id!r} has a pose without pose_provenance")
        if im.intrinsics and not all(math.isfinite(im.intrinsics[k]) and im.intrinsics[k] > 0 for k in ("fx", "fy")):
            p.append(f"{sid}: image {im.id!r} has non-positive focal length")
        if im.depth_path and (im.depth_sha256 is None or im.depth_unit_to_metre is None or im.depth_provenance is None):
            p.append(f"{sid}: image {im.id!r} depth needs sha256, unit_to_metre and depth_provenance")
        for rel, sha in ((im.path, im.sha256), (im.depth_path, im.depth_sha256)):
            if rel and verify_hashes:
                f = root / rel
                if not f.is_file():
                    p.append(f"{sid}: file {rel!r} does not exist")
                elif sha and sha256_file(f) != sha:
                    p.append(f"{sid}: file {rel!r} does not match its recorded sha256 (originals must not change)")
    for a in sm.annotations:
        if a.path:
            if a.sha256 is None:
                p.append(f"{sid}: annotation {a.id!r} is file-backed but has no sha256")
            elif verify_hashes:
                f = root / a.path
                if not f.is_file():
                    p.append(f"{sid}: annotation file {a.path!r} does not exist")
                elif sha256_file(f) != a.sha256:
                    p.append(f"{sid}: annotation file {a.path!r} does not match its recorded sha256")
        elif not a.data:
            p.append(f"{sid}: annotation {a.id!r} has neither a file nor inline data")
        if "synthetic" in sm.data_kinds and a.provenance_class == "measured":
            p.append(f"{sid}: annotation {a.id!r} claims 'measured' in a synthetic scene (nothing physical was measured)")
    classes = {a.provenance_class for a in sm.annotations}
    if "physical_gt" in sm.data_kinds and "measured" not in classes:
        p.append(f"{sid}: data kind 'physical_gt' requires at least one 'measured' annotation")
    if "measured" in classes and "physical_gt" not in sm.data_kinds:
        p.append(f"{sid}: has 'measured' annotations but data_kinds lacks 'physical_gt' (kinds and provenance are reported separately but must agree)")
    return p


def to_benchmark_manifest(scenes: list[SceneManifest], benchmark_id: str) -> dict[str, Any]:
    """Convert to the dict ``engine.multiengine.benchmark_manifest`` validates (hash / split-leakage / ground-truth-provenance checks).

    Ground-truth keys are filled only from items that carry provenance; ``dimension`` items are included only when ``measured``.
    """
    out_scenes = []
    for sm in scenes:
        gt: dict[str, Any] = {}
        posed = [i for i in sm.images if i.pose_c2w is not None]
        if posed:
            gt["camera_poses"] = {"provenance": ",".join(sorted({i.pose_provenance or "?" for i in posed})), "ref": f"{sm.scene_id}/images"}
        depth = [i for i in sm.images if i.depth_path]
        if depth:
            gt["depth"] = {"provenance": ",".join(sorted({i.depth_provenance or "?" for i in depth})), "ref": f"{sm.scene_id}/depth"}
        for key, kinds in (("instance_labels", ("box3d", "instance")), ("instance_masks", ("instance", "semantic")),
                           ("scene_completeness", ("mesh", "point_cloud"))):
            an = [a for a in sm.annotations if a.kind in kinds]
            if an:
                gt[key] = {"provenance": ",".join(sorted({a.provenance_class for a in an})), "ref": f"{sm.scene_id}/{key}"}
        dims = [a for a in sm.annotations_of("dimension") if a.provenance_class == "measured"]
        if dims:
            gt["room_dimensions"] = {"provenance": "measured", "items": [
                {"name": a.data.get("name", a.id), "value_m": a.data.get("value_m"), "uncertainty_m": a.uncertainty_m} for a in dims]}
        out_scenes.append({"scene_id": sm.scene_id, "split": sm.split, "kind": sm.kind, "ground_truth": gt,
                           "images": [{"path": i.path, "sha256": i.sha256} for i in sm.images]})
    return {"benchmark_id": benchmark_id, "version": SCHEMA_VERSION, "scenes": out_scenes}


def pose_c2w_opencv(image: ImageRecord, frame: CoordinateFrame) -> np.ndarray | None:
    """GT pose as camera-to-world in METRES with OpenCV camera axes. None if the image has no pose.

    When ``frame.camera_axes`` is ``unknown`` the rotation is returned UNCONVERTED, so only camera CENTRES (axes-independent) are trustworthy;
    callers must check ``frame.camera_axes`` before reporting rotation error.
    """
    if image.pose_c2w is None:
        return None
    m = np.asarray(image.pose_c2w, float).copy()
    if frame.camera_axes == "opengl":
        m[:3, :3] = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    m[:3, 3] *= frame.unit_to_metre
    return m
