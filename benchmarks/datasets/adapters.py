"""Dataset adapters: LOCAL on-disk dataset -> ``SceneManifest``. They never download, never extract archives, never open a network socket.

Every adapter documents the official layout it implements (source: the dataset's own documentation, retrieved 2026-11-23) and marks anything
it could not confirm as UNVERIFIED. Constructing one requires ``terms_acknowledged=True``: the USER attests that they obtained the data
under the dataset's terms (see ``licences.py`` for what they must do). The flag is an attestation recorded in the manifest, not a check.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .licences import LICENCES
from .manifest import Annotation, CoordinateFrame, ImageRecord, SceneManifest, sha256_file


class AccessNotAcknowledged(PermissionError):
    pass


class DatasetLayoutError(FileNotFoundError):
    pass


def _size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - not a decodable image (fixture bytes, unsupported format): size simply unknown
        return None, None


def _stride(items: list, max_n: int | None) -> list:
    """Deterministic uniform subsample (first and last kept) -- no RNG, same input -> same output."""
    if not max_n or len(items) <= max_n:
        return items
    if max_n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (max_n - 1))] for i in range(max_n)]


def _ints(stem: str) -> int:
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else -1


def _mat(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


class Adapter:
    name = ""
    layout_verified = False
    interior = True

    def __init__(self, root: str | Path, *, terms_acknowledged: bool = False) -> None:
        if not terms_acknowledged:
            raise AccessNotAcknowledged(
                f"{self.name}: pass terms_acknowledged=True to attest that YOU obtained this dataset under its terms. "
                f"What you must do: {LICENCES[self.name].user_actions}")
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetLayoutError(f"{self.name}: dataset root {self.root} does not exist (nothing is ever downloaded)")
        self.licence = replace(LICENCES[self.name], terms_accepted_by_user=True)  # user attestation, not verified

    # -- helpers
    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def image(self, id_: str, p: Path, **kw: Any) -> ImageRecord:
        w, h = _size(p)
        return ImageRecord(id=id_, path=self.rel(p), sha256=sha256_file(p), width=w, height=h, **kw)

    def file_ann(self, id_: str, kind: str, p: Path, prov: str, note: str, **kw: Any) -> Annotation:
        return Annotation(id=id_, kind=kind, provenance_class=prov, provenance_note=note, path=self.rel(p), sha256=sha256_file(p), **kw)

    def scene_ids(self) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def load_scene(self, scene_id: str, *, max_images: int | None = None) -> SceneManifest:  # pragma: no cover - interface
        raise NotImplementedError

    def _scene(self, scene_id: str, group_id: str, data_kinds: list[str], frame: CoordinateFrame, **kw: Any) -> SceneManifest:
        return SceneManifest(dataset=self.name, scene_id=scene_id, group_id=group_id, kind="interior", data_kinds=data_kinds, frame=frame,
                             licence=self.licence, layout_verified=self.layout_verified, **kw)


# =========================================================================== ScanNet
class ScanNetAdapter(Adapter):
    """ScanNet v2.

    VERIFIED (official README, github.com/ScanNet/ScanNet): scene directories ``scene%04d_%02d`` (space, scan) each holding
    ``<scanId>.sens``, ``<scanId>_vh_clean.ply``, ``<scanId>_vh_clean_2.ply``, ``<scanId>_vh_clean_2.labels.ply``,
    ``<scanId>_vh_clean_2.0.010000.segs.json``, ``<scanId>.aggregation.json`` / ``<scanId>_vh_clean.aggregation.json``; meshes are binary PLY.
    Scenes of one space (``scene0000_00``, ``scene0000_01``) are the same physical room -> the group id is the space.

    UNVERIFIED: frames come from the user running the official SensReader on the .sens. We expect its output under
    ``<scene>/exported/{color/N.jpg, depth/N.png, pose/N.txt, intrinsic/intrinsic_color.txt}`` (or directly in ``<scene>/``); poses
    camera-to-world, invalid poses non-finite; depth in millimetres; camera axes (left ``unknown`` => rotation error is not reported).
    Ground truth here is ESTIMATED: poses and meshes come from BundleFusion on the same RGB-D, labels from human annotation.
    There is no independent physical measurement, so dimension errors are BLOCKED for ScanNet.
    """

    name = "scannet"

    def _scans(self) -> Path:
        return self.root / "scans" if (self.root / "scans").is_dir() else self.root

    def scene_ids(self) -> list[str]:
        return sorted(d.name for d in self._scans().iterdir() if d.is_dir() and re.fullmatch(r"scene\d{4}_\d{2}", d.name))

    def load_scene(self, scene_id: str, *, max_images: int | None = None) -> SceneManifest:
        d = self._scans() / scene_id
        if not d.is_dir():
            raise DatasetLayoutError(f"ScanNet scene {scene_id} not found under {self._scans()}")
        ex = d / "exported" if (d / "exported").is_dir() else d
        notes = ["UNVERIFIED: exported-frame layout (SensReader output location/names), pose = camera-to-world, depth unit mm, camera axes"]
        K = None
        kf = ex / "intrinsic" / "intrinsic_color.txt"
        if kf.is_file():
            m = _mat(kf)
            K = {"fx": float(m[0, 0]), "fy": float(m[1, 1]), "cx": float(m[0, 2]), "cy": float(m[1, 2])}
        meta = d / f"{scene_id}.txt"
        wh = (None, None)
        if meta.is_file():
            kv = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", meta.read_text(encoding="utf-8", errors="replace"), re.M))
            if "colorWidth" in kv and "colorHeight" in kv:
                wh = (int(float(kv["colorWidth"])), int(float(kv["colorHeight"])))
        colors = sorted((ex / "color").glob("*.jpg"), key=lambda p: _ints(p.stem)) if (ex / "color").is_dir() else []
        images: list[ImageRecord] = []
        skipped = 0
        valid = []
        for c in colors:
            pf = ex / "pose" / f"{c.stem}.txt"
            if pf.is_file() and np.isfinite(_mat(pf)).all():
                valid.append((c, pf))
            else:
                skipped += 1
        for c, pf in _stride(valid, max_images):
            df = ex / "depth" / f"{c.stem}.png"
            kw: dict[str, Any] = {"intrinsics": K, "pose_c2w": _mat(pf).tolist(), "pose_provenance": "estimated"}
            if df.is_file():
                kw.update(depth_path=self.rel(df), depth_sha256=sha256_file(df), depth_unit_to_metre=0.001, depth_provenance="estimated")
            im = self.image(f"{scene_id}/{c.stem}", c, **kw)
            if im.width is None:
                im.width, im.height = wh
            images.append(im)
        if not colors:
            notes.append(f"needs export: no exported color frames under {ex}; run SensReader on {scene_id}.sens (nothing was extracted here)")
        if skipped:
            notes.append(f"{skipped} frame(s) skipped: missing or non-finite pose")
        an: list[Annotation] = []
        for suffix, kind, note in (("_vh_clean_2.ply", "mesh", "BundleFusion mesh of the same RGB-D (not independent of the sensor data)"),
                                   ("_vh_clean_2.labels.ply", "semantic", "human-annotated NYU40 labels coloured on the mesh"),
                                   ("_vh_clean.aggregation.json", "instance", "human-annotated instance groups over mesh segments"),
                                   ("_vh_clean_2.0.010000.segs.json", "instance", "mesh over-segmentation referenced by the aggregation")):
            f = d / f"{scene_id}{suffix}"
            if f.is_file():
                an.append(self.file_ann(f"{scene_id}{suffix}", kind, f, "estimated", note))
        space = scene_id.split("_")[0]
        return self._scene(scene_id, space, ["rgbd"], CoordinateFrame(name="scannet_native", unit_to_metre=1.0, units="m", up_axis="+z (per README, mesh)",
                                                                      camera_axes="unknown", note="axisAlignment in <scene>.txt NOT applied"),
                           images=images, annotations=an, notes=notes, extra={"space": space})


# ====================================================================== ScanNet++
class ScanNetPPAdapter(Adapter):
    """ScanNet++ (v2 documentation layout).

    VERIFIED (kaldir.vc.in.tum.de/scannetpp/documentation): ``data/<scene_id>/{scans,dslr,iphone,panocam}``; ``split/{nvs_sem_train,nvs_sem_val,
    nvs_test,sem_test}.txt``; ``scans/{pc_aligned.ply (laser point cloud), mesh_aligned_0.05.ply, mesh_aligned_0.05_semantic.ply (label -100 =
    unlabelled), segments_anno.json}``; ``dslr/{resized_images, resized_undistorted_images, nerfstudio/{transforms.json, transforms_undistorted.json}}``;
    the nerfstudio poses use the OpenGL/Blender convention and are metric (aligned to the laser scan); DSLR raw images are fisheye.
    nvs_test scenes carry NO scan data.

    UNVERIFIED: the nerfstudio JSON field names (fl_x, fl_y, cx, cy, w, h, frames[].file_path, frames[].transform_matrix, frames[].is_bad) follow the
    nerfstudio dataparser convention, and file_path is matched to image files by basename.
    GT classes: laser point cloud = measured (independent of the images), mesh = fitted to that cloud, DSLR poses = estimated (COLMAP aligned to scan).
    Anonymised pixels (magenta) exist in the images; masks are not applied by this adapter.
    """

    name = "scannetpp"

    def scene_ids(self) -> list[str]:
        return sorted(d.name for d in (self.root / "data").iterdir() if d.is_dir())

    def _official_split(self, scene_id: str) -> str | None:
        for f in ("nvs_sem_train", "nvs_sem_val", "nvs_test", "sem_test"):
            p = self.root / "split" / f"{f}.txt"
            if p.is_file() and scene_id in p.read_text(encoding="utf-8").split():
                return f
        return None

    def load_scene(self, scene_id: str, *, max_images: int | None = None, undistorted: bool = True) -> SceneManifest:
        d = self.root / "data" / scene_id
        if not d.is_dir():
            raise DatasetLayoutError(f"ScanNet++ scene {scene_id} not found under {self.root / 'data'}")
        dslr = d / "dslr"
        tf = dslr / "nerfstudio" / ("transforms_undistorted.json" if undistorted else "transforms.json")
        imgdir = dslr / ("resized_undistorted_images" if undistorted else "resized_images")
        notes = ["UNVERIFIED: nerfstudio field names (nerfstudio convention) and file_path -> basename matching"]
        images: list[ImageRecord] = []
        bad = 0
        if tf.is_file() and imgdir.is_dir():
            j = json.loads(tf.read_text(encoding="utf-8"))
            K = {k: float(j[s]) for k, s in (("fx", "fl_x"), ("fy", "fl_y"), ("cx", "cx"), ("cy", "cy"))}
            frames = []
            for fr in list(j.get("frames", [])) + list(j.get("test_frames", [])):
                if fr.get("is_bad"):
                    bad += 1
                    continue
                f = imgdir / Path(fr["file_path"]).name
                if f.is_file():
                    frames.append((f, fr["transform_matrix"]))
            for f, mtx in _stride(sorted(frames, key=lambda t: t[0].name), max_images):
                im = self.image(f"{scene_id}/{f.stem}", f, intrinsics=K, camera_model=j.get("camera_model"), pose_c2w=mtx, pose_provenance="estimated")
                im.width, im.height = im.width or j.get("w"), im.height or j.get("h")
                images.append(im)
            if not undistorted:
                notes.append("distorted fisheye images selected: a pinhole pipeline will mis-handle them")
        else:
            notes.append(f"missing DSLR frames or {tf.name}: no images (nothing downloaded)")
        if bad:
            notes.append(f"{bad} frame(s) flagged is_bad (blur/shadows) were skipped")
        an: list[Annotation] = []
        sc = d / "scans"
        for fn, kind, prov, note in (
                ("pc_aligned.ply", "point_cloud", "measured", "terrestrial laser scanner point cloud (independent of the DSLR images)"),
                ("mesh_aligned_0.05.ply", "mesh", "fitted", "mesh obtained from the laser cloud, decimated to 5 percent"),
                ("mesh_aligned_0.05_semantic.ply", "semantic", "estimated", "human-annotated per-vertex labels (-100 = unlabelled)"),
                ("segments_anno.json", "instance", "estimated", "human-annotated objects over mesh segments")):
            if (sc / fn).is_file():
                an.append(self.file_ann(f"{scene_id}/{fn}", kind, sc / fn, prov, note))
        kinds = ["real_image"] + (["physical_gt"] if any(a.provenance_class == "measured" for a in an) else [])
        if not an:
            notes.append("no scan data for this scene (e.g. nvs_test): geometry metrics are BLOCKED")
        return self._scene(scene_id, scene_id, kinds, CoordinateFrame(name="scannetpp_scan_aligned", unit_to_metre=1.0, units="m", camera_axes="opengl",
                                                                      note="poses metric, aligned to the laser scan (documentation)"),
                           images=images, annotations=an, notes=notes, extra={"official_split": self._official_split(scene_id)})


# ====================================================================== ARKitScenes
class ARKitScenesAdapter(Adapter):
    """ARKitScenes 3DOD.

    VERIFIED (DATA.md, threedod/README.md): ``<root>/{Training,Validation}/<video_id>/`` (sample layout: ``<video_id>/``) with
    ``<video_id>_3dod_annotation.json`` (data[]: label, axesLengths, centroid, normalizedAxes), ``<video_id>_3dod_mesh.ply``
    (the mesh GENERATED BY ARKIT, not a laser mesh), and ``<video_id>_frames/`` with ``lowres_wide`` (sample: ``wide``) colour frames,
    ``lowres_wide.traj`` (sample: ``color.traj``; columns: timestamp, axis-angle rotation, translation in metres), ``lowres_wide_intrinsics``
    (sample: ``color_intrinsics``; ``.pincam`` = ``width height fx fy cx cy``), depth ``lowres_depth`` (sample ``depth_densified``) as uint16 mm PNG.
    ``_3dod_annotation.json`` boxes are human annotations on the ARKit mesh => ESTIMATED.

    UNVERIFIED: (1) whether the .traj pose is world-to-camera (we assume so and invert, ``traj_is_world_to_camera=True``; if wrong the ATE is
    meaningless -- check against the official tenFpsDataLoader before trusting any pose metric); (2) frame filenames ``<video_id>_<timestamp>.png``
    vs bare ``<timestamp>.png`` (both accepted; the timestamp is the last ``_`` token); (3) camera axes (left ``unknown``).
    Independent physical GT exists only as raw laser-scanner point clouds, which are NOT wired: data kind is ``rgbd`` only.
    """

    name = "arkitscenes"

    def _folds(self) -> list[Path]:
        base = self.root / "3dod" if (self.root / "3dod").is_dir() else self.root
        named = [base / f for f in ("Training", "Validation") if (base / f).is_dir()]
        return named or [base]

    def _locate(self, video_id: str) -> tuple[Path, str]:
        for fold in self._folds():
            if (fold / video_id).is_dir():
                return fold / video_id, ("" if fold.name not in ("Training", "Validation") else fold.name)
        raise DatasetLayoutError(f"ARKitScenes video {video_id} not found under {self.root}")

    def scene_ids(self) -> list[str]:
        return sorted(d.name for f in self._folds() for d in f.iterdir() if d.is_dir() and d.name.isdigit())

    @staticmethod
    def _rodrigues(w: np.ndarray) -> np.ndarray:
        th = float(np.linalg.norm(w))
        if th < 1e-12:
            return np.eye(3)
        k = w / th
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * kx @ kx

    def load_scene(self, scene_id: str, *, max_images: int | None = None, traj_is_world_to_camera: bool = True) -> SceneManifest:
        d, fold = self._locate(scene_id)
        fr = d / f"{scene_id}_frames"
        pick = lambda *names: next((fr / n for n in names if (fr / n).exists()), None)  # noqa: E731
        cdir, tfile, kdir, ddir = pick("lowres_wide", "wide"), pick("lowres_wide.traj", "color.traj"), pick("lowres_wide_intrinsics", "color_intrinsics"), \
            pick("lowres_depth", "depth_densified")
        notes = ["UNVERIFIED: .traj pose direction (assumed world-to-camera), frame file naming, camera axes"]
        traj: list[tuple[float, np.ndarray]] = []
        if tfile and tfile.is_file():
            for line in tfile.read_text(encoding="utf-8").splitlines():
                v = line.replace(",", " ").split()
                if len(v) >= 7:
                    f = [float(x) for x in v[:7]]
                    e = np.eye(4)
                    e[:3, :3], e[:3, 3] = self._rodrigues(np.array(f[1:4])), f[4:7]
                    traj.append((f[0], np.linalg.inv(e) if traj_is_world_to_camera else e))
        ts_arr = np.array([t for t, _ in traj]) if traj else np.array([])

        def ts_of(p: Path) -> float:
            return float(p.stem.split("_")[-1])

        frames = sorted((p for p in cdir.glob("*.png")), key=ts_of) if cdir and cdir.is_dir() else []
        images: list[ImageRecord] = []
        for p in _stride(frames, max_images):
            t = ts_of(p)
            pose = None
            if len(ts_arr):
                i = int(np.argmin(np.abs(ts_arr - t)))
                pose = traj[i][1].tolist() if abs(ts_arr[i] - t) <= 0.05 else None
            kw: dict[str, Any] = {}
            pc = (kdir / f"{p.stem}.pincam") if kdir else None
            w = h = None
            if pc and pc.is_file():
                w, h, fx, fy, cx, cy = (float(x) for x in pc.read_text(encoding="utf-8").split()[:6])
                kw["intrinsics"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            if pose is not None:
                kw.update(pose_c2w=pose, pose_provenance="estimated")
            dp = (ddir / p.name) if ddir else None
            if dp and dp.is_file():
                kw.update(depth_path=self.rel(dp), depth_sha256=sha256_file(dp), depth_unit_to_metre=0.001, depth_provenance="estimated")
            im = self.image(f"{scene_id}/{p.stem}", p, **kw)
            im.width, im.height = im.width or (int(w) if w else None), im.height or (int(h) if h else None)
            images.append(im)
        if not frames:
            notes.append(f"no colour frames found under {fr} (nothing downloaded)")
        an: list[Annotation] = []
        mesh = d / f"{scene_id}_3dod_mesh.ply"
        if mesh.is_file():
            an.append(self.file_ann(f"{scene_id}/mesh", "mesh", mesh, "estimated", "mesh generated by ARKit during capture (not a laser mesh)"))
        ann = d / f"{scene_id}_3dod_annotation.json"
        if ann.is_file():
            for i, b in enumerate(json.loads(ann.read_text(encoding="utf-8")).get("data", [])):
                an.append(Annotation(id=f"{scene_id}/box{i}", kind="box3d", provenance_class="estimated",
                                     provenance_note="human-annotated oriented box on the ARKit mesh", label=b.get("label"),
                                     data={"centroid": b.get("centroid"), "axesLengths": b.get("axesLengths"), "normalizedAxes": b.get("normalizedAxes"),
                                           "uid": b.get("uid")}))
        return self._scene(scene_id, scene_id, ["rgbd"], CoordinateFrame(name="arkit_world", unit_to_metre=1.0, units="m", camera_axes="unknown"),
                           images=images, annotations=an, notes=notes, extra={"official_split": fold or None})


# ===================================================================== Structured3D
class Structured3DAdapter(Adapter):
    """Structured3D (SYNTHETIC: photo-realistic renders of designer house layouts).

    VERIFIED (README + data_organization.md): ``scene_<id>/2D_rendering/<roomID>/perspective/<empty|full>/<positionID>/{rgb_rawlight.png,
    depth.png (uint16 mm; 0 = no reading), semantic.png, instance.png, normal.png, layout.json, camera_pose.txt}``; ``scene_<id>/{annotation_3d.json
    (junctions, lines, planes, semantics), bbox_3d.json (ID, basis, coeffs, centroid)}``; ``camera_pose.txt`` = ``vx vy vz tx ty tz ux uy uz xfov
    yfov 1`` with the eye in MILLIMETRES, view and up directions in the global frame (z up), xfov/yfov HALF-angles in radians. Official split by
    scene number: 0-2999 train, 3000-3249 val, 3250-3499 test.

    Derived here (by construction, not documented): OpenCV camera axes (z = view, y = -up, x = view x up) and pinhole intrinsics
    fx = (W/2)/tan(xfov), fy = (H/2)/tan(yfov), c = (W/2, H/2) -- needs the real image size, so intrinsics are None when it cannot be read.
    Everything is ``synthetic`` provenance: this data can show the pipeline is wired, it says NOTHING about real-capture accuracy.
    """

    name = "structured3d"
    layout_verified = True   # every path/format used is in the official documentation (derived camera model is documented above)

    def scene_ids(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and re.fullmatch(r"scene_\d+", d.name))

    @staticmethod
    def _c2w(v: list[float]) -> list[list[float]]:
        eye, fwd, up = np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9])
        fwd = fwd / np.linalg.norm(fwd)
        up = up - fwd * float(up @ fwd)
        up = up / np.linalg.norm(up)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = np.cross(fwd, up), -up, fwd, eye
        return m.tolist()

    def load_scene(self, scene_id: str, *, max_images: int | None = None, config: str = "full") -> SceneManifest:
        d = self.root / scene_id
        if not d.is_dir():
            raise DatasetLayoutError(f"Structured3D scene {scene_id} not found under {self.root}")
        views = sorted((p for p in d.glob(f"2D_rendering/*/perspective/{config}/*/rgb_rawlight.png")), key=lambda p: p.as_posix())
        images: list[ImageRecord] = []
        notes = ["intrinsics/poses derived from camera_pose.txt by this adapter (OpenCV axes, half-angle fov); mm -> m via frame.unit_to_metre"]
        for p in _stride(views, max_images):
            pos = p.parent
            cp = pos / "camera_pose.txt"
            if not cp.is_file():
                continue
            v = [float(x) for x in cp.read_text(encoding="utf-8").split()]
            if len(v) < 11:
                continue
            kw: dict[str, Any] = {"pose_c2w": self._c2w(v), "pose_provenance": "synthetic"}
            dp = pos / "depth.png"
            if dp.is_file():
                kw.update(depth_path=self.rel(dp), depth_sha256=sha256_file(dp), depth_unit_to_metre=0.001, depth_provenance="synthetic")
            room = pos.parents[2].name
            im = self.image(f"{scene_id}/{room}_{pos.name}", p, **kw)
            if im.width:
                im.intrinsics = {"fx": im.width / 2 / math.tan(v[9]), "fy": im.height / 2 / math.tan(v[10]), "cx": im.width / 2, "cy": im.height / 2}
            images.append(im)
        an: list[Annotation] = []
        for fn, kind, note in (("annotation_3d.json", "structure", "designer-authored planes/junctions/semantics (planes unbounded)"),
                               ("bbox_3d.json", "box3d", "designer-authored oriented boxes, millimetres")):
            if (d / fn).is_file():
                an.append(self.file_ann(f"{scene_id}/{fn}", kind, d / fn, "synthetic", note))
        n = int(scene_id.split("_")[1])
        split = "train" if n < 3000 else "val" if n < 3250 else "test"
        return self._scene(scene_id, scene_id, ["synthetic"], CoordinateFrame(name="structured3d_world", unit_to_metre=0.001, units="mm", up_axis="+z",
                                                                              camera_axes="opencv"),
                           images=images, annotations=an, notes=notes, extra={"official_split": split, "configuration": config})


# ========================================================================== 3RScan
class RScan3Adapter(Adapter):
    """3RScan (reference scans + rescans of changing rooms).

    VERIFIED (README): ``<scanId>/{mesh.refined.v2.obj, mesh.refined.mtl, mesh.refined_0.png, sequence.zip, labels.instances.annotated.v2.ply,
    mesh.refined.0.010000.segs.v2.json, semseg.v2.json (segGroups[].{id, objectId, label, segments, obb.{centroid, axesLengths}})}`` and
    ``3RScan.json`` = list of {reference, type (train/test/val), scans: [{reference (rescan id), transform (16 floats, rescan -> reference)}]}.
    Reference and its rescans are the same room -> one group, one split.

    UNVERIFIED: the unpacked ``<scanId>/sequence/`` layout (``frame-XXXXXX.color.jpg``, ``.pose.txt``, ``.depth.pgm``, ``_info.txt`` with
    ``m_calibrationColorIntrinsic``, ``m_colorWidth/Height``); pose direction and camera axes (axes left ``unknown``); colour frames are believed to
    be stored rotated 90 degrees (a known 3RScan gotcha) -- NOT corrected here. This adapter never unzips: unzip sequence.zip yourself.
    GT is ESTIMATED (reconstruction + human labels). Depth (.pgm) is not hashed/ingested.
    """

    name = "3rscan"

    def _groups(self) -> dict[str, dict[str, Any]]:
        mj = self.root / "3RScan.json"
        out: dict[str, dict[str, Any]] = {}
        if mj.is_file():
            try:
                for ref in json.loads(mj.read_text(encoding="utf-8")):
                    out[ref["reference"]] = {"group": ref["reference"], "role": "reference", "type": ref.get("type"), "transform": None}
                    for sc in ref.get("scans", []):
                        out[sc["reference"]] = {"group": ref["reference"], "role": "rescan", "type": ref.get("type"), "transform": sc.get("transform")}
            except (ValueError, KeyError, TypeError):
                return {}
        return out

    def scene_ids(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and ((d / "sequence").is_dir() or (d / "sequence.zip").is_file()
                                                                           or (d / "mesh.refined.v2.obj").is_file()))

    def load_scene(self, scene_id: str, *, max_images: int | None = None) -> SceneManifest:
        d = self.root / scene_id
        if not d.is_dir():
            raise DatasetLayoutError(f"3RScan scan {scene_id} not found under {self.root}")
        g = self._groups().get(scene_id)
        notes = ["UNVERIFIED: unpacked sequence layout, pose direction/camera axes; frames believed stored rotated (not corrected)"]
        if g is None:
            notes.append("3RScan.json missing/unparsable or scan not listed: group = the scan itself (rescans of the same room could LEAK across splits)")
        seq = d / "sequence"
        K, wh = None, (None, None)
        info = seq / "_info.txt"
        if info.is_file():
            kv = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", info.read_text(encoding="utf-8", errors="replace"), re.M))
            v = [float(x) for x in kv.get("m_calibrationColorIntrinsic", "").split()]
            if len(v) >= 7:
                K = {"fx": v[0], "fy": v[5], "cx": v[2], "cy": v[6]}
            if "m_colorWidth" in kv and "m_colorHeight" in kv:
                wh = (int(float(kv["m_colorWidth"])), int(float(kv["m_colorHeight"])))
        colors = sorted(seq.glob("frame-*.color.jpg"), key=lambda p: _ints(p.name)) if seq.is_dir() else []
        images = []
        for c in _stride(colors, max_images):
            pf = c.with_name(c.name.replace(".color.jpg", ".pose.txt"))
            kw: dict[str, Any] = {"intrinsics": K}
            if pf.is_file() and np.isfinite(_mat(pf)).all():
                kw.update(pose_c2w=_mat(pf).tolist(), pose_provenance="estimated")
            im = self.image(f"{scene_id}/{c.name.split('.')[0]}", c, **kw)
            im.width, im.height = im.width or wh[0], im.height or wh[1]
            images.append(im)
        if not colors:
            notes.append("no unpacked colour frames: unzip sequence.zip into <scanId>/sequence/ yourself (this tooling never extracts archives)")
        an: list[Annotation] = []
        for fn, kind, note in (("mesh.refined.v2.obj", "mesh", "reconstructed mesh from the same RGB-D sequence (not independent)"),
                               ("semseg.v2.json", "instance", "human-annotated instance groups with oriented boxes"),
                               ("mesh.refined.0.010000.segs.v2.json", "instance", "mesh over-segmentation"),
                               ("labels.instances.annotated.v2.ply", "semantic", "visualisation of the instance annotation")):
            if (d / fn).is_file():
                an.append(self.file_ann(f"{scene_id}/{fn}", kind, d / fn, "estimated", note))
        return self._scene(scene_id, g["group"] if g else scene_id, ["rgbd"], CoordinateFrame(name="3rscan_native", unit_to_metre=1.0, units="m",
                                                                                               up_axis="+z (obj, per README)", camera_axes="unknown"),
                           images=images, annotations=an, notes=notes,
                           extra={"role": g["role"] if g else None, "official_split": g["type"] if g else None,
                                  "transform_to_reference": g["transform"] if g else None})


# ======================================================================== MultiScan
class MultiScanAdapter(Adapter):
    """MultiScan.

    VERIFIED (docs: dataset/index, files/acquired, files/annotation): ``scene_xxxxx_xx/{.annotations.json, .align.json, .ply, .json, .jsonl, .mp4,
    .depth.zlib, .confidence.zlib, textured_mesh/}``; scene id = environment, last two digits = scan of it (same environment -> one group);
    ``.jsonl``: one JSON object per frame with ``intrinsics`` (9 floats, column-major 3x3), ``transform`` (16 floats, column-major 4x4,
    camera-to-world, ARKit axes = x right, y up, z toward the viewer, i.e. OpenGL-like), ``quaternion`` (wxyz), ``timestamp``;
    ``.annotations.json``: ``objects[]`` {objectId, label, mobilityType, partIds, obb{centroid, axesLengths, normalizedAxes (col-major 9), ...}}
    and ``parts[]`` {partId, label, parentId, articulations[]}; the ``.ply`` carries per-face objectId/partId.

    NOT PROVIDED BY THE DATASET: RGB exists only as H.264 ``.mp4`` (depth as float16/uint16 ``.depth.zlib``); the docs list the decode code as a
    TODO. We never decode video. Frames must be user-extracted to ``<scene>/frames/<NNNNNN>.png|jpg`` (NNNNNN = 0-based ``.jsonl`` line index): that
    naming is OUR convention and UNVERIFIED against the capture app's frame count/ordering. Without it the scene has no images (-> BLOCKED).
    Provenance: ARKit poses and the reconstructed mesh = ESTIMATED; human object/part/articulation annotations = ESTIMATED.
    """

    name = "multiscan"

    def _scenes(self) -> Path:
        return self.root / "scenes" if (self.root / "scenes").is_dir() else self.root

    def scene_ids(self) -> list[str]:
        return sorted(d.name for d in self._scenes().iterdir() if d.is_dir() and re.fullmatch(r"scene_\d{5}_\d{2}", d.name))

    def load_scene(self, scene_id: str, *, max_images: int | None = None) -> SceneManifest:
        d = self._scenes() / scene_id
        if not d.is_dir():
            raise DatasetLayoutError(f"MultiScan scan {scene_id} not found under {self._scenes()}")
        notes = ["UNVERIFIED: frames/ naming convention (ours), frame-count agreement between mp4 and jsonl"]
        cams: list[dict[str, Any]] = []
        jl = d / f"{scene_id}.jsonl"
        if jl.is_file():
            cams = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]
        fdir = d / "frames"
        files = {int(_ints(p.stem)): p for p in sorted(fdir.glob("*.*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg")} if fdir.is_dir() else {}
        images = []
        for i in _stride(sorted(i for i in files if 0 <= i < len(cams)), max_images):
            c = cams[i]
            kw: dict[str, Any] = {}
            if c.get("intrinsics") and len(c["intrinsics"]) == 9:
                k = np.asarray(c["intrinsics"], float).reshape(3, 3, order="F")
                kw["intrinsics"] = {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])}
            if c.get("transform") and len(c["transform"]) == 16:
                kw.update(pose_c2w=np.asarray(c["transform"], float).reshape(4, 4, order="F").tolist(), pose_provenance="estimated")
            images.append(self.image(f"{scene_id}/{i:06d}", files[i], **kw))
        if not images:
            notes.append("no decoded frames: decode <scene>.mp4 to <scene>/frames/ yourself (this tooling never decodes video or reads .depth.zlib)")
        an: list[Annotation] = []
        for suffix, kind, note in ((".ply", "mesh", "reconstructed mesh (Open3D-based pipeline) with per-face objectId/partId"),):
            if (d / f"{scene_id}{suffix}").is_file():
                an.append(self.file_ann(f"{scene_id}{suffix}", kind, d / f"{scene_id}{suffix}", "estimated", note))
        af = d / f"{scene_id}.annotations.json"
        if af.is_file():
            aj = json.loads(af.read_text(encoding="utf-8"))
            for o in aj.get("objects", []):
                an.append(Annotation(id=f"{scene_id}/object{o.get('objectId')}", kind="box3d", provenance_class="estimated",
                                     provenance_note="human-annotated semantic OBB", label=o.get("label"),
                                     data={"mobilityType": o.get("mobilityType"), "partIds": o.get("partIds"), **(o.get("obb") or {})}))
            for p in aj.get("parts", []):
                an.append(Annotation(id=f"{scene_id}/part{p.get('partId')}", kind="part", provenance_class="estimated",
                                     provenance_note="human-annotated part (+ articulation)", label=p.get("label"),
                                     data={"parentId": p.get("parentId"), "articulations": p.get("articulations", [])}))
        group = scene_id.rsplit("_", 1)[0]
        return self._scene(scene_id, group, ["rgbd"], CoordinateFrame(name="multiscan_scan", unit_to_metre=1.0, units="m", camera_axes="opengl",
                                                                      note="ARKit camera axes (y up, z backward); align.json NOT applied"),
                           images=images, annotations=an, notes=notes, extra={"environment": group})


ADAPTERS: dict[str, type[Adapter]] = {c.name: c for c in (ScanNetAdapter, ScanNetPPAdapter, ARKitScenesAdapter, Structured3DAdapter, RScan3Adapter,
                                                          MultiScanAdapter)}
