#!/usr/bin/env python3
"""Per-stage evidence report for a REAL image folder, through the production code path.

What it runs (the same functions ``apps/api/jobs.py::_run_reconstruct_session_inner`` runs, minus the DB/world-store
bookkeeping that needs a running API):

    inspect_image (evidence_quality) -> jobs._scene_gate -> jobs._intrinsics_options -> VerticalSliceOptions
    -> progressive.run_progressive -> vertical_slice -> ladder.run_fine_stages
       (pose, scale, depth, layout, perception, mesh, dense, detail, finalize)

Every fine stage class is wrapped (no behaviour change) to record, per stage: wall time, peak RSS of this process and of
any ``colmap.exe`` child, input/output counts, the StageResult (status, metrics, warnings, error), the model checkpoints
opened while it ran (path/size/sha256), every WARNING+ log record, and every fallback/skip found in the stage facts.
Nothing is hidden: a fallback that the pipeline records is listed with the effect the pipeline states (or "NOT STATED").

Stages that need the API database (quality_report, compile_validate, changeset_commit) are NOT run here; the report says so.

    RE_MEMORY_ADMISSION=record python scripts/stage_report.py --name south_building --images datasets/south_building/images
    python scripts/stage_report.py --name room11 --images datasets/user_room_nov14/images
    python scripts/stage_report.py --name room6 --images datasets/user_room_nov14/images --subset 6

Output is merged into docs/reconstruction/REAL_RUN_STAGE_REPORT.json (one entry per --name) and the .md is regenerated.
Run ONE dataset per process (RSS is per process; the 6 GB GPU / ~16 GB RAM machine cannot take two heavy jobs).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "docs" / "reconstruction" / "REAL_RUN_STAGE_REPORT.json"
OUT_MD = ROOT / "docs" / "reconstruction" / "REAL_RUN_STAGE_REPORT.md"
MODEL_EXT = (".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt", ".gguf")
SKIP_STATES = {"skipped", "refused", "failed", "unavailable", "rejected", "degraded", "error"}


# ------------------------------------------------------------------ small helpers
def _clean(o, depth=0):
    """JSON-safe, size-bounded view of anything the pipeline hands back."""
    if depth > 6:
        return repr(o)[:120]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else repr(o)
    if isinstance(o, dict):
        return {str(k): _clean(v, depth + 1) for k, v in list(o.items())[:80]}
    if isinstance(o, (list, tuple, set)):
        seq = list(o)
        out = [_clean(v, depth + 1) for v in seq[:40]]
        return out + [f"... +{len(seq) - 40} more"] if len(seq) > 40 else out
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _clean(dataclasses.asdict(o), depth + 1)
    if hasattr(o, "tolist"):
        try:
            return _clean(o.tolist(), depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return repr(o)[:160]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _children_rss_mb() -> float:
    """Working set (MB) of every colmap.exe on the machine (the pipeline runs COLMAP as a child process; this process's RSS
    does not include it). Windows tasklist; 0.0 elsewhere or on failure."""
    if sys.platform != "win32":
        return 0.0
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq colmap.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=5).stdout
        total = 0.0
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 5 and parts[0].lower().startswith("colmap"):
                total += float(parts[4].replace(",", "").replace(" K", "").replace(" ", "").strip('" ')) / 1024.0
        return total
    except Exception:  # noqa: BLE001
        return 0.0


def _gpu_used_mb() -> float | None:
    """Whole-GPU memory in use (MB) via nvidia-smi: the only view that includes COLMAP's child-process VRAM."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return float(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


def _free_ram_mb() -> float | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ae", ctypes.c_ulonglong)]
        ms = MS()
        ms.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ap / 2**20
    except Exception:  # noqa: BLE001
        return None


class Sampler:
    """Peak RSS of this process (+ colmap children) while a stage runs; min system free RAM."""

    def __init__(self, interval=0.25):
        from engine.hardware.admission import read_rss_mb
        self._read, self._iv = read_rss_mb, interval
        self.peak_self = self.peak_child = 0.0
        self.peak_gpu_used = None
        self.min_free = None
        self._stop = threading.Event()
        self._n = 0

    def _tick(self):
        v = self._read() or 0.0
        self.peak_self = max(self.peak_self, v)
        self._n += 1
        if self._n % 4 == 0:  # tasklist is ~50 ms; sample children ~1/s
            self.peak_child = max(self.peak_child, _children_rss_mb())
            g = _gpu_used_mb()
            if g is not None:
                self.peak_gpu_used = g if self.peak_gpu_used is None else max(self.peak_gpu_used, g)
        f = _free_ram_mb()
        if f is not None:
            self.min_free = f if self.min_free is None else min(self.min_free, f)

    def __enter__(self):
        self._tick()
        self._t = threading.Thread(target=lambda: [self._tick() for _ in iter(lambda: self._stop.wait(self._iv), True)], daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=3)
        self._tick()


# ------------------------------------------------------------------ model/checkpoint + log capture
class Recorder:
    """Per-stage capture of opened model files and WARNING+ log records."""

    def __init__(self):
        self.models: dict[str, dict] = {}
        self.logs: list[dict] = []
        self.stage = "setup"
        self._patched = False

    def note_model(self, path, how):
        try:
            p = Path(str(path))
            if p.suffix.lower() in MODEL_EXT and p.is_file() and p.stat().st_size > 1_000_000:
                key = str(p.resolve())
                ent = self.models.setdefault(key, {"path": key, "size_mb": round(p.stat().st_size / 2**20, 1), "stages": [], "opened_via": how})
                if self.stage not in ent["stages"]:
                    ent["stages"].append(self.stage)
        except Exception:  # noqa: BLE001
            pass

    def install(self):
        if self._patched:
            return
        self._patched = True
        rec = self

        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
                rec.note_model(args[0], "python open()")
        sys.addaudithook(hook)

        class H(logging.Handler):
            def emit(self, r):
                if r.levelno >= logging.WARNING:
                    rec.logs.append({"stage": rec.stage, "logger": r.name, "level": r.levelname, "msg": r.getMessage()[:400]})
        logging.getLogger().addHandler(H())
        logging.getLogger().setLevel(logging.WARNING)

    def patch_safetensors(self):
        """Rust safetensors opens files outside the Python audit hook: wrap ``safe_open``/``load_file`` wherever imported."""
        rec = self
        for name, mod in list(sys.modules.items()):
            if mod is None or not name.split('.')[0] in ('safetensors', 'transformers', 'diffusers', 'huggingface_hub'):
                continue  # never touch torch.* (its lazy torch.classes namespace fabricates attributes)
            for attr in ("load_file",):  # safe_open is a torch custom class: wrapping it breaks the pipeline (observed)
                fn = getattr(mod, attr, None)
                if fn is None or getattr(fn, "_stage_report", False) or not callable(fn):
                    continue
                if attr == "safe_open" and isinstance(fn, type):
                    orig = fn

                    def w(path, *a, _o=orig, **k):
                        rec.note_model(path, "safetensors.safe_open")
                        return _o(path, *a, **k)
                else:
                    orig = fn

                    def w(path, *a, _o=orig, **k):
                        rec.note_model(path, "safetensors.load_file")
                        return _o(path, *a, **k)
                w._stage_report = True
                try:
                    setattr(mod, attr, w)
                except Exception:  # noqa: BLE001
                    pass


REC = Recorder()


# ------------------------------------------------------------------ fallback discovery
def walk_fallbacks(facts, path="", out=None):
    """Every skip/refusal/fallback a stage's facts admit to, with the reason and the effect the pipeline itself states."""
    out = [] if out is None else out
    if isinstance(facts, dict):
        st = str(facts.get("status", facts.get("state", ""))).lower()
        keys = [k for k in facts if "fallback" in str(k).lower()]
        if st in SKIP_STATES or keys or facts.get("used_fallback"):
            reason = next((facts[k] for k in ("reason", "note", "error", "detail", "why") if facts.get(k)), None)
            effect = next((facts[k] for k in ("quality_effect", "effect", "impact", "consequence") if facts.get(k)), None)
            out.append({"where": path or "<stage>", "state": st or "fallback", "reason": _clean(reason),
                        "effect_stated_by_pipeline": _clean(effect) if effect else "NOT STATED by the pipeline",
                        "fallback_keys": {k: _clean(facts[k]) for k in keys}})
        for k, v in facts.items():
            walk_fallbacks(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(facts, (list, tuple)):
        for i, v in enumerate(facts[:20]):
            walk_fallbacks(v, f"{path}[{i}]", out)
    return out


def observation_summary(facts: dict) -> dict:
    """Detections accepted vs rejected, with the pipeline's own reasons, from the perception facts."""
    res = facts.get("resolution") or {}
    dropped: dict = {}
    for d in res.get("dropped_detail") or []:
        dropped[str(d.get("reason"))] = dropped.get(str(d.get("reason")), 0) + 1
    dets = facts.get("detections_2d") or {}
    return {"detections_2d": sum(len(v) for v in dets.values()), "masks_considered": facts.get("masks_considered"),
            "lifted_to_3d": facts.get("hypotheses_lifted"), "merged_candidates": facts.get("candidates_merged"),
            "promoted_entities": facts.get("entities_promoted"), "association_decisions": res.get("decisions"),
            "dropped_candidates_by_reason": dropped, "mask_source": facts.get("masks"),
            "appearance": res.get("appearance")}


def counts_for(stage, ctx):
    """(inputs, outputs) counts read defensively from the pipeline's own state objects."""
    s = ctx.state
    g = lambda o, *names: next((getattr(o, n) for n in names if hasattr(o, n)), None)  # noqa: E731
    inp, out = {}, {}
    try:
        if stage == "pose":
            inp = {"images": len(s["evidence_items"])}
            if "pose" in s:
                r = s["pose"].result
                out = {"cameras_registered": len(r.camera_poses), "sparse_points": len(r.points),
                       "registration_status": r.registration_status}
        elif stage == "scale" and "scale" in s:
            o = s["scale"]
            out = {"scale_state": o.scale_state, "meters_per_unit": o.meters_per_unit, "note": o.scale_note[:200]}
        elif stage == "depth" and "depth" in s:
            o = s["depth"]
            inp = {"cameras": len(s["pose"].result.camera_poses), "scale_state": s["scale"].scale_state}
            out = {"metric_depth_maps": len(o.metric_maps)}
        elif stage == "layout" and "layout" in s:
            w = s["layout"].world
            out = {"entities": len(w.entities)}
        elif stage == "perception":
            inp = {"metric_depth_maps": len(s["depth"].metric_maps), "images": len(s["pose"].evidence_items)}
            if "perception" in s:
                f = s["perception"].facts
                out = {k: f.get(k) for k in ("status", "views_segmented", "masks_considered", "hypotheses_lifted",
                                             "candidates_merged", "entities_promoted", "objects") if k in f}
        elif stage in ("mesh", "dense", "detail", "finalize") and stage in s:
            o = s[stage]
            out = {"keys": [f.name for f in dataclasses.fields(o)][:10]} if dataclasses.is_dataclass(o) else {}
    except Exception as exc:  # noqa: BLE001 - counting must never break a stage
        out["count_error"] = repr(exc)
    return _clean(inp), _clean(out)


def instrument(report_stages):
    """Replace ladder.fine_chain so every fine stage records its measurements. Behaviour is unchanged."""
    from engine.pipeline.stages import ladder

    orig = ladder.fine_chain

    def make(cls):
        class Wrapped(cls):
            def run(self, ctx):
                name = cls.spec.name
                REC.stage = name
                REC.patch_safetensors()
                entry = {"stage": name, "status": "running"}
                report_stages.append(entry)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except Exception:  # noqa: BLE001
                    pass
                inp, _ = counts_for(name, ctx)
                t0 = time.perf_counter()
                with Sampler() as sm:
                    try:
                        res = super().run(ctx)
                        err = None
                    except Exception as exc:  # noqa: BLE001
                        res, err = None, {"type": type(exc).__name__, "message": str(exc)[:600],
                                          "trace_tail": traceback.format_exc().splitlines()[-6:]}
                        raise
                    finally:
                        REC.stage = "ladder_after_" + name  # later model loads belong to the ladder's fallback path, not this stage
                        entry["wall_s"] = round(time.perf_counter() - t0, 2)
                        entry["peak_rss_mb"] = round(sm.peak_self, 0)
                        entry["peak_child_colmap_rss_mb"] = round(sm.peak_child, 0)
                        entry["peak_gpu_used_mb_whole_device"] = sm.peak_gpu_used
                        entry["min_system_free_ram_mb"] = None if sm.min_free is None else round(sm.min_free, 0)
                        try:
                            import torch
                            entry["peak_cuda_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0) if torch.cuda.is_available() else None
                            entry["cuda_residual_after_stage_mb"] = round(torch.cuda.memory_allocated() / 2**20, 0) if torch.cuda.is_available() else None
                        except Exception:  # noqa: BLE001
                            entry["peak_cuda_alloc_mb"] = None
                        _, out = counts_for(name, ctx)
                        entry["inputs"], entry["outputs"] = inp, out
                        if err:
                            entry.update(status="failed", failure=err)
                        elif res is not None:
                            entry.update(status=res.status, metrics=_clean(res.metrics), warnings=_clean(res.warnings))
                            facts = _stage_facts(name, ctx)
                            entry["facts"] = _clean(facts)
                            entry["fallbacks"] = walk_fallbacks(facts)
                            if name == "perception" and isinstance(facts, dict):
                                entry["observations"] = observation_summary(facts)
                            for w in res.warnings:
                                entry["fallbacks"].append({"where": "StageResult.warnings", "state": res.status, "reason": w[:400],
                                                           "effect_stated_by_pipeline": "see reason"})
                return res
        Wrapped.__name__ = cls.__name__
        return Wrapped

    ladder.fine_chain = lambda: [make(c) for c in orig()]


def _stage_facts(name, ctx):
    s = ctx.state
    o = s.get(name)
    if o is None:
        return None
    if name == "pose":
        r = o.result
        return {"colmap_gpu_decision": _clean(getattr(o.backend, "gpu_decision", None)),
                "backend": getattr(o.run, "diagnostics", None) and _clean(o.run.diagnostics),
                "registration_status": r.registration_status, "cameras": len(r.camera_poses),
                "notes": _clean(getattr(r, "notes", None))}
    for attr in ("facts", "dense_mvs_facts", "scale_note"):
        if hasattr(o, attr):
            v = getattr(o, attr)
            if name == "dense":
                return {"dense_mvs": o.dense_mvs_facts, "dense_gate": o.dense_gate_facts, "structure": o.structure_facts,
                        "plane_sweep": o.plane_sweep_facts, "fusion": o.fusion_facts}
            return v
    if name == "layout":
        return o.layout_facts
    if name == "depth":
        return o.facts
    return None


# ------------------------------------------------------------------ pipeline driver
def run_pipeline(images: list[Path], work: Path) -> dict:
    os.environ.setdefault("RE_MEMORY_ADMISSION", "record")
    os.environ["WORLDSTORE_ROOT"] = str(work / "worldstore")
    REC.install()
    rep: dict = {"stages": []}

    # --- evidence_resolve: what the job does with stored evidence (readable, non-duplicate)
    from apps.api import jobs
    from engine.pipeline.progressive import EvidenceInput, run_progressive
    from engine.pipeline.vertical_slice import VerticalSliceOptions
    from evidence.image_check import inspect_image
    from evidence.session import EvidenceItem, EvidenceKind

    REC.stage = "evidence_resolve"
    t0 = time.perf_counter()
    resolved = []
    for p in images:
        sha = jobs._file_sha256(p)
        resolved.append((p, sha))
    seen, uniq, dupes = {}, [], []
    for p, sha in resolved:
        (dupes if sha in seen else uniq).append((p, sha))
        seen.setdefault(sha, p)
    rep["stages"].append({"stage": "evidence_resolve", "status": "succeeded", "wall_s": round(time.perf_counter() - t0, 2),
                          "inputs": {"files": len(images)}, "outputs": {"unique": len(uniq), "exact_duplicates_dropped": len(dupes)}})

    # --- evidence_quality
    REC.stage = "evidence_quality"
    t0 = time.perf_counter()
    with Sampler() as sm:
        facts = [inspect_image(p, None) for p, _ in uniq]
    cls_counts: dict = {}
    for f in facts:
        cls_counts[f.evidence_class] = cls_counts.get(f.evidence_class, 0) + 1
    exif_f35 = [f.exif.get("focal_length_35mm") for f in facts]
    rep["stages"].append({
        "stage": "evidence_quality", "status": "succeeded", "wall_s": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(sm.peak_self, 0), "inputs": {"images": len(uniq)},
        "outputs": {"ok": sum(f.ok for f in facts), "geometry_eligible": sum(f.geometry_eligible for f in facts),
                    "evidence_class": cls_counts, "sizes": sorted({f"{f.width}x{f.height}" for f in facts}),
                    "images_with_exif_focal_35mm": sum(1 for v in exif_f35 if v), "exif_keys_seen": sorted({k for f in facts for k in f.exif})},
        "rejected": [{"name": p.name, "reason": f.reason or f"class={f.evidence_class}: {f.class_basis}"}
                     for (p, _), f in zip(uniq, facts) if not f.geometry_eligible]})
    eval_inputs = [EvidenceInput(item=EvidenceItem(id=f"ev_{i:03d}", kind=EvidenceKind.PHOTO, source_uri=p.resolve().as_uri(), sha256=sha),
                                 facts=f, name=p.name, path=p.resolve()) for i, ((p, sha), f) in enumerate(zip(uniq, facts))]
    rep["id_to_name"] = {e.item.id: e.name for e in eval_inputs}

    # --- scene identity gate
    REC.stage = "scene_gate"
    t0 = time.perf_counter()
    with Sampler() as sm:
        scene = jobs._scene_gate("stage_report_world", eval_inputs, {"s": [e.item.id for e in eval_inputs]})
    excluded = scene.get("excluded") or {}
    rep["stages"].append({"stage": "scene_gate", "status": scene.get("state"), "wall_s": round(time.perf_counter() - t0, 2),
                          "peak_rss_mb": round(sm.peak_self, 0), "inputs": {"images": len(eval_inputs)},
                          "outputs": {"groups": [len(g["members"]) for g in scene.get("groups", [])],
                                      "verified_pairs": scene.get("verified_pairs"), "excluded": len(excluded),
                                      "unlinked": _clean(scene.get("unlinked")), "uncertain_links": _clean(scene.get("uncertain_links"))},
                          "rejected": [{"id": k, "name": rep["id_to_name"].get(k), "reason": v} for k, v in excluded.items()],
                          "scene": _clean({k: v for k, v in scene.items() if k not in ("excluded",)})})
    run_inputs = [e for e in eval_inputs if e.item.id not in excluded]

    # --- options exactly as seg_reconstruct builds them (no world => no head, no scale references)
    from engine.pipeline.dense_gate import dense_auto_enabled
    from reconstruction.colmap_session import ColmapSession
    from engine.pipeline.open_vocab_furniture import combined_furniture_detector
    from engine.pipeline.single_image_furniture import maskrcnn_furniture_detector
    from engine.pipeline.single_image_layout import make_layout_analyzer
    from world_ir.artifact_store import FileArtifactStore

    store = FileArtifactStore(work / "pipeline-artifacts")
    session = ColmapSession(work / "colmap-sessions" / "world")
    intr_kwargs, intr_rec = jobs._intrinsics_options([e.facts for e in run_inputs], session, None)
    rep["intrinsics_record_before_solve"] = _clean(intr_rec)
    options = VerticalSliceOptions(
        artifact_store=store, colmap_session=session,
        perception_model="maskrcnn" if os.environ.get("REALITY_OPEN_VOCAB") == "0" else "grounded",
        **intr_kwargs, measured_baselines=(), dense_auto=dense_auto_enabled(),
        head_snapshot=None, head_conflicts=None, head_frame=None, head_unit_k=None,
        **jobs._dense_quality_options(), **jobs._dense_cpu_options(None))
    rep["options"] = _clean({"perception_model": options.perception_model, "depth_model": options.depth_model,
                             "intrinsics": options.intrinsics, "intrinsics_trust": options.intrinsics_trust,
                             "image_size": options.image_size, "measured_baselines": len(options.measured_baselines),
                             "dense_auto": options.dense_auto, "perception_mask_source": options.perception_mask_source,
                             "perception_appearance": options.perception_appearance})

    layout_analyzer, unload_layout = make_layout_analyzer()
    from engine.pipeline import vertical_slice as _vs
    fine_stages: list = []
    instrument(fine_stages)
    fine: dict = {}
    REC.stage = "reconstruct"
    t0 = time.perf_counter()
    prog, ladder_exc = None, None
    with Sampler() as sm_all:
        try:
            prog = run_progressive(
                run_inputs, vs_options=options, prior_ids=[], artifact_store=store,
                depth_model=(options.depth_model or "DPT_Hybrid"),
                furniture_detector=(maskrcnn_furniture_detector if os.environ.get("REALITY_OPEN_VOCAB") == "0" else combined_furniture_detector),
                layout_analyzer=layout_analyzer,
                vertical_slice_fn=lambda ev, o: _vs.vertical_slice(ev, o, stage_sink=fine))
        except Exception as exc:  # noqa: BLE001
            ladder_exc = {"type": type(exc).__name__, "message": str(exc)[:600], "trace_tail": traceback.format_exc().splitlines()[-8:]}
        finally:
            unload_layout()
    rep["stages"].extend(fine_stages)
    rep["ladder"] = {"wall_s": round(time.perf_counter() - t0, 2), "peak_rss_mb": round(sm_all.peak_self, 0),
                     "peak_child_colmap_rss_mb": round(sm_all.peak_child, 0), "exception": ladder_exc}
    if prog is not None:
        rep["ladder"].update(_clean({
            "level": prog.level, "level_name": prog.level_name, "model_state": prog.model_state,
            "attempts": prog.attempts, "degraded": prog.degraded, "scale_state": prog.scale_state,
            "registered": len(prog.registered_ids), "inputs": len(run_inputs), "excluded": prog.excluded,
            "entities": len(prog.world.entities),
            "entity_types": _count_types(prog.world)}))
        rep["fallbacks_ladder"] = [{"where": f"ladder.attempts[{i}]", "state": a.get("outcome"), "reason": a.get("detail"),
                                    "effect_stated_by_pipeline": "level achieved: " + str(prog.level_name)}
                                   for i, a in enumerate(prog.attempts) if a.get("outcome") != "succeeded"]
        rep["fallbacks_ladder"] += [{"where": "ladder.degraded", "state": "degraded", "reason": d,
                                     "effect_stated_by_pipeline": "grades the job partial"} for d in prog.degraded]
    if prog is not None:
        rep["geometry_validation"] = geometry_validation(prog.world, prog.camera_poses)
        try:   # kept in the work dir (not the report) so the validator can be re-run offline on exactly this world
            (work / "world.json").write_text(json.dumps({"world": prog.world.to_dict(), "camera_poses": [
                [p[0], list(p[1])] for p in prog.camera_poses]}, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            rep["geometry_validation"]["world_dump_error"] = f"{type(exc).__name__}: {exc}"[:200]
    rep["colmap"] = diagnose_colmap(session, work, rep["id_to_name"])
    rep["models_loaded"] = [dict(m, sha256=_sha256(Path(m["path"]))) for m in REC.models.values()]
    rep["logged_warnings"] = REC.logs[:200]
    rep["not_run_here"] = ["quality_report", "compile_validate", "changeset_commit"]
    return rep


def geometry_validation(world, camera_poses) -> dict:
    """The compile_validate room-geometry validator run on a finished world (the stage itself needs the API DB; the
    validator does not). Same function the stage calls (``engine.validation.room_geometry``); nothing is repaired."""
    from engine.validation.room_geometry import camera_centres_from_poses, validate_room_geometry

    centres = camera_centres_from_poses(camera_poses)
    rep = validate_room_geometry(world, camera_centres=centres)
    d = rep.to_dict()
    d["camera_centres_supplied"] = len(centres)
    d["entity_types"] = _count_types(world)
    return d


def synthetic_geometry_runs() -> dict:
    """The same validator on SYNTHETIC worlds (compiled by the production compiler from generated point clouds). A separate
    evidence class from the real-image runs: it proves the validator accepts known-good geometry, nothing about photos."""
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from tests.test_canonical_interior import UP, _canonical_interior_scene

    out: dict = {}
    result = _canonical_interior_scene()
    for pb in (False, True):
        world, _ = compile_reconstruction_to_world(result, CompileOptions(
            seed=42, up=UP, promote_corridors=True, promote_windows=True, promote_stairs=True, promote_building=pb))
        out[f"canonical_interior_promote_building_{pb}"] = geometry_validation(
            world, [(c.evidence_id, c.position, c.rotation) for c in result.camera_poses])
    try:
        from scripts.room_fidelity_report import build_and_compile
        _truth, _obs, world, _diag = build_and_compile("albedo", 0.08, True, "full")
        out["room_fidelity_fixture"] = geometry_validation(world, [])
    except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
        out["room_fidelity_fixture"] = {"verdict": "not_run", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def _count_types(world):
    out: dict = {}
    for e in world.entities.values():
        k = getattr(getattr(e, "type", None), "value", str(getattr(e, "type", "?")))
        out[k] = out.get(k, 0) + 1
    return out


# ------------------------------------------------------------------ COLMAP evidence (database + sparse models + SIFT check)
def diagnose_colmap(session, work: Path, id_to_name: dict) -> dict:
    import sqlite3

    res: dict = {}
    for label in ("staging", "current"):
        root = Path(getattr(session, label))
        db = root / "database.db"
        if not db.exists():
            continue
        d = {"root": str(root)}
        sparse = root / "sparse"
        models = sorted(p.name for p in sparse.iterdir() if p.is_dir()) if sparse.exists() else []
        d["sparse_model_dirs"] = models
        d["models"] = []
        for m in models:
            tmp = work / "txt" / label / m
            tmp.mkdir(parents=True, exist_ok=True)
            subprocess.run([os.environ.get("COLMAP", "colmap"), "model_converter", "--input_path", str(sparse / m), "--output_path", str(tmp),
                            "--output_type", "TXT"], capture_output=True, text=True)
            names, pts = [], 0
            try:
                lines = (tmp / "images.txt").read_text().splitlines()
                names = [l.split()[-1] for l in lines if l and not l.startswith("#")][::2]
                pts = sum(1 for l in (tmp / "points3D.txt").read_text().splitlines() if l and not l.startswith("#"))
            except OSError:
                pass
            cams = []
            try:
                cams = [l for l in (tmp / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")]
            except OSError:
                pass
            d["models"].append({"model": m, "registered_images": len(names), "points3D": pts, "cameras_txt": cams[:6],
                                "image_names": names})
        con = sqlite3.connect(str(db))
        d["images_in_db"] = con.execute("select count(*) from images").fetchone()[0]
        d["images_with_keypoints"] = con.execute("select count(*) from keypoints where rows>0").fetchone()[0]
        d["keypoints_per_image"] = dict(sorted(((n, r) for n, r in con.execute(
            "select i.name, k.rows from images i join keypoints k on k.image_id=i.image_id").fetchall()))[:64])
        d["cameras_in_db"] = [dict(zip(("camera_id", "model", "w", "h", "params", "prior_focal"),
                                       (r[0], r[1], r[2], r[3], list(__import__("numpy").frombuffer(r[4], dtype="float64")), r[5])))
                              for r in con.execute("select camera_id, model, width, height, params, prior_focal_length from cameras").fetchall()]
        d["matches_rows"] = con.execute("select count(*) from matches where rows>0").fetchone()[0]
        tv = con.execute("select pair_id, rows, config from two_view_geometries").fetchall()
        d["two_view_geometries"] = {"pairs_total": len(tv), "pairs_with_inliers": sum(1 for _, r, _ in tv if r > 0),
                                    "inlier_hist": _hist([r for _, r, _ in tv]), "config_counts": _count([c for _, _, c in tv])}
        names = dict(con.execute("select image_id, name from images").fetchall())
        pairs = []
        for pid, rows, cfg in tv:
            i2, i1 = divmod(pid, 2147483647)  # COLMAP: pair_id = id1 * 2147483647 + id2
            pairs.append({"a": names.get(i2), "b": names.get(i1), "inliers": rows, "config": cfg})
        pairs.sort(key=lambda p: -p["inliers"])
        d["top_pairs"] = pairs[:12]
        d["pairs_with_inliers_ge_15"] = sum(1 for p in pairs if p["inliers"] >= 15)
        d["pairs_with_inliers_ge_50"] = sum(1 for p in pairs if p["inliers"] >= 50)
        con.close()
        res[label] = d
    return res


def _hist(vals):
    edges = [0, 1, 15, 30, 50, 100, 200, 500, 10**9]
    out = {}
    for a, b in zip(edges, edges[1:]):
        out[f"{a}-{b - 1 if b < 10**9 else '+'}"] = sum(1 for v in vals if a <= v < b)
    return out


def _count(vals):
    out: dict = {}
    for v in vals:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def sift_pairwise(images: list[Path], max_side=1024) -> dict:
    """INDEPENDENT of COLMAP: SIFT + ratio test + RANSAC fundamental matrix for every pair (cv2). The fraction of
    pairs with a geometrically consistent match set is a data property, not a COLMAP-configuration property."""
    import cv2
    import numpy as np

    sift = cv2.SIFT_create(4000)
    feats = []
    for p in images:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        s = max_side / max(im.shape)
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
        feats.append(sift.detectAndCompute(im, None))
    bf = cv2.BFMatcher()
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            (ki, di), (kj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(ki) < 8 or len(kj) < 8:
                pairs.append({"a": images[i].name, "b": images[j].name, "ratio_matches": 0, "ransac_inliers": 0})
                continue
            good = [m for m, n in (x for x in bf.knnMatch(di, dj, k=2) if len(x) == 2) if m.distance < 0.8 * n.distance]
            inl = 0
            if len(good) >= 8:
                a = np.float32([ki[m.queryIdx].pt for m in good])
                b = np.float32([kj[m.trainIdx].pt for m in good])
                _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.999)
                inl = int(mask.sum()) if mask is not None else 0
            pairs.append({"a": images[i].name, "b": images[j].name, "ratio_matches": len(good), "ransac_inliers": inl})
    inl = [p["ransac_inliers"] for p in pairs]
    n = len(images)
    # connected components over pairs with >= 30 RANSAC inliers (union-find)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    idx = {p.name: k for k, p in enumerate(images)}
    for p in pairs:
        if p["ransac_inliers"] >= 30:
            parent[find(idx[p["a"]])] = find(idx[p["b"]])
    comps: dict = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(images[k].name)
    return {"images": n, "pairs": len(pairs), "inlier_hist": _hist(inl),
            "pairs_ge_30": sum(1 for v in inl if v >= 30), "pairs_ge_100": sum(1 for v in inl if v >= 100),
            "median_inliers": float(np.median(inl)) if inl else 0, "components_at_30_inliers": sorted((len(v) for v in comps.values()), reverse=True),
            "top_pairs": sorted(pairs, key=lambda p: -p["ransac_inliers"])[:8]}


# ------------------------------------------------------------------ markdown
def _geometry_line(name: str, gv: dict) -> str:
    chk = {k: v["state"] for k, v in (gv.get("checks") or {}).items()}
    return (f"**Room-geometry validation `{name}`**: verdict `{gv.get('verdict')}`, errors {gv.get('errors')}, warnings {gv.get('warnings')}, "
            f"unknown {gv.get('unknowns')}; checks `{json.dumps(chk)}`; rooms `{json.dumps({k: v.get('closure', {}).get('state') for k, v in (gv.get('rooms') or {}).items()})}`; "
            f"context `{json.dumps({k: gv.get('context', {}).get(k) for k in ('up_axis', 'scale_state', 'walls', 'walls_readable', 'rooms', 'cameras')})}`")


def render_md(data: dict) -> str:
    L = ["# Real-run stage report", "",
         "Generated by `scripts/stage_report.py` (production code path: `jobs._scene_gate`, `jobs._intrinsics_options`, "
         "`progressive.run_progressive`, `vertical_slice` -> `ladder.run_fine_stages`). Numbers below are measurements from the "
         "recorded runs; nothing here asserts physical accuracy. `quality_report`, `compile_validate`, `changeset_commit` need the API "
         "database and are not run by this script.", ""]
    L += data.get("diagnosis_md", [])
    L += data.get("geometry_md", [])
    for name, gv in data.get("synthetic_geometry", {}).items():
        L += [_geometry_line(f"synthetic:{name}", gv), ""] if gv.get("checks") else [f"**Room-geometry validation synthetic:{name}**: `{json.dumps(gv)[:300]}`", ""]
    for name, r in data.get("runs", {}).items():
        if "stages" not in r:
            continue
        L += [f"## Run `{name}`", "", f"- images: `{r['images_dir']}` ({r['n_images']} used), run at {r['when']}, {r['machine']}",
              f"- intrinsics before solve: `{json.dumps(r.get('intrinsics_record_before_solve'))}`",
              f"- ladder: `{json.dumps(r.get('ladder', {}), default=str)[:900]}`", "",
              "| stage | status | wall s | peak RSS MB (proc / colmap child) | min free RAM MB | torch CUDA peak / residual MB | GPU used MB (whole device) | inputs | outputs |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in r["stages"]:
            L.append(f"| {s['stage']} | {s.get('status')} | {s.get('wall_s', '-')} | {s.get('peak_rss_mb', '-')} / {s.get('peak_child_colmap_rss_mb', '-')} "
                     f"| {s.get('min_system_free_ram_mb', '-')} | {s.get('peak_cuda_alloc_mb', '-')} / {s.get('cuda_residual_after_stage_mb', '-')} | {s.get('peak_gpu_used_mb_whole_device', '-')} | `{json.dumps(s.get('inputs', {}))}` | `{json.dumps(s.get('outputs', {}), default=str)[:300]}` |")
        for s_ in r["stages"]:
            if s_.get("observations"):
                L += ["", f"**Observations (perception)**: `{json.dumps(s_['observations'], default=str)[:900]}`"]
        L += ["", "**Failures / rejections**", ""]
        any_f = False
        for s in r["stages"]:
            if s.get("failure"):
                any_f = True
                L.append(f"- `{s['stage']}` FAILED: `{s['failure']['type']}: {s['failure']['message']}`")
            for rej in s.get("rejected", []):
                any_f = True
                L.append(f"- `{s['stage']}` rejected {rej.get('name') or rej.get('id')}: {rej['reason']}")
        if r["ladder"].get("exception"):
            any_f = True
            L.append(f"- ladder raised: `{r['ladder']['exception']}`")
        L += (["- none"] if not any_f else []) + ["", "**Every fallback / skip taken**", ""]
        fb = [(s["stage"], f) for s in r["stages"] for f in s.get("fallbacks", [])] + [("ladder", f) for f in r.get("fallbacks_ladder", [])]
        for st, f in fb:
            L.append(f"- `{st}` @ `{f['where']}` [{f['state']}]: {json.dumps(f['reason'], default=str)[:350]} -- effect: {json.dumps(f['effect_stated_by_pipeline'], default=str)[:200]}")
        L += (["- none recorded"] if not fb else []) + ["", "**Models / checkpoints actually opened**", ""]
        for m in r.get("models_loaded", []):
            L.append(f"- `{m['path']}` {m['size_mb']} MB sha256 `{m['sha256'][:16]}...` stages {m['stages']} via {m['opened_via']}")
        L += (["- none captured (see caveat in README of this report)"] if not r.get("models_loaded") else []) + [""]
        for lbl, d in (r.get("colmap") or {}).items():
            L += [f"**COLMAP `{lbl}`**: models {d['sparse_model_dirs']}, registered per model "
                  f"{[m['registered_images'] for m in d['models']]}, images in db {d['images_in_db']}, "
                  f"two_view_geometries {json.dumps(d['two_view_geometries'])}", ""]
        if r.get("sift_check"):
            L += [f"**Independent SIFT+RANSAC check**: `{json.dumps({k: v for k, v in r['sift_check'].items() if k != 'top_pairs'})}`", ""]
        gv = r.get("geometry_validation")
        if gv:
            L += [_geometry_line(name, gv), ""]
        L += [f"warnings logged: {len(r.get('logged_warnings', []))}", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--images")
    ap.add_argument("--subset", type=int, default=0, help="use only the first N images (sorted by name)")
    ap.add_argument("--same-size", action="store_true", help="keep only images of the most common pixel size (isolates the mixed-size camera handling)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--sift-only", action="store_true", help="only the independent SIFT+RANSAC pair check (no pipeline)")
    ap.add_argument("--no-sift", action="store_true")
    ap.add_argument("--synthetic-geometry", action="store_true",
                    help="only run the room-geometry validator on the synthetic worlds and merge it into the JSON (no pipeline)")
    a = ap.parse_args()
    if a.synthetic_geometry:
        data = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"runs": {}}
        data["synthetic_geometry"] = _clean(synthetic_geometry_runs())
        OUT_JSON.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        OUT_MD.write_text(render_md(data), encoding="utf-8")
        print("synthetic geometry merged:", {k: v.get("verdict") for k, v in data["synthetic_geometry"].items()})
        return
    if not a.name or not a.images:
        ap.error("--name and --images are required (unless --synthetic-geometry)")
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in exts)
    if a.subset:
        imgs = imgs[:a.subset]
    if a.same_size:
        from collections import Counter
        from PIL import Image
        sz = {p: Image.open(p).size for p in imgs}
        top = Counter(sz.values()).most_common(1)[0][0]
        imgs = [p for p in imgs if sz[p] == top]
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix=f"stage_report_{a.name}_"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COLMAP", os.environ.get("COLMAP_BINARY", "colmap"))
    t0 = time.time()
    gpu0 = _gpu_used_mb()  # VRAM already taken by OTHER processes (desktop, other sessions) before this run starts
    if a.sift_only:
        out = {"sift_check": sift_pairwise(imgs)}
    else:
        out = run_pipeline(imgs, work)
        if not a.no_sift:
            out["sift_check"] = sift_pairwise(imgs)
    out.update(gpu_used_mb_before_run=gpu0, images_dir=str(a.images), n_images=len(imgs), image_files=[p.name for p in imgs], workdir=str(work),
               when=time.strftime("%Y-%m-%dT%H:%M:%S"), total_wall_s=round(time.time() - t0, 1),
               machine="RTX 4050 6GB laptop, Windows; COLMAP " + os.environ.get("COLMAP", "colmap"))
    # keep the run even if merging into the shared report fails (a 10+ minute run was once lost to a transient OneDrive
    # EINVAL on the shared file): write the standalone result first, then retry the merge
    (work / "run_result.json").write_text(json.dumps(_clean(out), indent=1, default=str), encoding="utf-8")
    data = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"runs": {}}
    data["runs"][a.name] = _clean(out)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            OUT_JSON.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
            break
        except OSError as exc:
            print(f"write of {OUT_JSON} failed ({exc}); standalone result kept at {work / 'run_result.json'}; retry {attempt + 1}/6")
            time.sleep(5)
    else:
        raise SystemExit(f"could not write {OUT_JSON}; run saved at {work / 'run_result.json'}")
    if not a.sift_only:
        OUT_MD.write_text(render_md(data), encoding="utf-8")
    print(f"wrote {OUT_JSON} run={a.name} wall={out['total_wall_s']}s workdir={work}")


if __name__ == "__main__":
    main()
