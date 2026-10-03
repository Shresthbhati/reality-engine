"""The canonical Reality Engine demo journey, run on REAL photographs through the REAL application.

    STEP 1   6 photos                       -> world V1
    STEP 2   add 4 more photos              -> V2 (same world)
    STEP 3   add 10 more photos             -> V3 (same world)
    STEP 4   inspect "What changed" (V1 -> V2 -> V3), the ten change categories
    STEP 5   inspect the previous version   (V1 is byte-identical to what it was; what it was built from)
    STEP 6   export the current world       (glTF, USDA, IFC, CityGML, CityJSON, Blender)

It drives the same HTTP routes the Studio calls (in-process FastAPI TestClient over a fresh database and WorldStore),
with real COLMAP and the real depth/perception stages -- nothing is mocked. Every number in the output is MEASURED in
this run and written to ``demo_metrics.json``; the script asserts only structural facts (one world, three distinct
chained versions, evidence accumulates, earlier versions immutable, every export non-empty and parseable). It never
checks a number against a hard-coded expected value, so a different machine or a different engine build produces a
different -- and honestly reported -- result.

Usage:
    python scripts/demo_journey.py [--dataset datasets/south_building/images] [--out DIR] [--keep] [--steps 6,4,10]

Needs: COLMAP on PATH (or REALITY_COLMAP), the South Building photographs (``python scripts/fetch_south_building.py``),
and the API dependencies (fastapi, sqlalchemy, aiosqlite, httpx). Exit status: 0 = journey completed and all structural
checks held; 1 = a check failed (the metrics file still records what happened); 2 = prerequisites missing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "south_building" / "images"
#: windows into the name-sorted South Building image list, chosen from the measured pairwise SIFT-match matrix
#: (see tests/integration/test_progressive_product_journey.py): SIX strongly overlap; ADD_4 continues the walk;
#: ADD_10 is weakly linked to SIX (12-40 matches).
WINDOWS = {"six": (15, 21), "add4": (21, 25), "add10": (5, 15)}
CORE_FORMATS = ("gltf", "usda", "ifc", "citygml", "cityjson", "blender")
CHANGE_KINDS = ("preserved", "refined", "extended", "reduced", "split", "merge", "regrouped", "ambiguous", "removed",
                "new")


class JourneyFailure(AssertionError):
    pass


def _check(ok: bool, what: str, checks: list) -> None:
    checks.append({"check": what, "ok": bool(ok)})
    if not ok:
        raise JourneyFailure(what)


def _prerequisites(dataset: Path) -> list:
    missing = []
    if not shutil.which(os.environ.get("REALITY_COLMAP", "colmap")):
        missing.append("COLMAP is not on PATH (set REALITY_COLMAP to its path)")
    if not dataset.is_dir() or not sorted(dataset.glob("*.JPG")):
        missing.append(f"South Building photographs not found in {dataset} (run scripts/fetch_south_building.py)")
    for mod in ("fastapi", "sqlalchemy", "aiosqlite", "httpx"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(f"python module {mod!r} is not installed")
    return missing


def _app(root: Path):
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(root / 'app.db').as_posix()}"
    os.environ["STORAGE_ROOT"] = str(root / "artifacts")
    os.environ["WORLDSTORE_ROOT"] = str(root / "ws")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    return TestClient(main_mod.app)


def _post(c, paths, world_id=None):
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
    r = c.post("/api/reconstructions", files=files, data={"world_id": world_id} if world_id else {})
    if r.status_code != 202:
        raise JourneyFailure(f"upload refused: {r.status_code} {r.text[:300]}")
    return r.json()


def _settle(c, world_id, timeout=1500.0):
    end, st = time.time() + timeout, None
    while time.time() < end:
        st = c.get(f"/api/worlds/{world_id}/status").json()
        if not st["in_progress"]:
            return st
        time.sleep(1.0)
    raise JourneyFailure(f"world {world_id} never settled; last status: {json.dumps(st)[:400]}")


def _step_metrics(label: str, n_uploaded: int, seconds: float, st: dict) -> dict:
    m = st.get("model") or {}
    phys = m.get("physical") or {}
    strat = m.get("strategy") or {}
    return {
        "step": label, "photos_uploaded_this_step": n_uploaded, "seconds": round(seconds, 1),
        "state": st["state"], "version": m.get("version_id"), "level": m.get("level"), "level_name": m.get("level_name"),
        "model_state": m.get("model_state"), "outcome": m.get("outcome"),
        "images_used": m.get("images_used"), "images_registered": m.get("images_registered"),
        "scale": m.get("scale"), "dense": m.get("dense"), "verdict": m.get("verdict"),
        "strategy": {k: strat.get(k) for k in ("mode", "reason", "frame", "prior_images", "incremental_registered",
                                               "full_registered", "arbitration")} if strat else None,
        "change_counts": phys.get("counts") if phys.get("available") else None,
        "changes": m.get("changes"), "uncertainties": m.get("uncertainties"),
        "conflicts_unresolved": ((phys.get("conflicts") or {}).get("unresolved") if phys else None),
        "last_run": st.get("last_run"),
        "evidence": {"count": st["evidence_summary"]["count"],
                     "in_current_model": sum(1 for e in st["evidence"] if e.get("in_current_model")),
                     "registered": sum(1 for e in st["evidence"] if e.get("registered")),
                     "waiting": sum(1 for e in st["evidence"] if e.get("registered") is False)},
        "versions": [{"label": v["label"], "images_used": v["images_used"], "is_current": v["is_current"]}
                     for v in st["versions"]],
    }


def run_journey(dataset: Path = DEFAULT_DATASET, out: Path | None = None, windows=WINDOWS) -> dict:
    """Run the journey; returns the metrics dict (also written to ``out/demo_metrics.json``). Raises JourneyFailure
    after writing the metrics if a structural check fails."""
    imgs = sorted(dataset.glob("*.JPG"))
    out = Path(out) if out else Path(tempfile.mkdtemp(prefix="reality_demo_"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "exports").mkdir(exist_ok=True)
    plan = [("STEP 1: first photos -> V1", imgs[slice(*windows["six"])]),
            ("STEP 2: add evidence -> V2", imgs[slice(*windows["add4"])]),
            ("STEP 3: add more evidence -> V3", imgs[slice(*windows["add10"])])]
    metrics: dict = {"dataset": str(dataset), "photos_available": len(imgs), "steps": [], "checks": [],
                     "root": str(out), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    checks = metrics["checks"]
    started = time.time()
    try:
        with _app(out) as c:
            wid, v_worldir, v_ids = None, {}, []
            for label, paths in plan:
                t0 = time.time()
                body = _post(c, paths, wid)
                if wid is None:
                    wid = body["world_id"]
                    metrics["world_id"] = wid
                else:
                    _check(body["world_id"] == wid and body["created_world"] is False,
                           f"{label}: the photos joined the SAME world", checks)
                st = _settle(c, wid)
                m = _step_metrics(label, len(paths), time.time() - t0, st)
                metrics["steps"].append(m)
                _check(m["state"] in ("READY_TO_INSPECT", "PARTIALLY_COMPLETE"),
                       f"{label}: reconstruction finished with a model (state {m['state']})", checks)
                v_ids.append(m["version"])
                # each version is snapshotted at birth: STEP 5 proves it never changes
                v_worldir[m["version"]] = c.get(f"/api/worlds/{wid}/worldir", params={"version": m["version"]}).json()
            _check(len(set(v_ids)) == 3, "three distinct immutable versions V1, V2, V3", checks)
            for m in metrics["steps"][1:]:
                arb = (m["strategy"] or {}).get("arbitration") or {}
                if arb.get("choice") == "incremental":
                    shift = ((arb.get("incremental") or {}).get("measures") or {}).get("frame_shift") or {}
                    _check(shift.get("changed") is not True,
                           f"{m['step']}: an incremental build keeps the established coordinate frame "
                           f"(measured rotation {shift.get('rotation_deg')} deg)", checks)
            _check(len(c.get("/api/worlds").json()["items"]) == 1, "one persistent World identity", checks)
            final = c.get(f"/api/worlds/{wid}/status").json()
            by_label = {v["label"]: v for v in final["versions"]}
            _check(set(by_label) == {"V1", "V2", "V3"}, "the status lists V1, V2 and V3", checks)
            _check(by_label["V2"]["parent_version_id"] == v_ids[0] and by_label["V3"]["parent_version_id"] == v_ids[1],
                   "versions are chained V1 -> V2 -> V3", checks)
            used = [by_label[k]["images_used"] for k in ("V1", "V2", "V3")]
            _check(used == sorted(used) and used[0] < used[-1], f"evidence accumulates across versions {used}", checks)
            metrics["accumulated_evidence"] = used

            # STEP 4: what changed
            diffs = {}
            for a, b in ((0, 1), (1, 2), (0, 2)):
                d = c.get(f"/api/worlds/{wid}/diff", params={"base": v_ids[a], "head": v_ids[b]}).json()
                diffs[f"V{a + 1}->V{b + 1}"] = d["categories"]
            metrics["step4_what_changed"] = {
                "diff_categories": diffs,
                "change_counts_per_version": {s["step"]: s["change_counts"] for s in metrics["steps"]},
            }
            same = c.get(f"/api/worlds/{wid}/diff", params={"base": v_ids[2], "head": v_ids[2]}).json()
            _check(sum(same["categories"].values()) == 0, "a version compared with itself shows no changes", checks)
            counted = [s["change_counts"] for s in metrics["steps"][1:] if s["change_counts"]]
            _check(all(set(cc) == set(CHANGE_KINDS) for cc in counted),
                   "the status reports all ten change categories", checks)

            # STEP 5: the previous version
            for vid in v_ids:
                again = c.get(f"/api/worlds/{wid}/worldir", params={"version": vid}).json()
                _check(json.dumps(again, sort_keys=True) == json.dumps(v_worldir[vid], sort_keys=True),
                       f"version {vid[:12]} is byte-identical to the moment it was created (immutable)", checks)
            v1 = by_label["V1"]
            metrics["step5_previous_version"] = {
                "version": "V1", "id": v_ids[0], "images_used": v1["images_used"], "model_state": v1.get("model_state"),
                "evidence_ids_recorded": v1.get("evidence_ids") is not None,
                "photos_in_v1": len(v1.get("evidence_ids") or []),
                "photos_placed_in_v1": len(v1.get("registered_ids") or []),
                "photos_added_after_v1": len(final["evidence"]) - len(v1.get("evidence_ids") or []),
            }
            _check(metrics["step5_previous_version"]["evidence_ids_recorded"],
                   "V1 records which photos it was built from", checks)

            # STEP 6: export the current world
            exp = {}
            for fmt in CORE_FORMATS:
                r = c.post(f"/api/worlds/{wid}/export", json={"format": fmt})
                if r.status_code != 200:
                    exp[fmt] = {"ok": False, "status": r.status_code, "detail": r.text[:200]}
                    continue
                body = r.json()
                data = c.get(body["download_url"]).content
                (out / "exports" / f"world.{fmt}").write_bytes(data)
                exp[fmt] = {"ok": True, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16],
                            "entities_exported": len(body["entities_exported"]),
                            "entities_skipped": len(body["entities_skipped"]),
                            "skip_reasons": sorted(set(body["skip_reasons"]))[:3]}
            metrics["step6_exports"] = exp
            for fmt, e in exp.items():
                _check(e["ok"] and e["bytes"] > 0, f"export {fmt}: produced a file ({e.get('status', 'ok')})", checks)
                _check(e["entities_exported"] > 0, f"export {fmt}: exported entities from the reconstructed world", checks)
            metrics["parsed_back"] = _parse_back(out / "exports")
            for fmt, ok in metrics["parsed_back"].items():
                _check(ok is not False, f"export {fmt}: parsed back with an independent reader", checks)
            metrics["final_status"] = _step_metrics("final", 0, 0.0, final)
    except JourneyFailure as exc:
        metrics["failure"] = str(exc)
    metrics["seconds_total"] = round(time.time() - started, 1)
    metrics["passed"] = "failure" not in metrics
    (out / "demo_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    if not metrics["passed"]:
        raise JourneyFailure(metrics["failure"])
    return metrics


def _parse_back(exports: Path) -> dict:
    """Read each export with a reader that is not the exporter. None = no reader available (not claimed)."""
    import ast
    import xml.etree.ElementTree as ET

    res: dict = {}
    try:
        g = json.loads((exports / "world.gltf").read_text(encoding="utf-8"))
        res["gltf"] = bool(g["nodes"]) and g["asset"]["version"] == "2.0"
    except Exception:  # noqa: BLE001
        res["gltf"] = False
    try:
        res["usda"] = (exports / "world.usda").read_text(encoding="utf-8").startswith("#usda 1.0")
    except Exception:  # noqa: BLE001
        res["usda"] = False
    try:
        cj = json.loads((exports / "world.cityjson").read_text(encoding="utf-8"))
        res["cityjson"] = cj["type"] == "CityJSON" and bool(cj["CityObjects"])
    except Exception:  # noqa: BLE001
        res["cityjson"] = False
    try:
        ET.fromstring((exports / "world.citygml").read_text(encoding="utf-8"))
        res["citygml"] = True
    except Exception:  # noqa: BLE001
        res["citygml"] = False
    try:
        ast.parse((exports / "world.blender").read_text(encoding="utf-8"))
        res["blender"] = True
    except Exception:  # noqa: BLE001
        res["blender"] = False
    try:
        import ifcopenshell

        res["ifc"] = len(ifcopenshell.open(str(exports / "world.ifc")).by_type("IfcBuildingElement")) > 0
    except ImportError:
        res["ifc"] = None
    except Exception:  # noqa: BLE001
        res["ifc"] = False
    return res


def format_report(m: dict) -> str:
    lines = ["# Reality Engine demo journey", "",
             f"world `{m.get('world_id')}` -- {m['photos_available']} photographs available, total {m['seconds_total']} s, "
             f"{'PASSED' if m['passed'] else 'FAILED: ' + m.get('failure', '')}", ""]
    for s in m["steps"]:
        d = s["dense"] or {}
        st = s["strategy"] or {}
        why = f" -- {st['arbitration']['why']}" if st.get("arbitration") else ""
        lines += [f"## {s['step']}",
                  f"- {s['images_used']} photos used, {s['images_registered']} placed; model {s['model_state']}; "
                  f"{'dense' if d.get('state') == 'dense' else 'sparse'}; scale {(s['scale'] or {}).get('state')}; "
                  f"{s['seconds']} s",
                  f"- built by: {st.get('mode')} ({st.get('frame')} frame){why}",
                  f"- verdict: {s['verdict']}; change counts: {s['change_counts']}", ""]
    if "step4_what_changed" in m:
        lines += ["## STEP 4: what changed",
                  f"- diff categories: `{json.dumps(m['step4_what_changed']['diff_categories'])}`", ""]
    if "step5_previous_version" in m:
        lines += ["## STEP 5: previous version", f"- `{json.dumps(m['step5_previous_version'])}`", ""]
    if "step6_exports" in m:
        lines += ["## STEP 6: export"] + [f"- {k}: {v}" for k, v in m["step6_exports"].items()] + \
            [f"- parsed back: {m.get('parsed_back')}", ""]
    lines += ["## checks"] + [f"- {'ok ' if c['ok'] else 'FAIL'} {c['check']}" for c in m["checks"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out", default=None, help="output directory (default: a temp directory)")
    ap.add_argument("--keep", action="store_true", help="print how to open the kept database in the Studio")
    ap.add_argument("--steps", default="", help="override the photo counts, e.g. 6,4,10 (default: the measured windows)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    dataset = Path(args.dataset)
    missing = _prerequisites(dataset)
    if missing:
        print("cannot run the demo journey:\n  - " + "\n  - ".join(missing), file=sys.stderr)
        return 2
    windows = WINDOWS
    if args.steps:
        n = [int(x) for x in args.steps.split(",")]
        windows = {"six": (15, 15 + n[0]), "add4": (15 + n[0], 15 + n[0] + n[1]), "add10": (5, 5 + n[2])}
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="reality_demo_"))
    try:
        m = run_journey(dataset, out, windows)
        code = 0
    except JourneyFailure as exc:
        print(f"JOURNEY FAILED: {exc}", file=sys.stderr)
        code = 1
        m = json.loads((out / "demo_metrics.json").read_text(encoding="utf-8"))
    report = format_report(m)
    (out / "DEMO_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"metrics: {out / 'demo_metrics.json'}")
    if args.keep:
        print(f"kept. Open it in the Studio:\n  DATABASE_URL=sqlite+aiosqlite:///{(out / 'app.db').as_posix()} "
              f"STORAGE_ROOT={out / 'artifacts'} WORLDSTORE_ROOT={out / 'ws'} uvicorn apps.api.main:app --port 8100")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
