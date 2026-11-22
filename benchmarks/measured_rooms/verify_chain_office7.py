"""Verify the whole persistence chain on a REAL production run, in a FRESH process (the restart):

    REAL INPUT hashes -> evidence -> WorldIR V1 (reload from disk, integrity) -> user correction -> V2 -> diff -> export

    PYTHONPATH=. python benchmarks/measured_rooms/verify_chain_office7.py <run_dir>

``run_dir`` is the output of ``run_office7_production.py``. Everything is read from what that run PERSISTED (app.db, ws/, art/); nothing
is recomputed or defaulted. The correction is a real HTTP call to the real API; it saves a NEW version and the report states whether the
original evidence was preserved. Writes ``chain_report.json`` beside the run. Physical accuracy is reported exactly as the readiness gate
reports it (NOT_MEASURED): this script checks persistence, lineage and export, not metric correctness."""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(run_dir: Path) -> int:
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    wid, jid, tenant = rj["world_id"], rj["job_id"], rj["tenant"]
    os.environ.update({"DATABASE_URL": f"sqlite+aiosqlite:///{(run_dir / 'app.db').as_posix()}", "STORAGE_ROOT": str(run_dir / "art"),
                       "WORLDSTORE_ROOT": str(run_dir / "ws")})
    os.environ.pop("REALITY_TEST_BACKEND", None)
    from fastapi.testclient import TestClient

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    from apps.api import evidence_world_stage as ew
    from engine.multiengine import run_manifest
    from perception.ontology import get_default_ontology
    from world_ir.diff import diff_worlds
    from worldstore.store import WorldStore

    H = {"x-tenant-id": tenant}
    base = f"/api/worlds/{wid}/evidence-world/{jid}"
    evroot = run_dir / "ws" / "evidence-world" / tenant / wid / jid
    rep: dict = {"run_dir": str(run_dir), "world_id": wid, "job_id": jid}

    man = json.loads((evroot / run_manifest.MANIFEST_FILE).read_text(encoding="utf-8"))
    rep["input_hashes"] = [{"sha256": i["sha256"]} for i in man["inputs"]]
    rep["manifest_verification"] = run_manifest.verify(evroot)

    store = WorldStore(evroot / ew.WORLDSTORE_REL)
    v1 = ew.head_version(evroot)
    w1 = store.load_version(v1)
    rep["v1"] = {"version_id": v1, "integrity_problems": store.verify_version(v1), "entities": len(w1.entities), "geometries": len(w1.geometries),
                 "relationship_kinds": dict(Counter(r.kind.value for e in w1.entities.values() for r in e.relationships)),
                 "entity_types": dict(Counter(str(getattr(e.type, "value", e.type)) for e in w1.entities.values()))}

    with TestClient(main_mod.app) as c:
        rd = c.get(f"{base}/readiness", headers=H).json()
        rep["readiness"] = {k: rd.get(k) for k in ("stage_status", "physical_accuracy", "export_ready", "metric_export_ready", "coverage", "integrity",
                                                     "scale", "detection", "stages", "rejected_predictions")}
        rep["quality"] = rd.get("quality")
        ents = c.get(f"{base}/entities", headers=H).json()["entities"]
        target = next((e for e in ents if not e["ontology_path"].endswith("/unknown") and "surf" not in e["entity_id"]), ents[0])
        onto = get_default_ontology()
        new_class = onto.resolve_or_unknown("sofa").path
        r = c.post(f"{base}/entities/{target['entity_id']}/reclassify", json={"new_class": new_class, "note": "chain verification",
                                                                                 "expected_version_id": v1}, headers=H)
        rep["correction"] = {"entity": target["entity_id"], "from": target["ontology_path"], "to": new_class, "status": r.status_code,
                             "result": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]}
        stale = c.post(f"{base}/entities/{target['entity_id']}/reclassify", json={"new_class": new_class, "expected_version_id": v1}, headers=H)
        rep["stale_correction_status"] = stale.status_code  # HEAD moved to V2: a correction made against V1 must be refused (409)
        v2 = ew.head_version(evroot)
        rep["v2"] = {"version_id": v2, "differs_from_v1": v2 != v1}
        if v2 and v2 != v1:
            w2 = WorldStore(evroot / ew.WORLDSTORE_REL).load_version(v2)
            d = diff_worlds(w1, w2)
            rep["v2"].update({"integrity_problems": store.verify_version(v2), "entities": len(w2.entities), "geometries": len(w2.geometries),
                              "entity_ids_unchanged": set(w1.entities) == set(w2.entities), "diff": d.summary(),
                              "modified_entities": list(d.modified_entity_ids)})
        rd2 = c.get(f"{base}/readiness", headers=H).json()
        rep["quality_after_correction"] = (rd2.get("quality") or {}).get("verdict")
        ex = c.post(f"{base}/export", json={"format": "glb", "acknowledge_unscaled": True}, headers=H)
        rep["export"] = {"status": ex.status_code, "bytes": len(ex.content), "magic": ex.content[:4].decode("latin1") if ex.status_code == 200 else None,
                         "headers": {k: v for k, v in ex.headers.items() if k.lower().startswith("x-")},
                         "detail": None if ex.status_code == 200 else ex.text[:300]}

    out = run_dir / "chain_report.json"
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("manifest_verification", "v1", "correction", "stale_correction_status", "v2", "quality_after_correction", "export")},
                     indent=1, default=str)[:6000])
    ok = (rep["manifest_verification"]["ok"] and not rep["v1"]["integrity_problems"] and rep["correction"]["status"] == 200
          and rep["stale_correction_status"] == 409 and rep["v2"].get("differs_from_v1") and rep["export"]["status"] == 200)
    print("CHAIN", "OK" if ok else "INCOMPLETE -- see chain_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
