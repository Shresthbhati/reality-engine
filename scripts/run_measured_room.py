#!/usr/bin/env python3
"""Photos in, scored report out, through the real API (E209).

Replays the golden-path call sequence (tests/test_golden_path_api_to_glb.py) against a running
API with one measured room's photos, waits for the job, fetches the stored world and scores it
against the room's ground truth. It measures the real engine only: it refuses to run while the
test-backend seam is active, and it never scores a failed or cancelled job.

    python scripts/run_measured_room.py --room-dir benchmarks/measured_rooms/room-1 \
        --api http://127.0.0.1:8000 --token <t> --out benchmarks/results/measured

Exit codes: 0 scored, 2 bad input (missing or invalid ground truth, no photos), 3 test backend
active, 4 the job failed/was cancelled/timed out, 5 the API could not be reached or answered
with an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA_PATH = ROOT / "benchmarks" / "measured_rooms" / "ground_truth.schema.json"
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
TERMINAL = {"succeeded", "partial", "failed", "cancelled"}
BACKEND_ENV_MESSAGE = "REALITY_TEST_BACKEND is set; this script measures the real engine only"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _schema_errors(ground_truth: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator  # dev dependency

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return [e.message for e in Draft202012Validator(schema).iter_errors(ground_truth)]


def _stages(client: Any, job_id: str) -> list[dict[str, Any]] | None:
    """The job's stage rows (GET /api/jobs/{id}/stages, P105); ``None`` if the API has no such route."""
    r = client.get(f"/api/jobs/{job_id}/stages")
    return r.json().get("items") if r.status_code == 200 else None


def _wait_for_job(client: Any, job_id: str, timeout_s: float, poll_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        r = client.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        if job.get("status") in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            return {**job, "status": "failed", "error": f"timed out after {timeout_s:.0f}s waiting for the job"}
        time.sleep(poll_s)


def main(argv: list[str] | None = None, transport: Any = None) -> int:
    p = argparse.ArgumentParser(description="Run one measured room through the real API and score it.")
    p.add_argument("--room-dir", type=Path, required=True)
    p.add_argument("--api", required=True, help="API base URL")
    p.add_argument("--token", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--poll-s", type=float, default=2.0)
    args = p.parse_args(argv)

    if os.environ.get("REALITY_TEST_BACKEND"):
        print(BACKEND_ENV_MESSAGE, file=sys.stderr)
        return 3

    gt_path = args.room_dir / "ground_truth.json"
    if not gt_path.is_file():
        print(f"missing {gt_path}", file=sys.stderr)
        return 2
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    errors = _schema_errors(ground_truth)
    if errors:
        print(f"{gt_path} does not match ground_truth.schema.json: {errors[0]}", file=sys.stderr)
        return 2
    photos = sorted(f for f in (args.room_dir / "photos").glob("*") if f.suffix.lower() in PHOTO_SUFFIXES)
    if not photos:
        print(f"no photos under {args.room_dir / 'photos'}", file=sys.stderr)
        return 2

    import httpx

    from perception.architecture.accuracy_report import evaluate_world_accuracy
    from world_ir import WorldIR

    room_id = args.room_dir.name
    sha = _git_short_sha()
    out_dir = args.out / room_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        with httpx.Client(
            base_url=args.api, headers={"Authorization": f"Bearer {args.token}"}, timeout=120.0, transport=transport
        ) as client:
            def call(method: str, url: str, **kw: Any) -> Any:
                r = client.request(method, url, **kw)
                r.raise_for_status()
                return r.json()

            sid = call("POST", "/api/sessions", json={"name": f"measured-{room_id}"})["id"]
            wid = call("POST", "/api/worlds", json={"name": f"measured-{room_id}"})["id"]
            call("POST", f"/api/worlds/{wid}/attach/{sid}")
            for photo in photos:
                mime = mimetypes.guess_type(photo.name)[0] or "application/octet-stream"
                call("POST", f"/api/uploads?session_id={sid}", files={"file": (photo.name, photo.read_bytes(), mime)})
            job_id = call("POST", f"/api/sessions/{sid}/reconstruct")["job_id"]
            job = _wait_for_job(client, job_id, args.timeout_s, args.poll_s)
            stages = _stages(client, job_id)

            if job["status"] in ("failed", "cancelled"):
                failure = out_dir / f"{sha}.failed.json"
                failure.write_text(
                    json.dumps({"git_sha": sha, "job_id": job_id, "job_status": job["status"],
                                "error": job.get("error"), "stages": stages, "job": job}, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"job {job_id} {job['status']}: {job.get('error')} (not scored; see {failure})", file=sys.stderr)
                return 4

            world_json = call("GET", f"/api/worlds/{wid}/worldir")
    except httpx.HTTPError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 5

    # GET /api/worlds/{id}/worldir returns WorldIR.to_dict(): keep it next to the score, then evaluate it.
    (out_dir / f"{sha}.worldir.json").write_text(json.dumps(world_json, indent=2) + "\n", encoding="utf-8")
    report = evaluate_world_accuracy(WorldIR.from_dict(world_json), ground_truth)

    result = {
        "git_sha": sha,
        "room_id": room_id,
        "evidence_class": "measured",
        "job_id": job_id,
        "job_status": job["status"],
        "stages": stages,
        "evaluation": report.to_dict(),
        "wall_clock_s": round(time.monotonic() - started, 1),
        "photos_sha256": {f.name: _sha256(f) for f in photos},
    }
    out_file = out_dir / f"{sha}.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{room_id}: job {job['status']}, Section 1 gate {'PASS' if report.meets_tolerances else 'FAIL'} -> {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
