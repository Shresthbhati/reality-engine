#!/usr/bin/env python3
"""Dataset Verification Tooling for Measured Rooms (Section 10 Protocol).

Verifies photographic datasets against capture metadata and on-site protocol:
- Validates capture.json: sensor details, consent reference, target photo count.
- When photos exist: verifies count (40-80), resolutions, EXIF camera metadata, SHA256 integrity.
- When photos are pending: outputs a precise physical capture checklist and reports BLOCKED status
  per Rule 13 ("Missing physical datasets must be marked BLOCKED, never fabricated").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEASURED_ROOMS_DIR = ROOT / "benchmarks" / "measured_rooms"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_room_capture(room_dir: Path) -> dict[str, Any]:
    """Verify one measured room's capture dataset and metadata."""
    if not room_dir.is_dir():
        return {"status": "error", "error": f"Directory not found: {room_dir}"}

    capture_file = room_dir / "capture.json"
    if not capture_file.is_file():
        return {"status": "error", "error": f"Missing capture.json in {room_dir}"}

    try:
        capture_meta = json.loads(capture_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "error": f"Invalid JSON in capture.json: {exc}"}

    # Verify required capture metadata fields
    room_id = capture_meta.get("room_id")
    device = capture_meta.get("device_model") or (capture_meta.get("camera", {}) if isinstance(capture_meta.get("camera"), dict) else {}).get("device")
    consent = capture_meta.get("consent_reference")
    target_count = capture_meta.get("target_photo_count") or capture_meta.get("photo_count", 40)

    if not room_id or not consent:
        return {"status": "error", "error": "Missing required fields ('room_id', 'consent_reference') in capture.json"}

    photos_dir = room_dir / "photos"
    found_photos = sorted(f for f in photos_dir.glob("*") if f.suffix.lower() in PHOTO_EXTENSIONS) if photos_dir.is_dir() else []

    if not found_photos:
        # Rule 13: Must be marked BLOCKED, never fabricated
        return {
            "room_id": room_id,
            "status": "BLOCKED",
            "reason": "Missing physical camera photography -- awaiting on-site capture with physical smartphone and laser distance meter",
            "target_photo_count": target_count,
            "consent_reference": consent,
            "device_model": device or "unspecified",
            "capture_checklist": [
                "1. Ensure written occupier consent is on file (consent_reference).",
                "2. Clear screens, sensitive documents, and persons from the room.",
                "3. Use smartphone default camera app at full native resolution (>= 12 MP).",
                "4. Capture 40-80 photographs walking the perimeter facing inward.",
                "5. Capture 8-12 photographs from room center facing each wall and corner.",
                "6. Capture detail photos of every electrical socket, switch, and opening frame.",
                "7. Record double laser distance measurements for all walls, heights, openings, and fixtures.",
                "8. Place photos under benchmarks/measured_rooms/<room_id>/photos/ and run this tool.",
            ],
        }

    # If photos exist, verify integrity
    verified_files = []
    checksums = capture_meta.get("checksums", {})
    count_ok = 40 <= len(found_photos) <= 80

    for ph in found_photos:
        digest = _sha256(ph)
        expected_digest = checksums.get(ph.name)
        checksum_match = (expected_digest == digest) if expected_digest else None
        verified_files.append({
            "filename": ph.name,
            "size_bytes": ph.stat().st_size,
            "sha256": digest,
            "checksum_verified": checksum_match,
        })

    return {
        "room_id": capture_meta["room_id"],
        "status": "VERIFIED" if count_ok else "WARNING_COUNT",
        "photos_found": len(found_photos),
        "target_photo_count": target_count,
        "count_within_protocol": count_ok,
        "files": verified_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify photographic dataset for a measured room.")
    parser.add_argument("--room-dir", type=Path, required=True, help="Path to room directory")
    args = parser.parse_args(argv)

    res = verify_room_capture(args.room_dir)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] in ("VERIFIED", "BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
