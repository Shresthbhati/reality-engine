"""Shared request validation for the application API.

Every creator/ingest endpoint funnels user-controlled scalars through
here so invalid input is rejected explicitly (422/413) instead of being
persisted and corrupting downstream state (NaN coordinates break JSON
consumers; unbounded blobs bloat the database).
"""

from __future__ import annotations

import json
import math

from fastapi import HTTPException

MAX_NAME_LENGTH = 200
MAX_JSON_BLOB_BYTES = 64 * 1024


def require_name(value: object, what: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(422, f"{what} must be a non-empty string")
    if len(value) > MAX_NAME_LENGTH:
        raise HTTPException(422, f"{what} must be at most {MAX_NAME_LENGTH} chars")
    return value


def require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(422, f"{label} must be a number")
    if not math.isfinite(value):
        raise HTTPException(422, f"{label} must be finite (no NaN/infinity)")
    return float(value)


def require_latitude(value: object) -> float:
    v = require_finite(value, "latitude")
    if not -90.0 <= v <= 90.0:
        raise HTTPException(422, "latitude must be within [-90, 90]")
    return v


def require_longitude(value: object) -> float:
    v = require_finite(value, "longitude")
    if not -180.0 <= v <= 180.0:
        raise HTTPException(422, "longitude must be within [-180, 180]")
    return v


def require_json_size(value: object, label: str, limit: int = MAX_JSON_BLOB_BYTES):
    if value is None:
        return value
    try:
        size = len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        raise HTTPException(422, f"{label} is not JSON-serializable")
    if size > limit:
        raise HTTPException(413, f"{label} exceeds {limit} bytes")
    return value
