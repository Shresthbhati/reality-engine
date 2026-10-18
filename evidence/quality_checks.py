"""Evidence Quality Firewall: stored image facts -> check results -> score.

Pure and deterministic. Inputs are the facts ``evidence.image_check.inspect_image``
ALREADY measured and stored on the Evidence row (``metadata_json["image_facts"]``);
nothing is re-measured here and nothing is random. A check that cannot be
evaluated is omitted rather than invented (exposure, for example, is not
measured by ``image_check`` and so is not a check).

Thresholds are POLICY values, not calibrated physics. They are named constants so
a reviewer can change them in one place:

  RESOLUTION_FAIL_MP / RESOLUTION_WARN_MP   megapixels (width*height/1e6)
  SHARPNESS_FAIL / SHARPNESS_WARN           variance of the Laplacian on a 256px grayscale
                                            copy. A RELATIVE ranking between images, not a
                                            calibrated blur score (see ImageFacts.sharpness).

Check statuses: ``pass`` | ``warn`` | ``fail``.

Score (0-100) = max(0, 100 - WARN_PENALTY * warns - FAIL_PENALTY * fails);
an undecodable image scores 0. It summarises the checks above and is NOT a
probability or a calibrated quality measure.

Automatic result (what the checks alone say; the operator may override it):
  rejected               undecodable, or an exact duplicate (same sha256) of earlier evidence
  needs_review           any other failed check
  accepted_with_warning  only warnings
  accepted               every check passed
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RESOLUTION_FAIL_MP = 0.25
RESOLUTION_WARN_MP = 1.0
SHARPNESS_FAIL = 10.0
SHARPNESS_WARN = 50.0
WARN_PENALTY = 15
FAIL_PENALTY = 40

DECISIONS = ("accepted", "accepted_with_warning", "needs_review", "rejected")

_GEOMETRY_CLASSES = ("photograph", "photograph_unverified")


def _check(name: str, status: str, value: Optional[str], threshold: Optional[str], message: str) -> Dict[str, Any]:
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": message}


def evaluate_checks(facts: Optional[dict], duplicate_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check results for one image. ``facts`` is the stored ``image_facts`` dict
    (None/empty = the image was never inspected -> no checks, not a pass).
    ``duplicate_of`` is the name of an EARLIER evidence item with the same sha256."""
    if not facts:
        return []
    if not facts.get("ok"):
        return [_check("Readable", "fail", None, "decodable image", facts.get("reason") or "not a decodable image")]

    checks: List[Dict[str, Any]] = [_check("Readable", "pass", None, None, "File decodes as an image")]

    w, h = int(facts.get("width") or 0), int(facts.get("height") or 0)
    mp = w * h / 1e6
    val = f"{w}x{h} ({mp:.2f} MP)"
    if mp < RESOLUTION_FAIL_MP:
        checks.append(_check("Resolution", "fail", val, f">= {RESOLUTION_FAIL_MP} MP",
                             f"Very low resolution ({mp:.2f} MP < {RESOLUTION_FAIL_MP} MP)"))
    elif mp < RESOLUTION_WARN_MP:
        checks.append(_check("Resolution", "warn", val, f">= {RESOLUTION_WARN_MP} MP",
                             f"Low resolution ({mp:.2f} MP < {RESOLUTION_WARN_MP} MP)"))
    else:
        checks.append(_check("Resolution", "pass", val, f">= {RESOLUTION_WARN_MP} MP", "Resolution is sufficient"))

    sharp = facts.get("sharpness")
    if sharp is None:
        pass  # not measured: omit rather than guess
    elif sharp < SHARPNESS_FAIL:
        checks.append(_check("Sharpness", "fail", f"{sharp:.1f}", f">= {SHARPNESS_WARN:.0f}",
                             f"Very blurry or featureless (Laplacian variance {sharp:.1f})"))
    elif sharp < SHARPNESS_WARN:
        checks.append(_check("Sharpness", "warn", f"{sharp:.1f}", f">= {SHARPNESS_WARN:.0f}",
                             f"Soft image (Laplacian variance {sharp:.1f})"))
    else:
        checks.append(_check("Sharpness", "pass", f"{sharp:.1f}", f">= {SHARPNESS_WARN:.0f}", "Sharpness is sufficient"))

    exif = facts.get("exif") or {}
    if exif.get("make") or exif.get("model"):
        cam = " ".join(x for x in (exif.get("make"), exif.get("model")) if x)
        checks.append(_check("Camera metadata", "pass", cam, "make/model EXIF", "Camera EXIF present"))
    else:
        checks.append(_check("Camera metadata", "warn", None, "make/model EXIF",
                             "No camera EXIF; cannot verify this is a photograph"))

    cls = facts.get("evidence_class") or "unknown"
    if cls in _GEOMETRY_CLASSES:
        checks.append(_check("Evidence class", "pass", cls, "photograph", "Eligible to feed geometry"))
    else:
        checks.append(_check("Evidence class", "fail", cls, "photograph",
                             f"Classified {cls}; not used for observed geometry"))

    if duplicate_of:
        checks.append(_check("Duplicate", "fail", duplicate_of, "unique sha256",
                             f"Byte-identical to earlier evidence '{duplicate_of}'"))
    else:
        checks.append(_check("Duplicate", "pass", None, "unique sha256", "No byte-identical earlier evidence"))
    return checks


def quality_score(checks: List[Dict[str, Any]]) -> Optional[int]:
    """None when there are no checks (never inspected); see module docstring for the formula."""
    if not checks:
        return None
    if any(c["name"] == "Readable" and c["status"] == "fail" for c in checks):
        return 0
    warns = sum(1 for c in checks if c["status"] == "warn")
    fails = sum(1 for c in checks if c["status"] == "fail")
    return max(0, 100 - WARN_PENALTY * warns - FAIL_PENALTY * fails)


def auto_result(checks: List[Dict[str, Any]]) -> Optional[str]:
    """What the checks alone conclude; None when nothing was inspected."""
    if not checks:
        return None
    failed = {c["name"] for c in checks if c["status"] == "fail"}
    if failed & {"Readable", "Duplicate"}:
        return "rejected"
    if failed:
        return "needs_review"
    if any(c["status"] == "warn" for c in checks):
        return "accepted_with_warning"
    return "accepted"


def key_issues(checks: List[Dict[str, Any]]) -> List[str]:
    return [c["message"] for c in checks if c["status"] != "pass"]


def explain(checks: List[Dict[str, Any]]) -> str:
    """Deterministic sentence built only from the check results (no model)."""
    if not checks:
        return "This file has not been inspected, so there are no check results to summarise."
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    parts = [f"{n_pass} of {len(checks)} checks passed."]
    for label, status in (("Failed", "fail"), ("Warnings", "warn")):
        items = [f"{c['name']} ({c['message']})" for c in checks if c["status"] == status]
        if items:
            parts.append(f"{label}: " + "; ".join(items) + ".")
    return " ".join(parts)
