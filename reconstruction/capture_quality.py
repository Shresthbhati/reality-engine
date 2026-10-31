"""Capture guidance (plan F1): tell the user about bad photos BEFORE they cost a reconstruction, pick better
video keyframes, and report which walls were / were not seen AFTER one.

Honesty rules
  * Every threshold below is PROVISIONAL: chosen from common computer-vision rules of thumb and checked only on
    synthetic images. None has been validated on a real photo set. Verdicts are predictions, not measurements.
  * Overlap / baseline is a cheap ORB-feature PROXY (not the SfM matcher). It says "these two photos look related
    and shifted", nothing more.
  * A sharp photo of a plain painted wall has a low Laplacian variance too (no texture to respond to); the blur
    reason says so rather than claiming the photo is out of focus.
  * Wall coverage is a 2-D plan-view visibility estimate from camera centre + viewing direction. It ignores
    furniture occlusion and lens FOV is an assumed constant (the true intrinsics are not passed here).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

try:  # OpenCV is an optional runtime dependency of the repo
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# ---------------------------------------------------------------- thresholds (all PROVISIONAL) -----------------
ANALYSIS_MAX_SIDE_PX = 640
#: Laplacian variance at <=640 px. 100 is the widely quoted "blurry below this" rule of thumb (not calibrated).
BLUR_WARN_BELOW = 100.0
#: below this almost nothing registers; set from synthetic Gaussian-blur sweeps only.
BLUR_DROP_BELOW = 25.0
#: fraction of pixels at >=250 (blown) or <=5 (crushed)
CLIP_WARN_FRACTION = 0.20
CLIP_DROP_FRACTION = 0.50
MEAN_LUMA_WARN = (45.0, 210.0)
MEAN_LUMA_DROP = (18.0, 238.0)
#: ORB proxy
ORB_FEATURES = 800
ORB_RATIO = 0.75
MIN_GOOD_MATCHES_WARN = 25
MIN_GOOD_MATCHES_DROP = 8
#: median keypoint shift, as a fraction of the image diagonal. Below this the two views have ~no baseline.
MIN_MEDIAN_SHIFT_FRACTION = 0.01
#: all-pairs when the batch is small, else a +-window in name order (assumes photos were taken in sequence)
ALL_PAIRS_UP_TO = 12
PAIR_WINDOW = 4
MAX_PHOTOS_FOR_PAIRS = 60

# video keyframes
#: mean absolute grey difference (0-255 scale, on a 32x24 thumbnail) a keyframe must differ from the last kept one
KEYFRAME_MIN_MOTION = 4.0
SCORES_PER_WINDOW = 8

# wall coverage
ASSUMED_HALF_FOV_DEG = 35.0
MAX_INCIDENCE_DEG = 75.0
WALL_SAMPLES = 12
SEEN_COVERED_FRACTION = 0.80
SEEN_MULTI_FRACTION = 0.50
NOT_SEEN_COVERED_FRACTION = 0.20
HINT_PHOTOS = {"not_seen": 3, "weak": 2}

OK, WARN, DROP = "ok", "warn", "will_be_dropped"
_RANK = {OK: 0, WARN: 1, DROP: 2}
PROVISIONAL_NOTE = ("Thresholds are provisional (rules of thumb tested on synthetic images only, not validated on "
                    "real photos). Verdicts are predictions about registration, not measurements.")


def _decode_gray(data: bytes):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    h, w = arr.shape
    s = ANALYSIS_MAX_SIDE_PX / max(h, w)
    if s < 1.0:
        arr = cv2.resize(arr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return arr


def _worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def photo_metrics(gray) -> dict:
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "blur_laplacian_var": round(lap, 2),
        "mean_luma": round(float(gray.mean()), 1),
        "clipped_high": round(float((gray >= 250).mean()), 4),
        "clipped_low": round(float((gray <= 5).mean()), 4),
    }


def judge_photo(m: dict) -> tuple[str, list[str]]:
    """(verdict, reasons) from one photo's metrics."""
    verdict, reasons = OK, []
    v = m["blur_laplacian_var"]
    if v < BLUR_DROP_BELOW:
        verdict = _worse(verdict, DROP)
        reasons.append(f"Very low detail (sharpness {v:.0f}): too blurry, or a plain surface with no texture, "
                       "so it will probably not match other photos.")
    elif v < BLUR_WARN_BELOW:
        verdict = _worse(verdict, WARN)
        reasons.append(f"Low detail (sharpness {v:.0f}): slightly blurry, or a plain surface. Hold still and "
                       "include some texture or edges.")
    for side, frac, word in (("high", m["clipped_high"], "overexposed (washed out)"),
                             ("low", m["clipped_low"], "underexposed (crushed black)")):
        if frac >= CLIP_DROP_FRACTION:
            verdict = _worse(verdict, DROP)
            reasons.append(f"{frac:.0%} of the photo is {word}: detail is lost.")
        elif frac >= CLIP_WARN_FRACTION:
            verdict = _worse(verdict, WARN)
            reasons.append(f"{frac:.0%} of the photo is {word}.")
    luma = m["mean_luma"]
    if luma < MEAN_LUMA_DROP[0] or luma > MEAN_LUMA_DROP[1]:
        verdict = _worse(verdict, DROP)
        reasons.append("Overall brightness is extreme; the room is not readable in this photo.")
    elif luma < MEAN_LUMA_WARN[0]:
        verdict = _worse(verdict, WARN)
        reasons.append("The photo is dark; turn on lights or open curtains.")
    elif luma > MEAN_LUMA_WARN[1]:
        verdict = _worse(verdict, WARN)
        reasons.append("The photo is very bright; avoid pointing at windows or lamps.")
    return verdict, reasons


def pair_overlap_proxy(g1, g2) -> dict:
    """ORB ratio-test matches + median keypoint shift between two grey images (a PROXY for overlap/baseline)."""
    orb = cv2.ORB_create(ORB_FEATURES)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return {"good_matches": 0, "median_shift_fraction": None}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < ORB_RATIO * m[1].distance]
    if not good:
        return {"good_matches": 0, "median_shift_fraction": None}
    # resize both to a common frame before measuring shift so differing sizes cannot fake a baseline
    sx2, sy2 = g1.shape[1] / g2.shape[1], g1.shape[0] / g2.shape[0]
    d = [math.hypot(k1[m.queryIdx].pt[0] - k2[m.trainIdx].pt[0] * sx2,
                    k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1] * sy2) for m in good]
    diag = math.hypot(*g1.shape[::-1])
    return {"good_matches": len(good), "median_shift_fraction": round(float(np.median(d)) / diag, 4)}


def assess_capture(photos: Sequence[tuple[str, bytes]]) -> dict:
    """Per-photo verdicts + a batch summary.

    ``photos`` is [(name, image_bytes)] in capture order. Returns
    ``{"photos": {name: {verdict, reasons, metrics, overlap}}, "summary": {...}, "provisional": True, ...}``.
    Never raises on an undecodable photo: it is reported as will_be_dropped."""
    if cv2 is None:
        return {"available": False, "reason": "OpenCV is not installed", "photos": {}, "summary": None}
    grays: list = []
    out: dict[str, dict] = {}
    for name, data in photos:
        g = _decode_gray(data)
        grays.append(g)
        if g is None:
            out[name] = {"verdict": DROP, "reasons": ["The file could not be decoded as an image."],
                         "metrics": None, "overlap": None}
            continue
        m = photo_metrics(g)
        verdict, reasons = judge_photo(m)
        out[name] = {"verdict": verdict, "reasons": reasons, "metrics": m, "overlap": None}

    names = [n for n, _ in photos]
    n = len(photos)
    if 2 <= n <= MAX_PHOTOS_FOR_PAIRS:
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)
                 if n <= ALL_PAIRS_UP_TO or j - i <= PAIR_WINDOW}
        best: dict[int, dict] = {}
        for i, j in sorted(pairs):
            if grays[i] is None or grays[j] is None:
                continue
            r = pair_overlap_proxy(grays[i], grays[j])
            for a, b in ((i, j), (j, i)):
                cur = best.get(a)
                if cur is None or r["good_matches"] > cur["good_matches"]:
                    best[a] = {**r, "best_partner": names[b]}
        for i, nm in enumerate(names):
            if grays[i] is None:
                continue
            ov = best.get(i) or {"good_matches": 0, "median_shift_fraction": None, "best_partner": None}
            ov["method"] = "ORB proxy (not the SfM matcher)"
            out[nm]["overlap"] = ov
            gm = ov["good_matches"]
            if gm < MIN_GOOD_MATCHES_DROP:
                out[nm]["verdict"] = _worse(out[nm]["verdict"], DROP)
                out[nm]["reasons"].append("Shares almost nothing with the other photos, so it probably cannot be "
                                          "placed. Take photos that overlap about half with a neighbour.")
            elif gm < MIN_GOOD_MATCHES_WARN:
                out[nm]["verdict"] = _worse(out[nm]["verdict"], WARN)
                out[nm]["reasons"].append("Little overlap with the other photos. Overlap about half of each view "
                                          "with the previous one.")
            elif (ov["median_shift_fraction"] or 0) < MIN_MEDIAN_SHIFT_FRACTION:
                out[nm]["verdict"] = _worse(out[nm]["verdict"], WARN)
                out[nm]["reasons"].append("Almost identical to another photo (no baseline). Take the next photo "
                                          "after stepping sideways about half a metre.")

    counts = {OK: 0, WARN: 0, DROP: 0}
    for r in out.values():
        counts[r["verdict"]] += 1
    summary = {"total": len(out), **{k: v for k, v in counts.items()}, "pair_check": 2 <= n <= MAX_PHOTOS_FOR_PAIRS}
    summary["headline"] = _headline(summary)
    return {"available": True, "photos": out, "summary": summary, "provisional": True,
            "provisional_note": PROVISIONAL_NOTE}


def _headline(s: dict) -> str:
    if s["total"] == 0:
        return "No photos checked."
    if s[WARN] == 0 and s[DROP] == 0:
        return f"All {s['total']} photos look usable (provisional check)."
    bits = []
    if s[DROP]:
        bits.append(f"{s[DROP]} will probably be dropped")
    if s[WARN]:
        bits.append(f"{s[WARN]} could be better")
    return f"{', '.join(bits)} out of {s['total']} photos."


def summarize_stored(per_photo: Iterable[dict | None]) -> dict:
    """Aggregate already-stored per-photo ``capture_quality`` dicts (for the status endpoint)."""
    counts = {OK: 0, WARN: 0, DROP: 0}
    n = 0
    for q in per_photo:
        if isinstance(q, dict) and q.get("verdict") in counts:
            counts[q["verdict"]] += 1
            n += 1
    s = {"total": n, **counts}
    s["headline"] = _headline(s)
    return s


# ------------------------------------------------------------------- video keyframes ----------------------------
def score_video_frames(path: str, frame_count: int, budget: int) -> list[tuple[int, float, np.ndarray]]:
    """[(frame_index, sharpness, 32x24 thumbnail)] for a strided subset of frames (grab() skips the rest)."""
    stride = max(1, frame_count // max(1, budget * SCORES_PER_WINDOW))
    cap = cv2.VideoCapture(path)
    scores = []
    try:
        for i in range(frame_count):
            if i % stride:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            s = ANALYSIS_MAX_SIDE_PX / max(h, w)
            if s < 1.0:
                g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            scores.append((i, float(cv2.Laplacian(g, cv2.CV_64F).var()),
                           cv2.resize(g, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)))
    finally:
        cap.release()
    return scores


def pick_keyframes(scores: Sequence[tuple[int, float, np.ndarray]], budget: int,
                   min_motion: float = KEYFRAME_MIN_MOTION) -> list[int]:
    """Sharpest frame per time window, skipping any that barely moved since the last kept frame.

    Windows are equal-count slices of the scored frames, so the result stays spread across the whole clip.
    Within a window frames are tried sharpest-first; the first that differs from the previous keyframe by at least
    ``min_motion`` (mean abs grey difference) wins, and a window with no such frame is skipped (static footage).
    Returns frame indices in time order. Deterministic."""
    n = len(scores)
    if n == 0:
        return []
    budget = min(budget, n)
    kept: list[int] = []
    last = None
    for w in range(budget):
        lo, hi = w * n // budget, (w + 1) * n // budget
        window = sorted(scores[lo:hi], key=lambda s: (-s[1], s[0]))
        for idx, _sharp, thumb in window:
            if last is None or float(np.abs(thumb - last).mean()) >= min_motion:
                kept.append(idx)
                last = thumb
                break
    return sorted(kept)


# ------------------------------------------------------------------ post-reconstruction wall coverage -----------
def _quat_forward(rot_wxyz: Sequence[float]) -> np.ndarray:
    """Camera looks along its local +Z (COLMAP/OpenCV; same convention as engine/pipeline/guidance.py)."""
    w, x, y, z = rot_wxyz
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _seg_hit(p, q, a, b) -> bool:
    """Proper intersection of segments pq and ab (2-D)."""
    def cr(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3, d4 = cr(a, b, p), cr(a, b, q), cr(p, q, a), cr(p, q, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def _corner_label(i: int) -> str:
    return chr(ord("A") + i % 26) * (1 + i // 26)


def wall_coverage(polygon_xyz: Sequence[Sequence[float]], cameras: Sequence[dict],
                  basis: dict | None = None) -> dict:
    """Per-wall coverage of a room footprint by the registered cameras.

    polygon_xyz  WorldIR ``footprint_polygon_xyz_m`` (room footprint corners, world frame)
    cameras      [{evidence_id, position_m, rotation_wxyz}] (the /cameras shape)
    basis        WorldIR ``footprint_basis`` ({e1, e2, up}); without it the plan is (x, z) with +Y up.

    Walls are labelled by index and corner letters, never by compass direction: no north reference exists."""
    P = np.asarray(polygon_xyz, float)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 3:
        return {"available": False, "reason": "no room footprint polygon", "walls": [], "hints": []}
    if basis and all(k in basis for k in ("e1", "e2")):
        E = np.array([basis["e1"], basis["e2"]], float)
    else:
        E = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poly = P @ E.T
    # make the polygon counter-clockwise so the inward normal is the left-hand normal
    area2 = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    if area2 < 0:
        poly = poly[::-1]
    n = len(poly)
    cams = []
    for c in cameras:
        try:
            pos = np.asarray(c["position_m"], float)
            fwd = _quat_forward(c.get("rotation_wxyz", [1, 0, 0, 0]))
        except (KeyError, TypeError, ValueError):
            continue
        f2 = E @ fwd
        if np.hypot(*f2) < 1e-6:  # looking straight up/down: sees no wall in plan view
            continue
        cams.append({"id": c.get("evidence_id"), "xy": E @ pos, "dir": f2 / np.hypot(*f2)})
    cos_fov = math.cos(math.radians(ASSUMED_HALF_FOV_DEG))
    cos_inc = math.cos(math.radians(MAX_INCIDENCE_DEG))
    walls = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        length = float(np.hypot(*e))
        normal = np.array([-e[1], e[0]]) / max(length, 1e-12)  # inward for a CCW polygon
        cover = np.zeros(WALL_SAMPLES, int)
        for s in range(WALL_SAMPLES):
            pt = a + e * ((s + 0.5) / WALL_SAMPLES)
            for c in cams:
                ray = pt - c["xy"]
                dist = float(np.hypot(*ray))
                if dist < 1e-9:
                    continue
                ray /= dist
                if np.dot(ray, c["dir"]) < cos_fov:      # outside the horizontal field of view
                    continue
                if -np.dot(ray, normal) < cos_inc:       # behind the wall, or too oblique to resolve
                    continue
                inset = pt + normal * 1e-6               # nudge off the wall so its own edge does not block it
                if any(_seg_hit(c["xy"], inset, poly[j], poly[(j + 1) % n]) for j in range(n) if j != i):
                    continue                             # another wall is in the way
                cover[s] += 1
        covered = float((cover >= 1).mean())
        multi = float((cover >= 2).mean())
        if covered >= SEEN_COVERED_FRACTION and multi >= SEEN_MULTI_FRACTION:
            status = "seen"
        elif covered < NOT_SEEN_COVERED_FRACTION:
            status = "not_seen"
        else:
            status = "weak"
        la, lb = _corner_label(i), _corner_label((i + 1) % n)
        label = f"wall {i + 1} (between corner {la} and {lb})"
        hint = None
        if status != "seen":
            k = HINT_PHOTOS[status]
            hint = (f"Take {k} more photos of {label}, standing on the opposite side of the room and "
                    f"{'overlapping each view with the last' if status == 'weak' else 'facing it squarely'}.")
        walls.append({"index": i + 1, "label": label, "between": [la, lb], "p0": poly[i].round(4).tolist(),
                      "p1": poly[(i + 1) % n].round(4).tolist(), "status": status,
                      "covered_fraction": round(covered, 3), "multi_view_fraction": round(multi, 3),
                      "length": round(length, 4), "hint": hint})
    return {
        "available": True, "reason": None, "provisional": True,
        "walls": walls,
        "corners": [{"label": _corner_label(i), "xy": poly[i].round(4).tolist()} for i in range(n)],
        "cameras": [{"id": c["id"], "xy": c["xy"].round(4).tolist(), "dir": c["dir"].round(4).tolist()}
                    for c in cams],
        "hints": [w["hint"] for w in walls if w["hint"]],
        "summary": {s: sum(w["status"] == s for w in walls) for s in ("seen", "weak", "not_seen")},
        "notes": [
            f"Plan-view estimate: camera field of view assumed {2 * ASSUMED_HALF_FOV_DEG:.0f} degrees, furniture "
            "occlusion ignored; thresholds are provisional.",
            "Walls are numbered around the footprint, not by compass: no north reference is available.",
        ],
    }
