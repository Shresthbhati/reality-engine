"""Evidence contribution: what does each image add, measured from pixels.

Image COUNT is not the product contract; spatial information is. This
module estimates, per image, whether it is

  * ``redundant``     near-identical viewpoint to an earlier image (many verified feature
                      matches with almost no image-plane motion)
  * ``new_view``      overlaps other evidence but from a different viewpoint (verified
                      matches with real parallax) -- what multi-view geometry feeds on
  * ``disconnected``  shares no verifiable content with any other image: either a new
                      area (useful only if more views join it) or unrelated
  * ``first``         the very first image (nothing earlier to compare against)

Method: SIFT features + ratio-test matching + RANSAC fundamental-matrix
verification (OpenCV). All numbers are counts of verified matches; the
labels are thresholded from them. Nothing here is a calibrated quality
score, and results carry the raw counts so a consumer can judge for
itself. If OpenCV is unavailable the report says so and labels every
image ``unknown`` -- it never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MIN_VERIFIED_MATCHES = 15          # below this a pair is "not overlapping" (measured: true wide-baseline
                                   # neighbours give 17+, unrelated pairs <=14 on real photos)
REDUNDANT_MEDIAN_SHIFT = 0.03      # of image diagonal; below = same viewpoint
MAX_IMAGES_ALL_PAIRS = 40          # beyond this only a sliding window of pairs is compared
WINDOW = 10                        # ponytail: windowed pairs assume arrival order ~ capture order
_FEATURES = 5000
_MAX_SIDE = 1024


@dataclass(frozen=True)
class ImageContribution:
    evidence_id: str
    label: str                       # first|new_view|redundant|disconnected|unknown
    best_partner: Optional[str] = None
    verified_matches: int = 0
    median_shift_fraction: Optional[float] = None
    is_new: bool = True              # True = arrived in this batch (vs. prior evidence)
    #: why the label is what it is -- NOVEL_VIEW | REDUNDANT | INSUFFICIENT_MATCHES |
    #: DEGENERATE_GEOMETRY | NOT_MEASURABLE | FIRST | None
    status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "label": self.label,
            "status": self.status,
            "best_partner": self.best_partner,
            "verified_matches": self.verified_matches,
            "median_shift_fraction": self.median_shift_fraction,
            "is_new": self.is_new,
        }


@dataclass(frozen=True)
class ContributionReport:
    available: bool
    note: str
    per_image: List[ImageContribution] = field(default_factory=list)
    components: Optional[int] = None            # connected groups in the overlap graph
    largest_component: Optional[int] = None
    pair_matches: Dict[str, int] = field(default_factory=dict)  # "a|b" -> verified matches

    def summary(self) -> dict:
        new = [c for c in self.per_image if c.is_new]
        return {
            "new_images": len(new),
            "new_view": sum(1 for c in new if c.label == "new_view"),
            "redundant": sum(1 for c in new if c.label == "redundant"),
            "disconnected": sum(1 for c in new if c.label == "disconnected"),
        }

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "note": self.note,
            "per_image": [c.to_dict() for c in self.per_image],
            "components": self.components,
            "largest_component": self.largest_component,
            "summary": self.summary(),
        }


def _features(path: Path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = _MAX_SIDE / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    try:
        kp, des = cv2.SIFT_create(nfeatures=_FEATURES).detectAndCompute(img, None)
    except cv2.error:
        return None
    if des is None or len(kp) < 8:
        return None
    return kp, des, float((img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5)


def _verified_pair(fa, fb):
    """(verified_matches, median_shift_fraction, status) for two feature sets.

    Never raises: a degenerate pair (collinear/duplicate points, singular
    fundamental matrix, OpenCV error) is a classification, not a crash.
    status: OK | INSUFFICIENT_MATCHES | DEGENERATE_GEOMETRY
    """
    import cv2
    import numpy as np

    kpa, dea, diag = fa
    kpb, deb, _ = fb
    try:
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dea, deb, k=2)
    except cv2.error:
        return 0, None, "DEGENERATE_GEOMETRY"
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 8:
        return 0, None, "INSUFFICIENT_MATCHES"
    pa = np.float32([kpa[m.queryIdx].pt for m in good])
    pb = np.float32([kpb[m.trainIdx].pt for m in good])
    # >= 8 matches can still be one repeated point or all on a line: F is undefined there
    if np.unique(pa, axis=0).shape[0] < 8 or np.unique(pb, axis=0).shape[0] < 8:
        return 0, None, "DEGENERATE_GEOMETRY"
    try:
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    except (cv2.error, ValueError):
        return 0, None, "DEGENERATE_GEOMETRY"
    if mask is None:
        return 0, None, "DEGENERATE_GEOMETRY"
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())
    if n == 0:
        return 0, None, "DEGENERATE_GEOMETRY"
    shift = np.linalg.norm(pa[inl] - pb[inl], axis=1)
    return n, float(np.median(shift) / diag), "OK"


def analyze_contribution(
    images: Sequence[tuple],            # (evidence_id, Path) in arrival order
    prior_ids: Sequence[str] = (),      # ids that were already part of the world
) -> ContributionReport:
    prior = set(prior_ids)
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        return ContributionReport(
            available=False, note=f"OpenCV unavailable ({exc}); contribution not measured",
            per_image=[ImageContribution(i, "unknown", is_new=i not in prior, status="NOT_MEASURABLE")
                       for i, _ in images],
        )

    feats: Dict[str, object] = {}
    order: List[str] = []
    for eid, path in images:
        order.append(eid)
        feats[eid] = _features(Path(path))

    pair_matches: Dict[str, int] = {}
    shifts: Dict[str, Optional[float]] = {}
    pair_status: Dict[str, str] = {}
    n = len(order)
    for j in range(n):
        lo = 0 if n <= MAX_IMAGES_ALL_PAIRS else max(0, j - WINDOW)
        for i in range(lo, j):
            a, b = order[i], order[j]
            if feats[a] is None or feats[b] is None:
                continue
            m, sh, st = _verified_pair(feats[a], feats[b])
            pair_matches[f"{a}|{b}"] = m
            shifts[f"{a}|{b}"] = sh
            pair_status[f"{a}|{b}"] = st

    parent = {e: e for e in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, m in pair_matches.items():
        if m >= MIN_VERIFIED_MATCHES:
            a, b = key.split("|")
            parent[find(a)] = find(b)
    groups: Dict[str, int] = {}
    for e in order:
        groups[find(e)] = groups.get(find(e), 0) + 1

    per_image: List[ImageContribution] = []
    for idx, eid in enumerate(order):
        is_new = eid not in prior
        if feats[eid] is None:
            per_image.append(ImageContribution(eid, "unknown", is_new=is_new, status="NOT_MEASURABLE"))
            continue
        if idx == 0:
            per_image.append(ImageContribution(eid, "first", is_new=is_new, status="FIRST"))
            continue
        best_key, best_m = None, 0
        for i in range(idx):
            key = f"{order[i]}|{eid}"
            m = pair_matches.get(key, 0)
            if m > best_m:
                best_key, best_m = key, m
        if best_m < MIN_VERIFIED_MATCHES:
            # it may overlap a LATER image, in which case it is not cut off from the set
            later = max((pair_matches.get(f"{eid}|{order[k]}", 0) for k in range(idx + 1, n)), default=0)
            label = "new_view" if later >= MIN_VERIFIED_MATCHES else "disconnected"
            # why it did not connect: a degenerate pair is not the same as "no overlap"
            sts = {pair_status.get(f"{order[i]}|{eid}") for i in range(idx)} - {None}
            status = ("NOVEL_VIEW" if label == "new_view" else
                      "DEGENERATE_GEOMETRY" if "DEGENERATE_GEOMETRY" in sts and "OK" not in sts else
                      "INSUFFICIENT_MATCHES")
            per_image.append(ImageContribution(eid, label, verified_matches=max(best_m, later),
                                               is_new=is_new, status=status))
            continue
        sh = shifts.get(best_key)
        label = "redundant" if (sh is not None and sh < REDUNDANT_MEDIAN_SHIFT) else "new_view"
        per_image.append(ImageContribution(
            eid, label, best_partner=best_key.split("|")[0], verified_matches=best_m,
            median_shift_fraction=sh, is_new=is_new,
            status="REDUNDANT" if label == "redundant" else "NOVEL_VIEW",
        ))

    return ContributionReport(
        available=True,
        note=("SIFT + RANSAC-verified feature matches; labels are thresholded match counts, "
              "not calibrated quality scores"),
        per_image=per_image,
        components=len(groups),
        largest_component=max(groups.values()) if groups else 0,
        pair_matches=pair_matches,
    )


def _azimuth_spread(azimuths_deg: Sequence[float]) -> float:
    """360 minus the largest empty angular gap between viewing directions."""
    a = sorted(x % 360.0 for x in azimuths_deg)
    if len(a) < 2:
        return 0.0
    gaps = [(a[(i + 1) % len(a)] - a[i]) % 360.0 for i in range(len(a))]
    return 360.0 - max(gaps)


def quality_levels(report: ContributionReport, total_images: int,
                   coverage: Optional[dict] = None) -> List[dict]:
    """Coarse, honest quality bars for the UI. Each entry:
    {name, level: none|low|moderate|good|unknown, basis}. ``unknown`` when
    the metric cannot be computed -- never a made-up number.

    ``coverage`` is engine.pipeline.guidance.coverage_degrees(): degrees of
    the circle the registered cameras cover, and how that was measured."""
    out: List[dict] = []
    if coverage and coverage.get("count", 0) >= 2:
        spread = coverage["degrees"]
        lvl = "good" if spread >= 270 else "moderate" if spread >= 120 else "low"
        how = ("camera positions around the scene" if coverage["mode"] == "around_scene"
               else "viewing directions")
        out.append({"name": "Coverage", "level": lvl,
                    "basis": f"registered {how} span {spread:.0f} of 360 degrees"})
    else:
        out.append({"name": "Coverage", "level": "unknown",
                    "basis": "needs >= 2 registered camera poses to measure viewing-direction spread"})

    if not report.available or total_images == 0:
        for name in ("Overlap", "Spatial diversity"):
            out.append({"name": name, "level": "unknown", "basis": report.note})
    elif total_images == 1:
        out.append({"name": "Overlap", "level": "none",
                    "basis": "single image: no second view to overlap with"})
        out.append({"name": "Spatial diversity", "level": "none", "basis": "single viewpoint"})
    else:
        frac = (report.largest_component or 0) / total_images
        lvl = ("good" if frac >= 0.9 else "moderate" if frac >= 0.6
               else "low" if (report.largest_component or 0) > 1 else "none")
        out.append({"name": "Overlap", "level": lvl,
                    "basis": f"{report.largest_component} of {total_images} images share verified "
                             f"features in one connected group ({report.components} group(s))"})
        distinct = [c for c in report.per_image if c.label in ("new_view", "first")]
        div = len(distinct) / total_images
        lvl = "good" if div >= 0.75 else "moderate" if div >= 0.5 else "low" if div > 0 else "none"
        redundant = sum(1 for c in report.per_image if c.label == "redundant")
        out.append({"name": "Spatial diversity", "level": lvl,
                    "basis": f"{len(distinct)} distinct viewpoint(s); {redundant} near-duplicate image(s)"})
    return out
