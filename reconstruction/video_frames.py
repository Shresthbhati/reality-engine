"""Video as a frame source (plan decision 7): pick sharp, well-spaced keyframes from a walk-through video so the
normal photo pipeline can take over. Streams with cv2.VideoCapture (two sequential passes, never the whole clip
in RAM). ffmpeg is not required.

Selection
  pass 1  per-frame sharpness (capture_quality's Laplacian variance at <=640 px) + timestamp.
  windows `min_gap_s` long; the sharpest frame of each window is the only candidate (a window whose best frame is
          blurred contributes nothing: reason "motion_blur").
  pass 2  a candidate is kept only if the scene moved >= `min_motion_px` since the LAST KEPT frame (phase
          correlation on a <=320 px grey image; shifts are in those analysis pixels). Static stretches give
          "near_duplicate".
  budget  more than min(target_count, max_frames) kept -> evenly thinned (reason "over_budget").

Honesty
  * Thresholds are provisional (see capture_quality) and checked on synthetic video only.
  * shift_px is a global 2-D translation proxy (pans). Large rotations / zooms / parallax make it unreliable;
    when phase correlation has low confidence the frame is judged on grey difference and shift_px is None.
  * No angular sweep / coverage is estimated: that needs camera poses, which exist only after reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from reconstruction import capture_quality as cq

MOTION_SIDE_PX = 320
MIN_GAP_S = 0.25
MAX_FRAMES = 120
MIN_MOTION_PX = 8.0
#: phase-correlation peak below this = not trustworthy; fall back to grey difference
MIN_PHASE_RESPONSE = 0.05
#: a frame also counts as blurred if below this fraction of the clip's median sharpness
RELATIVE_BLUR_FRACTION = 0.5
MIN_USABLE_FRAMES = 8
MIN_DURATION_S = 3.0
#: median shift between kept frames above this fraction of frame width = panning too fast for overlap
TOO_FAST_SHIFT_FRACTION = 0.30
JPEG_QUALITY = 92


class VideoError(ValueError):
    """The video cannot be opened or decoded."""


@dataclass(frozen=True)
class Keyframe:
    index: int
    time_s: float
    path: str | None
    sharpness: float
    shift_px: float | None  # vs the previously kept frame, analysis pixels; None for the first / low confidence


@dataclass
class KeyframeResult:
    source: str
    info: dict
    params: dict
    frames: list[Keyframe]
    rejected: list[dict]
    stats: dict
    diagnostics: list[str] = field(default_factory=list)


def _open(path):
    if cv2 is None:
        raise VideoError("OpenCV is not installed")
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"video file not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise VideoError(f"video could not be opened (unsupported or corrupt): {p}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # apply rotation metadata to decoded frames
    except Exception:  # pragma: no cover
        pass
    return cap


def read_video_info(path) -> dict:
    """{frames, fps, width, height, duration_s, rotation}. width/height are of the DECODED (display-oriented)
    frame; ``frames`` is the container's claim and may be off for some files."""
    cap = _open(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        ok, frame = cap.read()
        if not ok:
            raise VideoError(f"video has no decodable frames: {path}")
        h, w = frame.shape[:2]
    finally:
        cap.release()
    return {"frames": frames, "fps": round(fps, 3), "width": w, "height": h,
            "duration_s": round(frames / fps, 3) if fps > 0 and frames > 0 else None, "rotation": rotation}


def _small_gray(frame, max_side):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = g.shape
    s = max_side / max(h, w)
    if s < 1.0:
        g = cv2.resize(g, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return g


def _frames(path):
    """Yield (index, time_s_or_None, bgr) sequentially; stops at the first undecodable frame."""
    cap = _open(path)
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield i, (ms / 1000.0 if ms and ms > 0 else (0.0 if i == 0 else None)), frame
            i += 1
    finally:
        cap.release()


def _shift(prev, cur) -> tuple[float | None, bool]:
    """(shift_px, reliable). Phase correlation; unreliable (None) when the peak is weak."""
    win = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32), win)
    if resp >= MIN_PHASE_RESPONSE:
        return float(math.hypot(dx, dy)), True
    return None, False


def sample_keyframes(path, *, target_count: int | None = None, min_gap_s: float = MIN_GAP_S,
                     max_frames: int = MAX_FRAMES, blur_threshold: float = cq.BLUR_DROP_BELOW,
                     min_motion_px: float = MIN_MOTION_PX) -> KeyframeResult:
    info = read_video_info(path)
    params = {"target_count": target_count, "min_gap_s": min_gap_s, "max_frames": max_frames,
              "blur_threshold": blur_threshold, "relative_blur_fraction": RELATIVE_BLUR_FRACTION,
              "min_motion_px": min_motion_px, "motion_side_px": MOTION_SIDE_PX}

    # ---- pass 1: sharpness + time per frame (floats only)
    sharp: list[float] = []
    times: list[float | None] = []
    for _i, t, frame in _frames(path):
        sharp.append(float(cv2.Laplacian(_small_gray(frame, cq.ANALYSIS_MAX_SIDE_PX), cv2.CV_64F).var()))
        times.append(t)
    n = len(sharp)
    if n == 0:
        raise VideoError(f"video has no decodable frames: {path}")
    fps = info["fps"] or 0.0
    monotonic = all(t is not None for t in times) and (n < 3 or times[-1] > times[0]) and all(
        b >= a for a, b in zip(times, times[1:]))
    # backends without usable timestamps: constant fps assumed (variable fps then cannot be honoured)
    ts = times if monotonic else [i / fps if fps > 0 else float(i) for i in range(n)]
    duration = ts[-1] - ts[0] if n > 1 else 0.0

    median = float(np.median(sharp))
    blur_floor = max(blur_threshold, RELATIVE_BLUR_FRACTION * median)
    blurred_total = sum(1 for s in sharp if s < blur_floor)

    # ---- windows: sharpest frame of each min_gap_s window
    rejected: list[dict] = []
    cands: set[int] = set()
    windows: dict[int, list[int]] = {}
    for i, t in enumerate(ts):
        windows.setdefault(int((t - ts[0]) // min_gap_s), []).append(i)
    for w in sorted(windows):
        best = max(windows[w], key=lambda i: (sharp[i], -i))
        if sharp[best] < blur_floor:
            rejected.append({"index": best, "reason": f"motion_blur (sharpness {sharp[best]:.0f} < "
                                                       f"{blur_floor:.0f}); no sharp frame in this window"})
        else:
            cands.add(best)

    # ---- pass 2: motion gate against the last KEPT frame
    kept: list[Keyframe] = []
    last, last_t = None, None
    last_cand = max(cands, default=-1)
    for i, _t, frame in _frames(path):
        if i in cands and last_t is not None and ts[i] - last_t < min_gap_s - 1e-9:
            rejected.append({"index": i, "reason": f"too_close (< {min_gap_s:g}s after last kept frame)"})
        elif i in cands:
            g = _small_gray(frame, MOTION_SIDE_PX)
            shift, keep = None, True
            if last is not None:
                if last.shape != g.shape:
                    g = cv2.resize(g, (last.shape[1], last.shape[0]), interpolation=cv2.INTER_AREA)
                shift, reliable = _shift(last, g)
                if reliable:
                    keep = shift >= min_motion_px
                else:  # weak correlation: scene changed a lot or is featureless; judge by grey difference
                    keep = float(np.abs(g.astype(np.float32) - last.astype(np.float32)).mean()) \
                        >= cq.KEYFRAME_MIN_MOTION
            if keep:
                kept.append(Keyframe(i, round(ts[i], 4), None, round(sharp[i], 2),
                                     None if shift is None else round(shift, 2)))
                last, last_t = g, ts[i]
            else:
                rejected.append({"index": i, "reason": f"near_duplicate (moved {shift:.1f}px < "
                                                       f"{min_motion_px:g}px since last kept frame)"})
        if i >= last_cand:
            break

    # ---- budget
    limit = max(1, min(max_frames, target_count or max_frames))
    if len(kept) > limit:
        pick = {int(round(x)) for x in np.linspace(0, len(kept) - 1, limit)}
        for j, k in enumerate(kept):
            if j not in pick:
                rejected.append({"index": k.index, "reason": f"over_budget (limit {limit})"})
        kept = [k for j, k in enumerate(kept) if j in pick]
    rejected.sort(key=lambda r: r["index"])

    shifts = [k.shift_px for k in kept if k.shift_px is not None]
    width = min(MOTION_SIDE_PX, max(info["width"], info["height"]))
    stats = {"frames_decoded": n, "duration_s": round(duration, 3), "kept": len(kept), "rejected": len(rejected),
             "median_sharpness": round(median, 2), "blur_floor": round(blur_floor, 2),
             "frames_below_blur_floor": blurred_total, "windows": len(windows),
             "median_shift_px": round(float(np.median(shifts)), 2) if shifts else None,
             "timestamps_from": "container" if monotonic else "constant_fps_assumed"}
    return KeyframeResult(str(path), info, params, kept, rejected, stats, _diagnose(stats, len(kept), width))


def _diagnose(stats: dict, kept: int, width: int) -> list[str]:
    out = []
    if stats["duration_s"] < MIN_DURATION_S:
        out.append(f"The video is only {stats['duration_s']:.1f}s long. Record at least {MIN_DURATION_S:.0f}s, "
                   "walking slowly around the space.")
    if stats["frames_decoded"] and stats["frames_below_blur_floor"] / stats["frames_decoded"] > 0.4:
        out.append("Over 40% of frames are blurred. Move more slowly, hold the phone steady, and use good light.")
    ms = stats["median_shift_px"]
    if ms is not None and ms > TOO_FAST_SHIFT_FRACTION * width:
        out.append("Consecutive keyframes barely overlap (large camera movement between them): pan more slowly "
                   "so neighbouring frames share at least half their view.")
    if kept < MIN_USABLE_FRAMES:
        out.append(f"Only {kept} usable keyframes were found (need at least {MIN_USABLE_FRAMES}). "
                   "Walk slowly along and around the space so the view keeps changing, then record again.")
    return out


# ------------------------------------------------------------------------------------------ writing -----------
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_keyframes(result_or_path, out_dir, **kwargs) -> KeyframeResult:
    """Save keyframes as JPEG q92 named ``video-<sha8>-f<index>.jpg`` plus ``manifest.json`` (source sha256, fps,
    params, per-frame time). Accepts a KeyframeResult or a video path (then sample_keyframes(**kwargs) runs first).
    Returns the result with ``path`` filled in."""
    res = result_or_path if isinstance(result_or_path, KeyframeResult) else sample_keyframes(result_or_path, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = _sha256(res.source)
    want = {k.index: f"video-{sha[:8]}-f{k.index:06d}.jpg" for k in res.frames}
    last_want = max(want, default=-1)
    for i, _t, frame in _frames(res.source):
        if i in want:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise VideoError(f"could not encode frame {i}")
            (out / want[i]).write_bytes(buf.tobytes())
        if i >= last_want:
            break
    missing = [i for i, nm in want.items() if not (out / nm).is_file()]
    if missing:
        raise VideoError(f"frames {missing} could not be re-read from the video")
    frames = [replace(k, path=str(out / want[k.index])) for k in res.frames]
    done = replace(res, frames=frames)
    manifest = {"source": Path(res.source).name, "source_sha256": sha, "video": res.info, "params": res.params,
                "stats": res.stats, "diagnostics": res.diagnostics, "rejected": res.rejected,
                "frames": [{"file": Path(k.path).name, **{a: b for a, b in asdict(k).items() if a != "path"}}
                           for k in frames]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return done
