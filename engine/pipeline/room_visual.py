"""Visual evidence that two rooms are the same room: feature matches between their photographs (M4, second-visit proposals).

Walls, windows, floor texture and fixed furniture do not move between visits, so a second capture of a room shares many verified feature
matches with the first even when other furniture changed. Two different rooms share almost none (the matcher's measured ceiling for
look-alike different scenes is about 8 inliers, see ``reconstruction.scene_identity``). The score is the best pair's inlier count against a
saturation point; it is evidence for the person deciding, never an automatic merge.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

SATURATION_INLIERS = 30   # this many verified inliers in one pair = full visual agreement
MAX_PAIRS = 6


def visual_agreement(paths_a: Sequence[Path], paths_b: Sequence[Path], *, max_pairs: int = MAX_PAIRS) -> Optional[Dict[str, object]]:
    """``{score 0..1, best_inliers, pairs_tested}`` for room A's photos against room B's, or None when either side has no readable photo."""
    from reconstruction.scene_identity import SiftPairVerifier

    a = [Path(p) for p in paths_a if Path(p).is_file()]
    b = [Path(p) for p in paths_b if Path(p).is_file()]
    if not a or not b:
        return None
    pairs = list(product(range(len(a)), range(len(b))))
    pairs.sort(key=lambda ij: (ij[0] + ij[1], ij[0]))          # a spread of pairs, not all of one photo's
    pairs = pairs[:max_pairs]
    ids: Dict[str, Path] = {**{f"a{i}": p for i, p in enumerate(a)}, **{f"b{j}": p for j, p in enumerate(b)}}
    verify = SiftPairVerifier(ids)
    best, tested = 0, 0
    for i, j in pairs:
        try:
            ev = verify(f"a{i}", f"b{j}")
        except Exception:  # noqa: BLE001 -- an unreadable photo is skipped, never a fake zero
            continue
        tested += 1
        best = max(best, int(ev.inliers))
    if tested == 0:
        return None
    return {"score": round(min(1.0, best / SATURATION_INLIERS), 3), "best_inliers": best, "pairs_tested": tested}
