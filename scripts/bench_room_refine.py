"""Benchmark: room-shell refinement before vs after, over random synthetic rooms.

    python scripts/bench_room_refine.py [--seeds 30] [--budget 60] [--points 20000]

Rooms: random box, L-shaped (6 walls), and skewed (one wall 6-12 deg off orthogonal, to exercise the learned
orthogonality prior). Noise 0.4-1% of room size, 5-20% uniform outliers/floaters, one wall 45% uncovered and another
at 25% density, random yaw, scale and offset; the initial planes are tilted 2-5 deg and shifted 2-5% of room size.
Errors are relative to the true room (same dimension definition for truth, input and output).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_room_refine import evaluate, make_room  # noqa: E402


def q(a, p):
    return float(np.nanpercentile(a, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()
    hdr = (f"{'room':<6}{'metric':<22}{'before med':>11}{'before p90':>11}{'after med':>11}{'after p90':>11}")
    print(f"seeds={a.seeds} points={a.points} budget={a.budget}s patience={a.patience}\n{hdr}\n{'-' * len(hdr)}")
    for kind in ("box", "L", "skew"):
        scales = np.exp(np.random.default_rng(1).uniform(math_log(0.003), math_log(900), a.seeds))
        rows = []
        for s in range(a.seeds):
            room = make_room(1000 * (("box", "L", "skew").index(kind) + 1) + s, kind, scale=float(scales[s]),
                             shift=tuple(np.random.default_rng(s).normal(0, 50 * scales[s], 3)), n_pts=a.points)
            rows.append(evaluate(room, seed=s, time_budget_s=a.budget, patience=a.patience))
        col = lambda f: np.array([f(r) for r in rows])  # noqa: E731
        lines = [
            ("length err %", lambda r: r["dims_before"][0], lambda r: r["dims_after"][0]),
            ("width err %", lambda r: r["dims_before"][1], lambda r: r["dims_after"][1]),
            ("height err %", lambda r: r["dims_before"][2], lambda r: r["dims_after"][2]),
            ("mean dim err %", lambda r: r["dim_before"], lambda r: r["dim_after"]),
            ("plane angle err deg", lambda r: r["ang_before"], lambda r: r["ang_after"]),
            ("inlier rms (rel %)", lambda r: 100 * r["res"].rms_before_rel, lambda r: 100 * r["res"].rms_after_rel),
        ]
        for name, fb, fa in lines:
            b, f = col(fb), col(fa)
            print(f"{kind:<6}{name:<22}{q(b, 50):>11.3f}{q(b, 90):>11.3f}{q(f, 50):>11.3f}{q(f, 90):>11.3f}")
        sec = col(lambda r: r["seconds"])
        worse = int(np.sum(col(lambda r: r["dim_after"]) > col(lambda r: r["dim_before"]) + 0.5))
        print(f"{kind:<6}{'runtime s (med/p90/max)':<22}{q(sec, 50):>11.2f}{q(sec, 90):>11.2f}{sec.max():>11.2f}")
        print(f"{kind:<6}converged {sum(r['res'].converged for r in rows)}/{len(rows)}; prior selected "
              f"{int(sum(r['res'].orthogonality['selected'] for r in rows))}/{len(rows)}; mean-dim worse by >0.5pp in "
              f"{worse}/{len(rows)} seeds; rejected planes {sum(p.status != 'refined' for r in rows for p in r['res'].planes)}\n")


def math_log(x: float) -> float:
    import math
    return math.log(x)


if __name__ == "__main__":
    main()
