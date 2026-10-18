"""C5 perf gate: a 500-building district (25 x 20 grid, default 24 m lots).

Run: python -m benchmarks.district_bench
Prints MEASURED wall-clock times (perf_counter, this machine) and entity /
geometry counts for three detail policies. Nothing here is a budget: the
numbers are recorded in docs/BENCHMARKS.md as measured, not as promises.
"""

from __future__ import annotations

import json
import platform
import statistics
import time

from procedural import spec_store as ss
from procedural.district_grammar import DistrictSpec, generate_district

REPEATS = 3


def _spec(max_full):
    return DistrictSpec(name="bench", blocks_x=25, blocks_y=20, seed=1,
                        detail={"max_full": max_full})


def _time(fn):
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, ts


def main() -> None:
    print(f"python {platform.python_version()} on {platform.platform()} ({platform.processor()})")
    for label, max_full in (("footprint_only (max_full=0)", 0),
                            ("default cap (max_full=25)", 25),
                            ("all full (max_full=None)", None)):
        sp = _spec(max_full)
        world, ts = _time(lambda: generate_district(sp))
        gspec = ss.GeneratedSpec("district", "bench", params=sp.to_params(), seed=1)
        world2, ts2 = _time(lambda: ss.generate(None, gspec))
        n_json = len(json.dumps(world2.to_dict(), sort_keys=True))
        _, ts3 = _time(lambda: ss.regenerate(world2, "bench-district",
                                             {"parcels": {"3,3": {"family": 0}}}))
        print(f"\n{label}: buildings=500 entities={len(world.entities)} "
              f"geometries={len(world.geometries)} worldir_json={n_json / 1e6:.2f} MB")
        print(f"  generate_district       min {min(ts) * 1e3:8.1f} ms  median {statistics.median(ts) * 1e3:8.1f} ms")
        print(f"  spec_store.generate     min {min(ts2) * 1e3:8.1f} ms  median {statistics.median(ts2) * 1e3:8.1f} ms")
        print(f"  regenerate (1 parcel)   min {min(ts3) * 1e3:8.1f} ms  median {statistics.median(ts3) * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
